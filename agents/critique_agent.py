import logging
from context_mgr.schema import SharedContext, ClaimScore
from context_mgr.budget import BudgetManager
from agents.llm_client import chat_json

logger = logging.getLogger(__name__)

AGENT_ID = "critique_agent"
BUDGET = 5000

SYSTEM_PROMPT = """You are a critique agent. You review outputs from other agents and assign structured confidence scores per claim.

Rules:
- Do NOT critique the output as a whole — score individual claims.
- For each claim, assign: confidence (0.0-1.0), flagged (true/false), flagged_span (exact text you disagree with, or null), reason (why flagged, or null).
- Return JSON: {
    "claims": [
      {"claim": "...", "confidence": 0.9, "flagged": false, "flagged_span": null, "reason": null},
      {"claim": "...", "confidence": 0.4, "flagged": true, "flagged_span": "exact span here", "reason": "This is unsupported by retrieved chunks"}
    ],
    "summary": "Overall assessment..."
  }
- Be specific about what you disagree with. Never flag without a reason.
"""


def run(context: SharedContext, budget_manager: BudgetManager) -> SharedContext:
    budget_manager.declare_budget(AGENT_ID, BUDGET)
    context.prompts_used[AGENT_ID] = SYSTEM_PROMPT

    # Collect outputs to critique
    outputs_to_review = []
    if context.rag_answer:
        outputs_to_review.append(f"RAG Agent answer:\n{context.rag_answer}")
    for task in context.subtasks:
        if task.result:
            outputs_to_review.append(f"Subtask [{task.id}] result:\n{task.result}")

    if not outputs_to_review:
        logger.warning(f"[{AGENT_ID}] Nothing to critique")
        return context

    combined = "\n\n---\n\n".join(outputs_to_review)
    chunks_context = ""
    if context.retrieved_chunks:
        chunks_context = "\nSupporting chunks:\n" + "\n".join(
            [f"[{c.chunk_id}] {c.content}" for c in context.retrieved_chunks[:5]]
        )

    prompt = f"Review the following agent outputs for accuracy and consistency:\n\n{combined}{chunks_context}"

    if not budget_manager.check_before_add(AGENT_ID, prompt):
        logger.error(f"[{AGENT_ID}] Budget exceeded")
        return context

    budget_manager.consume(AGENT_ID, prompt)

    try:
        result, tokens = chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        budget_manager.consume(AGENT_ID, str(result))

        raw_claims = result.get("claims", [])
        claim_scores = []
        for c in raw_claims:
            claim_scores.append(ClaimScore(
                claim=c.get("claim", ""),
                confidence=float(c.get("confidence", 0.5)),
                flagged=bool(c.get("flagged", False)),
                flagged_span=c.get("flagged_span"),
                reason=c.get("reason"),
            ))

        context.claim_scores = claim_scores
        context.critique_summary = result.get("summary", "")

        flagged = [c for c in claim_scores if c.flagged]
        logger.info(f"[{AGENT_ID}] Reviewed {len(claim_scores)} claims, {len(flagged)} flagged")
        context.post_message(
            sender=AGENT_ID,
            recipient="orchestrator",
            content={"claims_reviewed": len(claim_scores), "flagged": len(flagged), "summary": context.critique_summary},
            message_type="critique_complete",
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error: {e}")
        context.critique_summary = f"Critique failed: {str(e)}"

    return context
