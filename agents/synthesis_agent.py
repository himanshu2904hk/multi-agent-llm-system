import logging
from context_mgr.schema import SharedContext, ProvenanceEntry
from context_mgr.budget import BudgetManager
from agents.llm_client import chat_json

logger = logging.getLogger(__name__)

AGENT_ID = "synthesis_agent"
BUDGET = 6000

SYSTEM_PROMPT = """You are a synthesis agent. You merge outputs from all sub-agents, resolve contradictions flagged by the critique agent, and produce a final answer.

Rules:
- Each sentence in your final answer must be traceable to a source agent and optionally a chunk_id.
- Contradictions flagged by the critique agent MUST be resolved in your answer (do not surface them to the user).
- Return JSON: {
    "final_answer": "Full answer here...",
    "provenance": [
      {"sentence": "First sentence.", "source_agent": "rag_agent", "source_chunk_id": "abc123"},
      {"sentence": "Second sentence.", "source_agent": "decomposer", "source_chunk_id": null}
    ],
    "contradictions_resolved": ["description of each contradiction resolved"]
  }
"""


def run(context: SharedContext, budget_manager: BudgetManager) -> SharedContext:
    budget_manager.declare_budget(AGENT_ID, BUDGET)
    context.prompts_used[AGENT_ID] = SYSTEM_PROMPT

    # Build synthesis input
    parts = [f"Original query: {context.original_query}\n"]

    if context.rag_answer:
        parts.append(f"RAG Agent answer:\n{context.rag_answer}")

    if context.claim_scores:
        flagged = [c for c in context.claim_scores if c.flagged]
        if flagged:
            flags_text = "\n".join([
                f"- Claim: '{c.claim}' | Flagged span: '{c.flagged_span}' | Reason: {c.reason}"
                for c in flagged
            ])
            parts.append(f"Critique agent flagged these issues (MUST BE RESOLVED):\n{flags_text}")

    if context.critique_summary:
        parts.append(f"Critique summary: {context.critique_summary}")

    if context.retrieved_chunks:
        chunk_list = "\n".join([f"[{c.chunk_id}] (hop {c.hop}) {c.content}" for c in context.retrieved_chunks[:6]])
        parts.append(f"Available chunks for provenance:\n{chunk_list}")

    prompt = "\n\n".join(parts)

    if not budget_manager.check_before_add(AGENT_ID, prompt):
        logger.error(f"[{AGENT_ID}] Budget exceeded")
        context.final_answer = "Budget exceeded during synthesis."
        return context

    budget_manager.consume(AGENT_ID, prompt)

    try:
        result, tokens = chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        budget_manager.consume(AGENT_ID, str(result))

        context.final_answer = result.get("final_answer", "")

        raw_provenance = result.get("provenance", [])
        context.provenance_map = [
            ProvenanceEntry(
                sentence=p.get("sentence", ""),
                source_agent=p.get("source_agent", "unknown"),
                source_chunk_id=p.get("source_chunk_id"),
            )
            for p in raw_provenance
        ]

        logger.info(f"[{AGENT_ID}] Final answer produced with {len(context.provenance_map)} provenance entries")
        context.post_message(
            sender=AGENT_ID,
            recipient="orchestrator",
            content={"final_answer_length": len(context.final_answer), "provenance_entries": len(context.provenance_map)},
            message_type="synthesis_complete",
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error: {e}")
        context.final_answer = context.rag_answer or f"Synthesis failed: {str(e)}"

    return context
