import logging
import difflib
from typing import List
from agents.llm_client import chat_json
from db.models import EvalResult, PromptRewrite, EvalRun
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

AGENT_ID = "meta_agent"

META_PROMPT = """You are a meta-agent that improves LLM pipelines by analyzing failure cases.

Given failing test cases and their scores, identify:
1. Which agent's prompt caused the worst failures
2. What specific dimension suffered most
3. Propose a rewritten prompt with clear improvements

Return JSON:
{
  "worst_agent": "agent_id",
  "worst_dimension": "dimension_name",
  "analysis": "Why this agent/dimension failed...",
  "original_prompt_summary": "Brief description of the current prompt",
  "proposed_changes": ["Change 1: ...", "Change 2: ..."],
  "proposed_prompt": "Full rewritten prompt text...",
  "justification": "Why this rewrite will improve performance..."
}
"""


def _score_dimension(result: EvalResult) -> tuple[str, float]:
    """Return the worst-scoring dimension for this result."""
    dims = {
        "correctness": result.correctness_score or 0,
        "citation_accuracy": result.citation_accuracy_score or 0,
        "contradiction_resolution": result.contradiction_resolution_score or 0,
        "tool_efficiency": result.tool_efficiency_score or 0,
        "budget_compliance": result.budget_compliance_score or 0,
        "critique_agreement": result.critique_agreement_score or 0,
    }
    worst = min(dims, key=dims.get)
    return worst, dims[worst]


DIMENSION_TO_AGENT = {
    "correctness": "rag_agent",
    "citation_accuracy": "rag_agent",
    "contradiction_resolution": "synthesis_agent",
    "tool_efficiency": "orchestrator",
    "budget_compliance": "orchestrator",
    "critique_agreement": "critique_agent",
}

CURRENT_PROMPTS = {}


def register_prompt(agent_id: str, prompt: str):
    CURRENT_PROMPTS[agent_id] = prompt


def analyze_and_propose(db: Session, eval_run_id: str) -> List[PromptRewrite]:
    failed = db.query(EvalResult).filter(
        EvalResult.run_id == eval_run_id,
        EvalResult.passed == False,
    ).all()

    if not failed:
        logger.info(f"[{AGENT_ID}] No failures in run {eval_run_id}")
        return []

    # Aggregate worst dimension
    dim_scores: dict[str, list[float]] = {}
    for r in failed:
        worst_dim, score = _score_dimension(r)
        dim_scores.setdefault(worst_dim, []).append(score)

    worst_dim = min(dim_scores, key=lambda d: sum(dim_scores[d]) / len(dim_scores[d]))
    worst_agent = DIMENSION_TO_AGENT.get(worst_dim, "rag_agent")
    original_prompt = CURRENT_PROMPTS.get(worst_agent, "No prompt registered")

    failure_summaries = []
    for r in failed[:5]:
        failure_summaries.append(
            f"Query: {r.query}\n"
            f"Expected: {r.expected_answer or 'N/A'}\n"
            f"Got: {r.actual_answer or 'N/A'}\n"
            f"Worst dim: {_score_dimension(r)[0]} = {_score_dimension(r)[1]:.2f}"
        )

    prompt_input = (
        f"Worst agent: {worst_agent}\nWorst dimension: {worst_dim}\n\n"
        f"Current prompt:\n{original_prompt}\n\n"
        f"Failure cases:\n" + "\n---\n".join(failure_summaries)
    )

    try:
        result, _ = chat_json([
            {"role": "system", "content": META_PROMPT},
            {"role": "user", "content": prompt_input},
        ])

        proposed = result.get("proposed_prompt", "")
        diff = "\n".join(difflib.unified_diff(
            original_prompt.splitlines(),
            proposed.splitlines(),
            fromfile="original",
            tofile="proposed",
            lineterm="",
        ))

        rewrite = PromptRewrite(
            eval_run_id=eval_run_id,
            agent_id=worst_agent,
            dimension=worst_dim,
            original_prompt=original_prompt,
            proposed_prompt=proposed,
            diff=diff,
            justification=result.get("justification", ""),
            status="pending",
        )
        db.add(rewrite)
        db.commit()
        db.refresh(rewrite)
        logger.info(f"[{AGENT_ID}] Proposed rewrite for {worst_agent}/{worst_dim} (id={rewrite.id})")
        return [rewrite]
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error proposing rewrite: {e}")
        return []
