import logging
from agents.llm_client import chat_json
from context_mgr.schema import SharedContext

logger = logging.getLogger(__name__)

SCORING_PROMPT = """You are an evaluation judge. Score the given answer on the specified dimension.

Return JSON:
{
  "score": 0.0-1.0,
  "justification": "One sentence explaining the score."
}

Scoring guidelines per dimension:
- correctness: Is the answer factually correct relative to the expected answer? (1.0 = fully correct, 0.0 = completely wrong)
- citation_accuracy: Did the answer cite sources and are those citations accurate? (1.0 = all citations accurate, 0.0 = no citations or all wrong)
- contradiction_resolution: Were contradictions resolved cleanly? (1.0 = all resolved, 0.0 = contradictions surfaced to user)
- tool_efficiency: Were tools used efficiently? Penalize unnecessary calls. (1.0 = optimal, 0.0 = grossly inefficient)
- budget_compliance: Did agents stay within token budgets? (1.0 = no violations, 0.0 = multiple violations)
- critique_agreement: Does the critique agent agree with the final answer? (1.0 = full agreement, 0.0 = major disagreement)
"""


def score_dimension(
    dimension: str,
    query: str,
    expected: str,
    actual: str,
    context: SharedContext,
    extra_context: str = "",
) -> tuple[float, str]:
    prompt_parts = [
        f"Dimension: {dimension}",
        f"Query: {query}",
        f"Expected answer: {expected or 'N/A (ambiguous query)'}",
        f"Actual answer: {actual or 'No answer produced'}",
    ]

    if dimension == "citation_accuracy" and context.rag_citations:
        prompt_parts.append(f"Citations provided: {context.rag_citations}")

    if dimension == "contradiction_resolution" and context.claim_scores:
        flagged = [c for c in context.claim_scores if c.flagged]
        prompt_parts.append(f"Flagged claims: {len(flagged)}")
        prompt_parts.append(f"Critique summary: {context.critique_summary or 'N/A'}")

    if dimension == "tool_efficiency":
        # Count tool calls from routing log
        prompt_parts.append(f"Agents invoked: {[p.get('agent') for p in context.routing_log]}")

    if dimension == "budget_compliance":
        violations = context.policy_violations
        prompt_parts.append(f"Policy violations: {violations}")

    if dimension == "critique_agreement" and context.claim_scores:
        avg_conf = sum(c.confidence for c in context.claim_scores) / len(context.claim_scores)
        flagged_count = sum(1 for c in context.claim_scores if c.flagged)
        prompt_parts.append(f"Average claim confidence: {avg_conf:.2f}, flagged: {flagged_count}/{len(context.claim_scores)}")

    if extra_context:
        prompt_parts.append(f"Extra context: {extra_context}")

    try:
        result, _ = chat_json([
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ])
        return float(result.get("score", 0.5)), result.get("justification", "")
    except Exception as e:
        logger.error(f"[scorer] {dimension} error: {e}")
        return 0.5, f"Scoring error: {str(e)}"


def score_all_dimensions(
    query: str,
    expected: str,
    actual: str,
    context: SharedContext,
) -> dict:
    dimensions = [
        "correctness",
        "citation_accuracy",
        "contradiction_resolution",
        "tool_efficiency",
        "budget_compliance",
        "critique_agreement",
    ]
    scores = {}
    for dim in dimensions:
        score, justification = score_dimension(dim, query, expected, actual, context)
        scores[dim] = {"score": score, "justification": justification}
        logger.debug(f"[scorer] {dim}={score:.2f}: {justification}")

    overall = sum(v["score"] for v in scores.values()) / len(scores)
    scores["overall"] = {"score": overall, "justification": f"Average of {len(scores)-1} dimensions"}
    return scores
