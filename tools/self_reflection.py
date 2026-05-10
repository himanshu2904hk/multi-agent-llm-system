import time
import re
from typing import List, Dict, Any
from tools.base import ToolResult, FailureMode


def self_reflect(previous_outputs: List[Dict[str, Any]], focus: str = "") -> ToolResult:
    """
    Re-reads previous agent outputs within the session and identifies contradictions.
    previous_outputs: list of {"agent": str, "content": str}
    focus: optional topic to focus contradiction detection on
    """
    start = time.time()

    if not previous_outputs:
        return ToolResult(
            success=False,
            failure_mode=FailureMode.empty_results,
            error_message="No previous outputs provided for reflection.",
            latency_ms=0.0,
        )

    if not isinstance(previous_outputs, list):
        return ToolResult(
            success=False,
            failure_mode=FailureMode.malformed_input,
            error_message="previous_outputs must be a list of dicts with 'agent' and 'content' keys.",
            latency_ms=0.0,
        )

    contradictions = []
    all_claims = []

    for entry in previous_outputs:
        if not isinstance(entry, dict) or "content" not in entry:
            continue
        agent = entry.get("agent", "unknown")
        content = entry.get("content", "")
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        for s in sentences:
            if s:
                all_claims.append({"agent": agent, "text": s})

    # Simple contradiction detection: look for negation patterns between claims
    negation_pairs = [
        ("is not", "is"), ("cannot", "can"), ("does not", "does"),
        ("never", "always"), ("false", "true"), ("incorrect", "correct"),
        ("no ", "yes "), ("doesn't", "does"), ("won't", "will"),
    ]

    for i, claim_a in enumerate(all_claims):
        for j, claim_b in enumerate(all_claims):
            if i >= j:
                continue
            if claim_a["agent"] == claim_b["agent"]:
                continue
            a_text = claim_a["text"].lower()
            b_text = claim_b["text"].lower()
            for neg, pos in negation_pairs:
                if neg in a_text and pos in b_text and neg not in b_text:
                    if focus and focus.lower() not in a_text and focus.lower() not in b_text:
                        continue
                    contradictions.append({
                        "claim_a": {"agent": claim_a["agent"], "text": claim_a["text"]},
                        "claim_b": {"agent": claim_b["agent"], "text": claim_b["text"]},
                        "pattern": f"'{neg}' vs '{pos}'",
                    })
                    break

    latency = (time.time() - start) * 1000
    return ToolResult(
        success=True,
        data={
            "total_claims_analyzed": len(all_claims),
            "contradictions_found": len(contradictions),
            "contradictions": contradictions[:10],
            "focus": focus or "general",
        },
        latency_ms=latency,
    )
