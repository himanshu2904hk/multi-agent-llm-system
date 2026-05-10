import logging
from typing import Optional
from context_mgr.schema import SharedContext

logger = logging.getLogger(__name__)

# Rough token estimator (4 chars ~ 1 token)
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class BudgetManager:
    def __init__(self, context: SharedContext):
        self.context = context

    def declare_budget(self, agent_id: str, max_tokens: int):
        self.context.token_budgets[agent_id] = {"budget": max_tokens, "used": 0}
        logger.info(f"[budget] {agent_id} declared budget={max_tokens}")

    def remaining(self, agent_id: str) -> int:
        entry = self.context.token_budgets.get(agent_id)
        if not entry:
            return 0
        return entry["budget"] - entry["used"]

    def consume(self, agent_id: str, text: str) -> bool:
        tokens = estimate_tokens(text)
        entry = self.context.token_budgets.get(agent_id)
        if not entry:
            logger.warning(f"[budget] {agent_id} has no declared budget")
            return True

        if entry["used"] + tokens > entry["budget"]:
            violation = f"{agent_id} exceeded budget: used={entry['used']} + new={tokens} > budget={entry['budget']}"
            logger.error(f"[budget] POLICY VIOLATION: {violation}")
            self.context.policy_violations.append(violation)
            return False

        entry["used"] += tokens
        logger.debug(f"[budget] {agent_id} used {tokens} tokens, total={entry['used']}/{entry['budget']}")
        return True

    def check_before_add(self, agent_id: str, text: str) -> bool:
        tokens = estimate_tokens(text)
        return self.remaining(agent_id) >= tokens

    def summary(self) -> dict:
        return {
            agent_id: {
                "budget": v["budget"],
                "used": v["used"],
                "remaining": v["budget"] - v["used"],
                "pct_used": round(v["used"] / v["budget"] * 100, 1) if v["budget"] > 0 else 0,
            }
            for agent_id, v in self.context.token_budgets.items()
        }
