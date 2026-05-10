import logging
from context_mgr.schema import SharedContext, SubTask
from context_mgr.budget import BudgetManager
from agents.llm_client import chat_json

logger = logging.getLogger(__name__)

AGENT_ID = "decomposer"
BUDGET = 4000

SYSTEM_PROMPT = """You are a query decomposition agent. Your job is to break ambiguous or complex queries into typed sub-tasks with explicit dependency graphs.

Rules:
- Each sub-task must have a unique id, a description, a task_type (one of: search, compute, lookup, synthesize, critique), and a list of dependency ids.
- Dependent sub-tasks must not list ids that don't exist.
- Return ONLY valid JSON with key "subtasks" as an array.
- Each subtask: {id, description, task_type, dependencies: []}
- If the query is simple, return 1-3 subtasks.
- If complex/ambiguous, return 3-6 subtasks with clear dependencies.
"""


def run(context: SharedContext, budget_manager: BudgetManager) -> SharedContext:
    budget_manager.declare_budget(AGENT_ID, BUDGET)

    prompt = f"Query: {context.original_query}\n\nDecompose this into typed sub-tasks with dependencies."
    context.prompts_used[AGENT_ID] = SYSTEM_PROMPT

    if not budget_manager.check_before_add(AGENT_ID, prompt):
        logger.error(f"[{AGENT_ID}] Budget exceeded before execution")
        return context

    budget_manager.consume(AGENT_ID, prompt)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        result, tokens = chat_json(messages)
        budget_manager.consume(AGENT_ID, str(result))

        raw_tasks = result.get("subtasks", [])
        subtasks = []
        for t in raw_tasks:
            subtasks.append(SubTask(
                id=t.get("id", f"task_{len(subtasks)}"),
                description=t.get("description", ""),
                task_type=t.get("task_type", "search"),
                dependencies=t.get("dependencies", []),
                status="pending",
            ))

        context.subtasks = subtasks
        logger.info(f"[{AGENT_ID}] Decomposed into {len(subtasks)} subtasks")
        context.post_message(
            sender=AGENT_ID,
            recipient="orchestrator",
            content={"subtasks": [s.dict() for s in subtasks]},
            message_type="decomposition_complete",
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error: {e}")
        context.subtasks = [SubTask(
            id="task_0",
            description=context.original_query,
            task_type="search",
            dependencies=[],
        )]

    return context
