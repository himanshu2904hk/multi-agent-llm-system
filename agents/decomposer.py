"""
Decomposer Agent — Breaks queries into typed sub-tasks with explicit dependency graphs.
Enforces execution order: dependent sub-tasks do NOT execute until dependencies resolve.
"""
import logging
from context_mgr.schema import SharedContext, SubTask
from context_mgr.budget import BudgetManager
from agents.llm_client import chat_json

logger = logging.getLogger(__name__)

AGENT_ID = "decomposer"
BUDGET = 4000

SYSTEM_PROMPT = """You are a query decomposition agent. Break complex or ambiguous queries into typed sub-tasks with explicit dependency graphs.

Rules:
- Each sub-task must have: id (string), description, task_type (search|compute|lookup|synthesize|critique), dependencies (list of ids that must complete first).
- Dependencies must reference valid ids in the same list.
- Simple queries: 1-2 tasks with no dependencies.
- Complex/ambiguous queries: 3-6 tasks with clear dependency chains.
- Return ONLY valid JSON: {"subtasks": [...]}

Example for "What is the cheapest product and how does its price compare to the average?":
{
  "subtasks": [
    {"id": "t1", "description": "Find cheapest product", "task_type": "lookup", "dependencies": []},
    {"id": "t2", "description": "Calculate average price of all products", "task_type": "compute", "dependencies": []},
    {"id": "t3", "description": "Compare cheapest to average price", "task_type": "synthesize", "dependencies": ["t1", "t2"]}
  ]
}
"""


def _resolve_execution_order(subtasks: list[SubTask]) -> list[list[SubTask]]:
    """
    Topological sort of subtasks by dependency graph.
    Returns layers: each layer can execute in parallel,
    next layer waits for previous to complete.
    Dependent sub-tasks are BLOCKED until dependencies resolve.
    """
    id_map = {t.id: t for t in subtasks}
    completed = set()
    layers = []
    remaining = list(subtasks)

    max_iterations = len(subtasks) + 1
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1
        ready = []
        still_blocked = []

        for task in remaining:
            deps = task.dependencies
            if all(dep in completed for dep in deps):
                ready.append(task)
            else:
                still_blocked.append(task)

        if not ready:
            # Circular dependency — mark all remaining as unblocked
            logger.warning(f"[{AGENT_ID}] Circular dependency detected, breaking cycle")
            ready = still_blocked
            still_blocked = []

        for task in ready:
            completed.add(task.id)

        if ready:
            layers.append(ready)

        remaining = still_blocked

    return layers


def run(context: SharedContext, budget_manager: BudgetManager) -> SharedContext:
    budget_manager.declare_budget(AGENT_ID, BUDGET)
    context.prompts_used[AGENT_ID] = SYSTEM_PROMPT

    prompt = f"Query: {context.original_query}\n\nDecompose into typed sub-tasks with dependency graph."

    if not budget_manager.check_before_add(AGENT_ID, prompt):
        logger.error(f"[{AGENT_ID}] Budget exceeded before execution")
        return context

    budget_manager.consume(AGENT_ID, prompt)

    try:
        result, tokens = chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
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

        # Enforce dependency graph — resolve execution order
        execution_layers = _resolve_execution_order(subtasks)

        logger.info(f"[{AGENT_ID}] Decomposed into {len(subtasks)} subtasks across {len(execution_layers)} execution layers")
        for i, layer in enumerate(execution_layers):
            layer_ids = [t.id for t in layer]
            logger.info(f"[{AGENT_ID}]   Layer {i+1}: {layer_ids}")
            # Mark tasks in this layer as ready, update status
            for task in layer:
                task.status = "ready"

        # Store subtasks in dependency order (flattened)
        ordered = [task for layer in execution_layers for task in layer]
        context.subtasks = ordered

        # Store execution layers in routing log for observability
        context.routing_log.append({
            "agent": AGENT_ID,
            "execution_layers": [[t.id for t in layer] for layer in execution_layers],
            "total_tasks": len(subtasks),
        })

        context.post_message(
            sender=AGENT_ID,
            recipient="orchestrator",
            content={
                "subtasks": [s.dict() for s in ordered],
                "execution_layers": [[t.id for t in layer] for layer in execution_layers],
            },
            message_type="decomposition_complete",
        )

    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error: {e}")
        context.subtasks = [SubTask(
            id="task_0",
            description=context.original_query,
            task_type="search",
            dependencies=[],
            status="ready",
        )]

    return context
