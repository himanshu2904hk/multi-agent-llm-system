import logging
import time
import asyncio
from typing import AsyncGenerator, Callable
from context_mgr.schema import SharedContext
from context_mgr.budget import BudgetManager
from agents.llm_client import chat_json, chat_stream
from agents import decomposer, rag_agent, critique_agent, synthesis_agent

logger = logging.getLogger(__name__)

AGENT_ID = "orchestrator"
BUDGET = 3000

ROUTING_PROMPT = """You are the master orchestrator of a multi-agent LLM pipeline. Given the query and current context state, decide which agents to invoke, in what order, and why.

Available agents:
- decomposer: Breaks the query into subtasks with dependencies
- rag_agent: Retrieves relevant information and performs multi-hop reasoning
- critique_agent: Reviews outputs and flags issues
- synthesis_agent: Merges all outputs into a final answer

Rules:
- You must always end with synthesis_agent.
- critique_agent must run after at least one other agent has produced output.
- Return JSON: {
    "plan": [
      {"agent": "decomposer", "reason": "...", "context_budget": 4000},
      {"agent": "rag_agent", "reason": "...", "context_budget": 6000},
      {"agent": "critique_agent", "reason": "...", "context_budget": 5000},
      {"agent": "synthesis_agent", "reason": "...", "context_budget": 6000}
    ]
  }
"""

COMPRESSION_PROMPT = """Summarize the following context to reduce token count.
Rules:
- Preserve ALL structured data: tool outputs, scores, citations, chunk IDs, numerical values
- You may compress conversational filler and verbose prose
- Return JSON: {"compressed": "..."}
"""

AGENT_MAP = {
    "decomposer": decomposer.run,
    "rag_agent": rag_agent.run,
    "critique_agent": critique_agent.run,
    "synthesis_agent": synthesis_agent.run,
}


async def run_pipeline(
    context: SharedContext,
    emit: Callable[[str, dict], None] = None,
) -> SharedContext:
    budget_manager = BudgetManager(context)
    budget_manager.declare_budget(AGENT_ID, BUDGET)

    def _emit(event_type: str, data: dict):
        if emit:
            emit(event_type, data)

    _emit("orchestrator_start", {"job_id": context.job_id, "query": context.original_query})

    # --- Routing decision ---
    routing_prompt = f"Query: {context.original_query}\n\nWhat is the optimal agent execution plan?"
    budget_manager.consume(AGENT_ID, routing_prompt)

    try:
        routing_result, _ = chat_json([
            {"role": "system", "content": ROUTING_PROMPT},
            {"role": "user", "content": routing_prompt},
        ])
        plan = routing_result.get("plan", [])
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Routing failed: {e}, using default plan")
        plan = [
            {"agent": "decomposer", "reason": "default", "context_budget": 4000},
            {"agent": "rag_agent", "reason": "default", "context_budget": 6000},
            {"agent": "critique_agent", "reason": "default", "context_budget": 5000},
            {"agent": "synthesis_agent", "reason": "default", "context_budget": 6000},
        ]

    context.routing_log = plan
    _emit("routing_decision", {"plan": plan})
    logger.info(f"[{AGENT_ID}] Plan: {[p['agent'] for p in plan]}")

    # --- Execute each agent in order ---
    for step in plan:
        agent_name = step["agent"]
        reason = step.get("reason", "")
        agent_fn = AGENT_MAP.get(agent_name)

        if not agent_fn:
            logger.warning(f"[{AGENT_ID}] Unknown agent: {agent_name}, skipping")
            continue

        _emit("agent_start", {
            "agent": agent_name,
            "reason": reason,
            "budget_summary": budget_manager.summary(),
        })
        logger.info(f"[{AGENT_ID}] Invoking {agent_name}: {reason}")

        start = time.time()
        try:
            # Run synchronously in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            context = await loop.run_in_executor(None, agent_fn, context, budget_manager)
        except Exception as e:
            logger.error(f"[{AGENT_ID}] Agent {agent_name} raised: {e}")
            context.policy_violations.append(f"{agent_name} crashed: {str(e)}")

        latency = (time.time() - start) * 1000

        _emit("agent_complete", {
            "agent": agent_name,
            "latency_ms": latency,
            "budget_summary": budget_manager.summary(),
            "policy_violations": context.policy_violations,
        })

    # --- Stream final answer token by token via SSE ---
    if context.final_answer and emit:
        tokens_buffer = []
        try:
            stream_messages = [
                {"role": "system", "content": "You are a synthesis agent. Present the following answer clearly and concisely."},
                {"role": "user", "content": f"Present this answer:\n{context.final_answer}"}
            ]
            streamed_answer = ""
            for token in chat_stream(stream_messages, max_tokens=1024):
                streamed_answer += token
                emit("token", {"agent": "synthesis_agent", "token": token})
            if streamed_answer:
                context.final_answer = streamed_answer
        except Exception as e:
            logger.warning(f"[{AGENT_ID}] Token streaming failed: {e}, using buffered answer")

    # --- Emit pipeline complete ---
    _emit("pipeline_complete", {
        "job_id": context.job_id,
        "final_answer": context.final_answer,
        "provenance_entries": len(context.provenance_map),
        "policy_violations": context.policy_violations,
        "budget_summary": budget_manager.summary(),
    })

    return context
