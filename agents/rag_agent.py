import logging
import uuid
from context_mgr.schema import SharedContext, RetrievedChunk
from context_mgr.budget import BudgetManager
from agents.llm_client import chat_json, chat
from tools.web_search import web_search
from tools.sql_lookup import sql_lookup

logger = logging.getLogger(__name__)

AGENT_ID = "rag_agent"
BUDGET = 6000

SYSTEM_PROMPT = """You are a retrieval-augmented generation agent. You perform multi-hop reasoning across retrieved chunks.

Rules:
- You must use at least 2 retrieved chunks before forming an answer.
- For each part of your answer, cite which chunk_id contributed to it.
- Return JSON: {
    "answer": "...",
    "citations": {"sentence or claim": "chunk_id"},
    "reasoning_hops": [{"hop": 1, "chunk_id": "...", "insight": "..."}]
  }
- Be factual. Do not fabricate information not present in the chunks.
"""

HOP2_PROMPT = """You have retrieved initial chunks. Now perform a second retrieval hop to get deeper information.
Based on the initial chunks provided, what additional search query would yield the most relevant follow-up information?
Return JSON: {"followup_query": "..."}
"""


def run(context: SharedContext, budget_manager: BudgetManager) -> SharedContext:
    budget_manager.declare_budget(AGENT_ID, BUDGET)
    context.prompts_used[AGENT_ID] = SYSTEM_PROMPT

    query = context.original_query

    # --- HOP 1: primary retrieval ---
    search_result = web_search(query, top_k=3)
    hop1_chunks = []

    if search_result.success and search_result.data:
        for i, r in enumerate(search_result.data.get("results", [])):
            chunk_id = str(uuid.uuid4())[:8]
            hop1_chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=r["snippet"],
                source=r["url"],
                relevance_score=r["relevance"],
                hop=1,
            ))

    # Try SQL lookup too
    sql_result = sql_lookup(query)
    if sql_result.success and sql_result.data:
        chunk_id = str(uuid.uuid4())[:8]
        hop1_chunks.append(RetrievedChunk(
            chunk_id=chunk_id,
            content=f"Database result: {sql_result.data.get('rows', [])}",
            source="local_database",
            relevance_score=0.75,
            hop=1,
        ))

    # --- HOP 2: follow-up retrieval ---
    hop2_chunks = []
    if hop1_chunks:
        chunks_text = "\n".join([f"[{c.chunk_id}] {c.content}" for c in hop1_chunks])
        try:
            if budget_manager.check_before_add(AGENT_ID, chunks_text):
                budget_manager.consume(AGENT_ID, chunks_text)
                followup_data, _ = chat_json([
                    {"role": "system", "content": HOP2_PROMPT},
                    {"role": "user", "content": f"Initial chunks:\n{chunks_text}\n\nOriginal query: {query}"},
                ])
                followup_query = followup_data.get("followup_query", query + " details")
                hop2_result = web_search(followup_query, top_k=2)
                if hop2_result.success and hop2_result.data:
                    for r in hop2_result.data.get("results", []):
                        chunk_id = str(uuid.uuid4())[:8]
                        hop2_chunks.append(RetrievedChunk(
                            chunk_id=chunk_id,
                            content=r["snippet"],
                            source=r["url"],
                            relevance_score=r["relevance"],
                            hop=2,
                        ))
        except Exception as e:
            logger.warning(f"[{AGENT_ID}] Hop 2 failed: {e}")

    all_chunks = hop1_chunks + hop2_chunks
    context.retrieved_chunks = all_chunks

    if not all_chunks:
        context.rag_answer = "No relevant information retrieved."
        return context

    # --- Generate answer with citations ---
    chunks_text = "\n".join([f"[chunk_id={c.chunk_id}, hop={c.hop}] {c.content}" for c in all_chunks])
    prompt = f"Query: {query}\n\nRetrieved chunks:\n{chunks_text}\n\nProvide your answer with citations."

    if not budget_manager.check_before_add(AGENT_ID, prompt):
        logger.error(f"[{AGENT_ID}] Budget exceeded")
        context.rag_answer = "Budget exceeded."
        return context

    budget_manager.consume(AGENT_ID, prompt)

    try:
        result, tokens = chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        budget_manager.consume(AGENT_ID, str(result))
        context.rag_answer = result.get("answer", "")
        context.rag_citations = result.get("citations", {})

        logger.info(f"[{AGENT_ID}] Answer generated with {len(all_chunks)} chunks, {len(context.rag_citations)} citations")
        context.post_message(
            sender=AGENT_ID,
            recipient="orchestrator",
            content={"answer": context.rag_answer, "citations": context.rag_citations, "chunks": len(all_chunks)},
            message_type="rag_complete",
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error generating answer: {e}")
        context.rag_answer = f"RAG agent error: {str(e)}"

    return context
