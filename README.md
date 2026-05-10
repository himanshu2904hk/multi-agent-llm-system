# Mega AI — Real-Time Multi-Agent LLM Orchestration System

Production-grade multi-agent system with a self-improving evaluation loop, dynamic tool orchestration, and adversarial robustness testing.

---

## Quick Start (5 minutes)

```bash
git clone <your-repo-url>
cd mega-ai

# Add your Groq API key to .env
echo "GROQ_API_KEY=your_key_here" >> .env

docker compose up --build
```

- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- DB admin: http://localhost:8080 (server: `db`, user: `megaai`, pass: `megaai123`)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT                               │
│         POST /query  ──────────►  SSE Stream                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI (port 8000)                       │
│  /query  /jobs/:id/trace  /eval/latest                      │
│  /eval/rewrites/:id/review  /eval/rerun-failed              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  ORCHESTRATOR                               │
│  • Calls Groq LLM to decide agent execution plan at runtime │
│  • Logs routing decision with justification                 │
│  • Mediates ALL inter-agent handoffs via SharedContext      │
└──┬──────────┬──────────┬──────────┬──────────────────────── ┘
   │          │          │          │
   ▼          ▼          ▼          ▼
DECOMPOSER  RAG       CRITIQUE  SYNTHESIS
   │        AGENT      AGENT      AGENT
   │          │          │          │
   │        ┌─┴──────────┴──────────┘
   │        │       SHARED CONTEXT
   │        │  (typed schema, message bus,
   │        │   token budgets, provenance)
   │        │
   │      TOOLS
   │  ┌───┬───┬───┬───┐
   │  │WEB│SQL│COD│REF│
   │  │SRC│LKP│EXC│LCT│
   │  └───┴───┴───┴───┘
   │
EVAL HARNESS
   │
META AGENT ──► Propose rewrite ──► Human approval ──► Re-eval
```

---

## Agents

### Orchestrator
**Decision boundary**: Reads the query, calls Groq LLM with a structured routing prompt to decide which agents to invoke, in what order, and why. Routing is never hardcoded. Every routing decision is logged with agent name, reason, and context budget. Mediates all handoffs via `SharedContext` — agents never call each other.

### Decomposer
**Decision boundary**: Breaks the input query into typed sub-tasks (`search`, `compute`, `lookup`, `synthesize`, `critique`) with explicit dependency graphs. Dependent tasks are blocked until their dependencies resolve. Returns structured JSON with subtask IDs, descriptions, and dependency lists.

### RAG Agent
**Decision boundary**: Performs two retrieval hops — first retrieves top-3 chunks via web search and SQL lookup, then generates a follow-up query based on those chunks and retrieves again. Must cite which chunk_id contributed to each part of the answer. Single-hop retrieval is rejected by design.

### Critique Agent
**Decision boundary**: Reviews outputs from all other agents. Assigns a per-claim confidence score (0.0–1.0), flags specific spans of text it disagrees with (not the output as a whole), and provides a reason. Never flags without a reason.

### Synthesis Agent
**Decision boundary**: Merges all sub-agent outputs into a final answer. Contradictions flagged by the critique agent are resolved internally — they are never surfaced to the user. Produces a provenance map linking each sentence to its source agent and chunk_id.

### Compression Agent
**Decision boundary**: Invoked automatically when any agent's assembled context would exceed its declared budget. Preserves all structured data (citations, scores, chunk IDs, numbers) and applies lossy compression only to conversational prose.

### Meta Agent
**Decision boundary**: After each eval run, reads failure cases, identifies the worst-performing agent/dimension by average score, and proposes a rewritten prompt with a unified diff and justification. Proposed rewrites are stored but never auto-applied. A human must approve via `POST /eval/rewrites/:id/review`.

---

## Tools

| Tool | Purpose | Failure modes handled |
|------|---------|----------------------|
| `web_search` | Returns structured results with URLs and relevance scores | timeout, empty_results, malformed_input |
| `execute_python` | Runs Python snippets, returns stdout/stderr/exit_code | timeout, execution_error, malformed_input (dangerous patterns blocked) |
| `sql_lookup` | Natural language → SQL → local SQLite query | malformed_input, empty_results, execution_error |
| `self_reflect` | Re-reads previous outputs, detects contradictions via negation patterns | empty_results, malformed_input |

Each tool returns a `ToolResult` with `success`, `data`, `failure_mode`, `error_message`, and `latency_ms`. The orchestrator handles each failure mode differently (not via prompt instructions — via explicit code logic).

---

## API Endpoints

### `POST /query`
Submit a query. Returns an SSE stream with real-time agent activity.

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

SSE events: `orchestrator_start`, `routing_decision`, `agent_start`, `agent_complete`, `pipeline_complete`, `error`, `done`

### `GET /jobs/{job_id}/trace`
Full execution trace: exact sequence of agent decisions, tool calls, token counts, and policy violations.

### `GET /eval/latest`
Latest eval run summary broken down by category (baseline/ambiguous/adversarial) and all 6 scoring dimensions.

### `POST /eval/rewrites/{rewrite_id}/review`
Approve or reject a pending prompt rewrite. Approved rewrites are immediately applied to the pipeline.

```json
{"approved": true, "reviewer_note": "Looks good"}
```

### `POST /eval/rerun-failed`
Re-runs only the previously failed test cases using the latest approved prompts, returns performance deltas.

### `POST /eval/run` *(bonus)*
Triggers a full 15-case evaluation run.

---

## Evaluation

15 test cases across 3 categories:
- **5 baseline**: Known correct answers, tests basic retrieval and reasoning
- **5 ambiguous**: Underspecified queries, tests decomposition quality
- **5 adversarial**: Prompt injections (2), wrong premises (3), tests robustness

6 scoring dimensions (each produces a numeric score + written justification):
1. **Correctness** — factual accuracy vs expected answer
2. **Citation accuracy** — citations present and accurate
3. **Contradiction resolution** — flagged contradictions resolved, not surfaced
4. **Tool efficiency** — penalizes unnecessary tool calls
5. **Budget compliance** — penalizes policy violations
6. **Critique agreement** — final answer aligns with critique agent assessment

Every run is stored in PostgreSQL with exact prompts, tool calls, outputs, scores, and timestamps. Re-running on the same inputs produces a diff-able output.

---

## Self-Improving Loop

```
Eval run completes
       │
       ▼
Meta agent reads failures
       │
       ▼
Identifies worst agent/dimension
       │
       ▼
Proposes rewritten prompt (stored, not applied)
       │
       ▼
Human reviews via POST /eval/rewrites/:id/review
       │
    approved?
   ┌───┴───┐
  YES      NO
   │        │
   ▼        ▼
Apply    Reject (logged)
prompt
   │
   ▼
POST /eval/rerun-failed
   │
   ▼
Delta logged in DB
```

Every proposed rewrite, approval/rejection, and performance delta is stored with timestamps and queryable via the DB.

---

## Known Limitations

1. **Web search is a stub** — returns pre-seeded results. A real deployment would use Serper, Brave Search, or Bing API.
2. **RAG has no vector store** — retrieval is keyword-based. Production would use pgvector or Pinecone for semantic search.
3. **Code execution sandbox is partial** — dangerous pattern detection is heuristic. A real sandbox would use Docker-in-Docker or Firecracker microVMs.
4. **Self-reflection contradiction detection is rule-based** — simple negation pattern matching. LLM-based contradiction detection would be more accurate.
5. **Meta-agent proposes one rewrite per run** — a more sophisticated loop would rank all agents/dimensions and batch proposals.
6. **No authentication** — all endpoints are open. Production would require API keys or OAuth.
7. **Worker and API share the same Groq rate limits** — under load, the worker may be throttled.

---

## What I Would Build Next

1. **Vector-based RAG** with pgvector — real semantic retrieval instead of keyword stubs
2. **Streaming token-by-token from Groq** — currently streams events, not individual tokens
3. **Agent memory** — persistent context across sessions using Redis
4. **Rate limit handling** — exponential backoff with Groq's 429 responses
5. **Webhook notifications** — push job completion to a client URL instead of requiring SSE connection to stay open
6. **Prompt versioning system** — git-like history of all prompt changes with rollback

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Groq model ID | `llama-3.3-70b-versatile` |
| `DATABASE_URL` | PostgreSQL connection string | set by Docker Compose |
| `REDIS_URL` | Redis connection string | set by Docker Compose |
| `LOG_LEVEL` | Logging level | `INFO` |

No credentials are hardcoded anywhere. All configuration is via environment variables.
