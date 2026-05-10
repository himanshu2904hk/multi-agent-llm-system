# AI Collaboration Attestation

This document discloses how AI tools were used in building this project, as required by the assessment guidelines.

---

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| **Claude (Anthropic)** | claude-sonnet-4-6 | Primary development assistant |
| **Groq API** | llama-3.3-70b-versatile | LLM backend for all agents at runtime |

---

## How Claude Was Used

### Architecture Design
Claude assisted in designing the overall system architecture including:
- The shared `SharedContext` schema for inter-agent communication
- The token budget manager design with policy violation logging
- The 6-dimensional evaluation scoring framework

### Code Generation
Claude generated boilerplate and scaffolding for:
- FastAPI endpoint structure and SSE streaming setup
- SQLAlchemy model definitions
- Docker Compose configuration
- LangChain + ChromaDB integration setup

### What I Designed / Reviewed Myself
- The dependency graph topological sort logic in the decomposer
- The failure mode handling strategy (timeout vs empty vs malformed)
- The adversarial test case design (prompt injections, wrong premises)
- The self-improving prompt loop audit trail design
- All tool failure contracts and retry logic
- The meta-agent scoring dimension → agent mapping

---

## Where AI Assistance Stops

The following were **not** AI-generated and reflect my own engineering decisions:

1. **The evaluation dimensions** — choosing correctness, citation accuracy, contradiction resolution, tool efficiency, budget compliance, and critique agreement as meaningful metrics for this specific system
2. **Adversarial test case design** — the specific prompt injections and wrong-premise queries were crafted to test real failure modes
3. **The failure contract design** — deciding that `malformed_input` should not retry while `timeout` and `empty_results` should, with different input modifications
4. **Known limitations** — the honest assessment of where the system breaks was written by me

---

## Validation

All AI-generated code was:
- Read and understood before inclusion
- Tested by running the full Docker Compose stack
- Verified against the assessment requirements manually

---

## Signal for Reviewers

If you are running AI-collaboration detection:
- All agent prompt strings are in the respective agent files (`agents/decomposer.py`, etc.)
- The `prompts_used` field in `SharedContext` stores the exact prompt used per agent per run
- Every eval run stores `exact_prompts` in `EvalResult` for full reproducibility
