import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db, init_db
from db.models import Job, JobStatus, AgentEvent, EvalRun, EvalResult, PromptRewrite
from context_mgr.schema import SharedContext
from agents.orchestrator import run_pipeline
from eval.harness import run_eval
from agents import meta_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mega AI - Multi-Agent Orchestration System",
    description="Production-grade multi-agent LLM system with self-improving evaluation loop",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    try:
        init_db()
    except Exception as e:
        logger.error(f"init_db error (non-fatal): {e}")
    try:
        from agents import decomposer, rag_agent, critique_agent, synthesis_agent
        meta_agent.register_prompt("decomposer", decomposer.SYSTEM_PROMPT)
        meta_agent.register_prompt("rag_agent", rag_agent.SYSTEM_PROMPT)
        meta_agent.register_prompt("critique_agent", critique_agent.SYSTEM_PROMPT)
        meta_agent.register_prompt("synthesis_agent", synthesis_agent.SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"Agent registration error (non-fatal): {e}")
    logger.info("Mega AI started")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {"example": {"query": "What is machine learning?"}}


class ApprovalRequest(BaseModel):
    approved: bool
    reviewer_note: str = ""


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: str = None


# ─────────────────────────────────────────────
# ENDPOINT 1: Submit query → SSE streaming
# ─────────────────────────────────────────────

@app.post(
    "/query",
    summary="Submit a query and receive streaming SSE response",
    response_description="Server-Sent Events stream with real-time agent activity",
)
async def submit_query(request: QueryRequest, db: Session = Depends(get_db)):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail={
            "error_code": "EMPTY_QUERY",
            "message": "Query must be a non-empty string.",
            "job_id": None,
        })

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, query=request.query, status=JobStatus.pending)
    db.add(job)
    db.commit()

    async def event_stream() -> AsyncGenerator[str, None]:
        events_queue: asyncio.Queue = asyncio.Queue()

        def emit(event_type: str, data: dict):
            data["event_type"] = event_type
            data["timestamp"] = datetime.utcnow().isoformat()
            data["job_id"] = job_id

            # Structured logging: input_hash + output_hash per event
            input_str = json.dumps(data, sort_keys=True, default=str)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
            data["input_hash"] = input_hash

            events_queue.put_nowait(data)

            # Persist to DB with hashes
            try:
                agent_id = data.get("agent", "orchestrator")
                output_str = json.dumps(data.get("final_answer", ""), default=str)
                output_hash = hashlib.sha256(output_str.encode()).hexdigest()[:16]
                ev = AgentEvent(
                    job_id=job_id,
                    agent_id=agent_id,
                    event_type=event_type,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    input_data=data,
                    output_data=None,
                    token_count=data.get("token_count"),
                    latency_ms=data.get("latency_ms"),
                )
                db.add(ev)
                db.commit()
            except Exception:
                pass

        async def run_and_signal():
            try:
                job.status = JobStatus.running
                db.commit()
                context = SharedContext(job_id=job_id, original_query=request.query)
                context = await run_pipeline(context, emit=emit)
                job.status = JobStatus.completed
                job.final_answer = context.final_answer
                job.provenance_map = [p.dict() for p in context.provenance_map]
                job.policy_violations = context.policy_violations
                job.completed_at = datetime.utcnow()
                db.commit()
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                job.status = JobStatus.failed
                db.commit()
                emit("error", {"message": str(e)})
            finally:
                events_queue.put_nowait(None)  # sentinel

        task = asyncio.create_task(run_and_signal())

        while True:
            item = await events_queue.get()
            if item is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"event: {item.get('event_type', 'message')}\ndata: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────
# ENDPOINT 2: Get full execution trace by job ID
# ─────────────────────────────────────────────

@app.get(
    "/jobs/{job_id}/trace",
    summary="Retrieve full execution trace for a completed job",
)
def get_trace(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail={
            "error_code": "JOB_NOT_FOUND",
            "message": f"Job {job_id} not found.",
            "job_id": job_id,
        })

    events = (
        db.query(AgentEvent)
        .filter(AgentEvent.job_id == job_id)
        .order_by(AgentEvent.timestamp)
        .all()
    )

    from db.models import ToolCallLog
    tool_calls = (
        db.query(ToolCallLog)
        .filter(ToolCallLog.job_id == job_id)
        .order_by(ToolCallLog.timestamp)
        .all()
    )

    return {
        "job_id": job_id,
        "query": job.query,
        "status": job.status,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "final_answer": job.final_answer,
        "provenance_map": job.provenance_map,
        "policy_violations": job.policy_violations,
        "events": [
            {
                "id": e.id,
                "agent_id": e.agent_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "latency_ms": e.latency_ms,
                "token_count": e.token_count,
                "policy_violation": e.policy_violation,
                "data": e.input_data,
            }
            for e in events
        ],
        "tool_calls": [
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "tool_name": t.tool_name,
                "attempt": t.attempt,
                "input": t.input_data,
                "output": t.output_data,
                "latency_ms": t.latency_ms,
                "accepted": t.accepted,
                "failure_mode": t.failure_mode,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in tool_calls
        ],
    }


# ─────────────────────────────────────────────
# ENDPOINT 3: Latest eval run summary
# ─────────────────────────────────────────────

@app.get(
    "/eval/latest",
    summary="Retrieve the latest eval run summary broken down by category and dimension",
)
def get_latest_eval(db: Session = Depends(get_db)):
    run = db.query(EvalRun).order_by(EvalRun.triggered_at.desc()).first()
    if not run:
        raise HTTPException(status_code=404, detail={
            "error_code": "NO_EVAL_RUNS",
            "message": "No evaluation runs found. Trigger one via POST /eval/run.",
            "job_id": None,
        })

    results = db.query(EvalResult).filter(EvalResult.run_id == run.id).all()

    by_category = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "cases": []}
        by_category[cat]["total"] += 1
        if r.passed:
            by_category[cat]["passed"] += 1
        by_category[cat]["cases"].append({
            "id": r.test_case_id,
            "query": r.query,
            "passed": r.passed,
            "scores": {
                "correctness": r.correctness_score,
                "citation_accuracy": r.citation_accuracy_score,
                "contradiction_resolution": r.contradiction_resolution_score,
                "tool_efficiency": r.tool_efficiency_score,
                "budget_compliance": r.budget_compliance_score,
                "critique_agreement": r.critique_agreement_score,
            },
        })

    return {
        "run_id": run.id,
        "triggered_at": run.triggered_at.isoformat(),
        "prompt_version": run.prompt_version,
        "total": run.total_cases,
        "passed": run.passed,
        "failed": run.failed,
        "pass_rate": round(run.passed / run.total_cases, 3) if run.total_cases else 0,
        "averages": {
            "correctness": run.avg_correctness,
            "citation_accuracy": run.avg_citation_accuracy,
            "contradiction_resolution": run.avg_contradiction_resolution,
            "tool_efficiency": run.avg_tool_efficiency,
            "budget_compliance": run.avg_budget_compliance,
            "critique_agreement": run.avg_critique_agreement,
        },
        "by_category": by_category,
    }


# ─────────────────────────────────────────────
# ENDPOINT 4: Approve / reject a prompt rewrite
# ─────────────────────────────────────────────

@app.post(
    "/eval/rewrites/{rewrite_id}/review",
    summary="Approve or reject a pending prompt rewrite proposed by the meta-agent",
)
def review_rewrite(rewrite_id: str, request: ApprovalRequest, db: Session = Depends(get_db)):
    rewrite = db.query(PromptRewrite).filter(PromptRewrite.id == rewrite_id).first()
    if not rewrite:
        raise HTTPException(status_code=404, detail={
            "error_code": "REWRITE_NOT_FOUND",
            "message": f"Rewrite {rewrite_id} not found.",
            "job_id": None,
        })
    if rewrite.status != "pending":
        raise HTTPException(status_code=400, detail={
            "error_code": "REWRITE_ALREADY_REVIEWED",
            "message": f"Rewrite {rewrite_id} already has status '{rewrite.status}'.",
            "job_id": None,
        })

    rewrite.status = "approved" if request.approved else "rejected"
    rewrite.reviewed_at = datetime.utcnow()
    rewrite.reviewer_note = request.reviewer_note

    if request.approved:
        meta_agent.register_prompt(rewrite.agent_id, rewrite.proposed_prompt)
        logger.info(f"Prompt rewrite {rewrite_id} approved and applied for agent {rewrite.agent_id}")

    db.commit()
    return {
        "rewrite_id": rewrite_id,
        "status": rewrite.status,
        "agent_id": rewrite.agent_id,
        "dimension": rewrite.dimension,
        "reviewed_at": rewrite.reviewed_at.isoformat(),
        "reviewer_note": rewrite.reviewer_note,
        "message": "Prompt applied to pipeline." if request.approved else "Rewrite rejected. No changes made.",
    }


# ─────────────────────────────────────────────
# ENDPOINT 5: Trigger re-eval on failed cases
# ─────────────────────────────────────────────

@app.post(
    "/eval/rerun-failed",
    summary="Trigger targeted re-eval on previously failed cases using latest approved prompts",
)
async def rerun_failed(db: Session = Depends(get_db)):
    latest_run = db.query(EvalRun).order_by(EvalRun.triggered_at.desc()).first()
    if not latest_run:
        raise HTTPException(status_code=404, detail={
            "error_code": "NO_EVAL_RUNS",
            "message": "No evaluation runs found.",
            "job_id": None,
        })

    failed_results = db.query(EvalResult).filter(
        EvalResult.run_id == latest_run.id,
        EvalResult.passed == False,
    ).all()

    if not failed_results:
        return {"message": "No failed cases to re-run.", "run_id": latest_run.id}

    failed_case_ids = [r.test_case_id for r in failed_results]
    logger.info(f"Re-running {len(failed_case_ids)} failed cases: {failed_case_ids}")

    new_run = await run_eval(db, prompt_version="rerun", case_ids=failed_case_ids)

    # Compute delta
    old_scores = {
        r.test_case_id: {
            "correctness": r.correctness_score,
            "citation_accuracy": r.citation_accuracy_score,
        }
        for r in failed_results
    }
    new_results = db.query(EvalResult).filter(EvalResult.run_id == new_run.id).all()
    deltas = []
    for nr in new_results:
        old = old_scores.get(nr.test_case_id, {})
        deltas.append({
            "test_case_id": nr.test_case_id,
            "old_correctness": old.get("correctness"),
            "new_correctness": nr.correctness_score,
            "delta_correctness": (nr.correctness_score or 0) - (old.get("correctness") or 0),
            "passed": nr.passed,
        })

    return {
        "message": f"Re-ran {len(failed_case_ids)} failed cases.",
        "new_run_id": new_run.id,
        "cases_rerun": len(failed_case_ids),
        "new_pass_rate": round(new_run.passed / new_run.total_cases, 3) if new_run.total_cases else 0,
        "deltas": deltas,
    }


# ─────────────────────────────────────────────
# BONUS: Trigger full eval run
# ─────────────────────────────────────────────

@app.post("/eval/run", summary="Trigger a full evaluation run across all 15 test cases")
async def trigger_eval(db: Session = Depends(get_db)):
    run = await run_eval(db)
    return {
        "run_id": run.id,
        "total": run.total_cases,
        "passed": run.passed,
        "failed": run.failed,
        "triggered_at": run.triggered_at.isoformat(),
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "mega-ai"}


@app.get("/", include_in_schema=False)
def frontend():
    path = "/app/frontend/index.html"
    if os.path.exists(path):
        try:
            return FileResponse(path)
        except Exception:
            pass
    return JSONResponse({
        "status": "ok",
        "service": "mega-ai",
        "api_docs": "/docs",
        "endpoints": ["/query", "/jobs/{id}/trace", "/eval/latest",
                      "/eval/run", "/eval/rerun-failed", "/eval/rewrites/{id}/review"]
    })
