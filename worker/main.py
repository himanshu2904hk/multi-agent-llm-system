"""
Background worker: polls Redis for jobs and processes them asynchronously.
Jobs are enqueued as JSON: {"job_id": "...", "query": "..."}
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

import redis
from sqlalchemy.orm import Session

from db.database import SessionLocal, init_db
from db.models import Job, JobStatus
from context_mgr.schema import SharedContext
from agents.orchestrator import run_pipeline
from agents import meta_agent, decomposer, rag_agent, critique_agent, synthesis_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "mega_ai:jobs"
POLL_INTERVAL = 2


def get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


async def process_job(job_id: str, query: str, db: Session):
    logger.info(f"Processing job {job_id}: {query[:60]}")

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        job = Job(id=job_id, query=query, status=JobStatus.running)
        db.add(job)
    else:
        job.status = JobStatus.running
    db.commit()

    try:
        context = SharedContext(job_id=job_id, original_query=query)
        context = await run_pipeline(context)

        job.status = JobStatus.completed
        job.final_answer = context.final_answer
        job.provenance_map = [p.dict() for p in context.provenance_map]
        job.policy_violations = context.policy_violations
        job.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"Job {job_id} completed")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.failed
        db.commit()


async def worker_loop():
    r = get_redis()
    db = SessionLocal()
    logger.info(f"Worker started, polling {QUEUE_KEY}")

    while True:
        try:
            item = r.blpop(QUEUE_KEY, timeout=POLL_INTERVAL)
            if item:
                _, payload = item
                data = json.loads(payload)
                job_id = data.get("job_id", str(uuid.uuid4()))
                query = data.get("query", "")
                await process_job(job_id, query, db)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            await asyncio.sleep(POLL_INTERVAL)


def main():
    init_db()
    meta_agent.register_prompt("decomposer", decomposer.SYSTEM_PROMPT)
    meta_agent.register_prompt("rag_agent", rag_agent.SYSTEM_PROMPT)
    meta_agent.register_prompt("critique_agent", critique_agent.SYSTEM_PROMPT)
    meta_agent.register_prompt("synthesis_agent", synthesis_agent.SYSTEM_PROMPT)
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
