import asyncio
import logging
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from eval.test_cases import TEST_CASES
from eval.scorer import score_all_dimensions
from context_mgr.schema import SharedContext
from agents.orchestrator import run_pipeline
from agents.meta_agent import analyze_and_propose
from db.models import EvalRun, EvalResult, Job, JobStatus

logger = logging.getLogger(__name__)


async def run_eval(db: Session, prompt_version: str = "v1", case_ids: list = None) -> EvalRun:
    cases = TEST_CASES
    if case_ids:
        cases = [c for c in TEST_CASES if c["id"] in case_ids]

    run = EvalRun(
        id=str(uuid.uuid4()),
        prompt_version=prompt_version,
        total_cases=len(cases),
        triggered_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()

    results = []
    score_totals = {
        "correctness": [],
        "citation_accuracy": [],
        "contradiction_resolution": [],
        "tool_efficiency": [],
        "budget_compliance": [],
        "critique_agreement": [],
    }

    for case in cases:
        logger.info(f"[eval] Running case {case['id']} ({case['category']})")

        job_id = str(uuid.uuid4())
        context = SharedContext(job_id=job_id, original_query=case["query"])

        # Save job
        job = Job(id=job_id, query=case["query"], status=JobStatus.running)
        db.add(job)
        db.commit()

        try:
            context = await run_pipeline(context)
            job.status = JobStatus.completed
            job.final_answer = context.final_answer
            job.provenance_map = [p.dict() for p in context.provenance_map]
            job.policy_violations = context.policy_violations
            job.completed_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"[eval] Pipeline error for {case['id']}: {e}")
            context.final_answer = f"Pipeline error: {str(e)}"
            job.status = JobStatus.failed

        db.commit()

        # Score
        scores = score_all_dimensions(
            query=case["query"],
            expected=case.get("expected_answer"),
            actual=context.final_answer,
            context=context,
        )

        passed = scores["overall"]["score"] >= 0.6

        eval_result = EvalResult(
            run_id=run.id,
            job_id=job_id,
            test_case_id=case["id"],
            category=case["category"],
            query=case["query"],
            expected_answer=case.get("expected_answer"),
            actual_answer=context.final_answer,
            correctness_score=scores["correctness"]["score"],
            correctness_justification=scores["correctness"]["justification"],
            citation_accuracy_score=scores["citation_accuracy"]["score"],
            citation_accuracy_justification=scores["citation_accuracy"]["justification"],
            contradiction_resolution_score=scores["contradiction_resolution"]["score"],
            contradiction_resolution_justification=scores["contradiction_resolution"]["justification"],
            tool_efficiency_score=scores["tool_efficiency"]["score"],
            tool_efficiency_justification=scores["tool_efficiency"]["justification"],
            budget_compliance_score=scores["budget_compliance"]["score"],
            budget_compliance_justification=scores["budget_compliance"]["justification"],
            critique_agreement_score=scores["critique_agreement"]["score"],
            critique_agreement_justification=scores["critique_agreement"]["justification"],
            passed=passed,
            exact_prompts=context.prompts_used,
        )
        db.add(eval_result)
        results.append(eval_result)

        for dim in score_totals:
            score_totals[dim].append(scores[dim]["score"])

        logger.info(f"[eval] {case['id']}: overall={scores['overall']['score']:.2f} passed={passed}")

    # Update run aggregates
    n = len(results)
    run.passed = sum(1 for r in results if r.passed)
    run.failed = n - run.passed
    run.avg_correctness = sum(score_totals["correctness"]) / n if n else 0
    run.avg_citation_accuracy = sum(score_totals["citation_accuracy"]) / n if n else 0
    run.avg_contradiction_resolution = sum(score_totals["contradiction_resolution"]) / n if n else 0
    run.avg_tool_efficiency = sum(score_totals["tool_efficiency"]) / n if n else 0
    run.avg_budget_compliance = sum(score_totals["budget_compliance"]) / n if n else 0
    run.avg_critique_agreement = sum(score_totals["critique_agreement"]) / n if n else 0
    db.commit()

    # Run meta-agent to propose improvements
    analyze_and_propose(db, run.id)

    logger.info(f"[eval] Run {run.id} complete: {run.passed}/{n} passed")
    return run
