"""
Tool executor with retry logic and per-attempt DB logging.
Each retry is logged as a separate ToolCallLog entry.
The orchestrator handles each failure mode differently — not via prompts.
"""
import time
import logging
import uuid
from typing import Callable, Any
from sqlalchemy.orm import Session

from tools.base import ToolResult, FailureMode
from db.models import ToolCallLog

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _log_tool_call(
    db: Session,
    job_id: str,
    agent_id: str,
    tool_name: str,
    attempt: int,
    input_data: Any,
    result: ToolResult,
    accepted: bool,
):
    if db is None:
        return
    try:
        entry = ToolCallLog(
            id=str(uuid.uuid4()),
            job_id=job_id,
            agent_id=agent_id,
            tool_name=tool_name,
            attempt=attempt,
            input_data=input_data if isinstance(input_data, dict) else {"input": str(input_data)},
            output_data=result.data,
            latency_ms=result.latency_ms,
            accepted=accepted,
            failure_mode=result.failure_mode.value if not result.success else None,
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        logger.warning(f"[tool_executor] Failed to log tool call: {e}")


def _handle_failure(failure_mode: FailureMode, tool_name: str, attempt: int) -> dict:
    """
    Orchestrator-level failure handling — explicit code logic, NOT prompt instructions.
    Each failure mode is handled differently.
    """
    if failure_mode == FailureMode.timeout:
        logger.warning(f"[tool_executor] {tool_name} timed out (attempt {attempt}) — will retry with shorter input")
        return {"action": "retry", "modification": "reduce_input_size"}

    elif failure_mode == FailureMode.empty_results:
        logger.warning(f"[tool_executor] {tool_name} returned empty (attempt {attempt}) — will retry with broader query")
        return {"action": "retry", "modification": "broaden_query"}

    elif failure_mode == FailureMode.malformed_input:
        logger.error(f"[tool_executor] {tool_name} received malformed input (attempt {attempt}) — no retry")
        return {"action": "abort", "modification": None}

    elif failure_mode == FailureMode.execution_error:
        logger.warning(f"[tool_executor] {tool_name} execution error (attempt {attempt}) — will retry")
        return {"action": "retry", "modification": "simplify_input"}

    return {"action": "abort", "modification": None}


def execute_with_retry(
    tool_fn: Callable,
    tool_name: str,
    tool_input: Any,
    agent_id: str,
    job_id: str,
    db: Session = None,
    input_modifier: Callable = None,
) -> ToolResult:
    """
    Execute a tool with up to MAX_RETRIES retries.
    Each attempt is logged separately in the DB.
    Returns the first successful result, or the last failed result.
    """
    current_input = tool_input
    last_result = None

    for attempt in range(1, MAX_RETRIES + 2):  # 1, 2, 3
        logger.info(f"[tool_executor] {tool_name} attempt {attempt}/{MAX_RETRIES+1}")

        # Execute tool
        try:
            result = tool_fn(current_input) if not isinstance(current_input, dict) else tool_fn(**current_input)
        except Exception as e:
            result = ToolResult(
                success=False,
                failure_mode=FailureMode.execution_error,
                error_message=str(e),
                latency_ms=0.0,
            )

        last_result = result

        # Decide if agent accepts the result
        accepted = result.success

        # Log this attempt
        _log_tool_call(
            db=db,
            job_id=job_id,
            agent_id=agent_id,
            tool_name=tool_name,
            attempt=attempt,
            input_data=current_input,
            result=result,
            accepted=accepted,
        )

        if result.success:
            logger.info(f"[tool_executor] {tool_name} succeeded on attempt {attempt}")
            return result

        # Check if we should retry
        if attempt > MAX_RETRIES:
            logger.error(f"[tool_executor] {tool_name} failed after {MAX_RETRIES+1} attempts")
            break

        failure_decision = _handle_failure(result.failure_mode, tool_name, attempt)

        if failure_decision["action"] == "abort":
            logger.error(f"[tool_executor] {tool_name} aborting — malformed input cannot be retried")
            break

        # Modify input for retry based on failure mode
        if input_modifier and failure_decision["modification"]:
            current_input = input_modifier(current_input, failure_decision["modification"])
            logger.info(f"[tool_executor] Retrying {tool_name} with modified input ({failure_decision['modification']})")

    return last_result
