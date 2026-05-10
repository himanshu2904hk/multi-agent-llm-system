import subprocess
import time
import tempfile
import os
from tools.base import ToolResult, FailureMode


def execute_python(code: str, timeout_seconds: int = 10) -> ToolResult:
    start = time.time()

    if not code or not code.strip():
        return ToolResult(
            success=False,
            failure_mode=FailureMode.malformed_input,
            error_message="Code must be a non-empty string.",
            latency_ms=0.0,
        )

    if timeout_seconds <= 0:
        return ToolResult(
            success=False,
            failure_mode=FailureMode.timeout,
            error_message="timeout_seconds must be positive.",
            latency_ms=0.0,
        )

    # Reject obviously dangerous patterns
    dangerous = ["import os", "import sys", "subprocess", "open(", "__import__", "eval(", "exec("]
    for pattern in dangerous:
        if pattern in code:
            return ToolResult(
                success=False,
                failure_mode=FailureMode.malformed_input,
                error_message=f"Forbidden pattern detected: {pattern}",
                latency_ms=0.0,
            )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        latency = (time.time() - start) * 1000
        return ToolResult(
            success=result.returncode == 0,
            data={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            },
            failure_mode=FailureMode.execution_error if result.returncode != 0 else FailureMode.none,
            error_message=result.stderr if result.returncode != 0 else None,
            latency_ms=latency,
        )
    except subprocess.TimeoutExpired:
        latency = (time.time() - start) * 1000
        return ToolResult(
            success=False,
            failure_mode=FailureMode.timeout,
            error_message=f"Code execution timed out after {timeout_seconds}s.",
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        return ToolResult(
            success=False,
            failure_mode=FailureMode.execution_error,
            error_message=str(e),
            latency_ms=latency,
        )
    finally:
        os.unlink(tmp_path)
