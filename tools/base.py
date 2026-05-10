from pydantic import BaseModel
from typing import Any, Optional
from enum import Enum


class FailureMode(str, Enum):
    timeout = "timeout"
    empty_results = "empty_results"
    malformed_input = "malformed_input"
    execution_error = "execution_error"
    none = "none"


class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    failure_mode: FailureMode = FailureMode.none
    error_message: Optional[str] = None
    latency_ms: float = 0.0
