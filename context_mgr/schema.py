from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    task_type: str
    dependencies: List[str] = []
    status: str = "pending"
    result: Optional[Any] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    relevance_score: float
    hop: int = 1


class ClaimScore(BaseModel):
    claim: str
    confidence: float
    flagged: bool = False
    flagged_span: Optional[str] = None
    reason: Optional[str] = None


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: str
    source_chunk_id: Optional[str] = None


class AgentMessage(BaseModel):
    sender: str
    recipient: str
    content: Any
    message_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SharedContext(BaseModel):
    job_id: str
    original_query: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # decomposer output
    subtasks: List[SubTask] = []

    # rag agent output
    retrieved_chunks: List[RetrievedChunk] = []
    rag_answer: Optional[str] = None
    rag_citations: Dict[str, str] = {}

    # critique agent output
    claim_scores: List[ClaimScore] = []
    critique_summary: Optional[str] = None

    # synthesis output
    final_answer: Optional[str] = None
    provenance_map: List[ProvenanceEntry] = []

    # orchestrator routing log
    routing_log: List[Dict[str, Any]] = []

    # message bus (agents post here, orchestrator reads)
    messages: List[AgentMessage] = []

    # token budgets: agent_id -> {budget, used}
    token_budgets: Dict[str, Dict[str, int]] = {}

    # tool outputs keyed by tool call id
    tool_outputs: Dict[str, Any] = {}

    # policy violations
    policy_violations: List[str] = []

    # per-agent prompts used (for eval reproducibility)
    prompts_used: Dict[str, str] = {}

    def post_message(self, sender: str, recipient: str, content: Any, message_type: str):
        self.messages.append(AgentMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
        ))

    def get_messages_for(self, recipient: str) -> List[AgentMessage]:
        return [m for m in self.messages if m.recipient == recipient]
