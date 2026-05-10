from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, JSON, ForeignKey, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

Base = declarative_base()


def new_uuid():
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=new_uuid)
    query = Column(Text, nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    final_answer = Column(Text, nullable=True)
    provenance_map = Column(JSON, nullable=True)
    policy_violations = Column(JSON, default=list)
    events = relationship("AgentEvent", back_populates="job", cascade="all, delete-orphan")
    tool_calls = relationship("ToolCallLog", back_populates="job", cascade="all, delete-orphan")


class AgentEvent(Base):
    __tablename__ = "agent_events"
    id = Column(String, primary_key=True, default=new_uuid)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    input_hash = Column(String, nullable=True)
    output_hash = Column(String, nullable=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    policy_violation = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    job = relationship("Job", back_populates="events")


class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"
    id = Column(String, primary_key=True, default=new_uuid)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    tool_name = Column(String, nullable=False)
    attempt = Column(Integer, default=1)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)
    accepted = Column(Boolean, nullable=True)
    failure_mode = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    job = relationship("Job", back_populates="tool_calls")


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id = Column(String, primary_key=True, default=new_uuid)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    prompt_version = Column(String, default="v1")
    total_cases = Column(Integer, default=0)
    passed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    avg_correctness = Column(Float, nullable=True)
    avg_citation_accuracy = Column(Float, nullable=True)
    avg_contradiction_resolution = Column(Float, nullable=True)
    avg_tool_efficiency = Column(Float, nullable=True)
    avg_budget_compliance = Column(Float, nullable=True)
    avg_critique_agreement = Column(Float, nullable=True)
    results = relationship("EvalResult", back_populates="run", cascade="all, delete-orphan")


class EvalResult(Base):
    __tablename__ = "eval_results"
    id = Column(String, primary_key=True, default=new_uuid)
    run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=True)
    test_case_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    actual_answer = Column(Text, nullable=True)
    correctness_score = Column(Float, nullable=True)
    correctness_justification = Column(Text, nullable=True)
    citation_accuracy_score = Column(Float, nullable=True)
    citation_accuracy_justification = Column(Text, nullable=True)
    contradiction_resolution_score = Column(Float, nullable=True)
    contradiction_resolution_justification = Column(Text, nullable=True)
    tool_efficiency_score = Column(Float, nullable=True)
    tool_efficiency_justification = Column(Text, nullable=True)
    budget_compliance_score = Column(Float, nullable=True)
    budget_compliance_justification = Column(Text, nullable=True)
    critique_agreement_score = Column(Float, nullable=True)
    critique_agreement_justification = Column(Text, nullable=True)
    passed = Column(Boolean, default=False)
    exact_prompts = Column(JSON, nullable=True)
    run = relationship("EvalRun", back_populates="results")


class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"
    id = Column(String, primary_key=True, default=new_uuid)
    eval_run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    original_prompt = Column(Text, nullable=False)
    proposed_prompt = Column(Text, nullable=False)
    diff = Column(Text, nullable=False)
    justification = Column(Text, nullable=False)
    status = Column(String, default="pending")
    proposed_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_note = Column(Text, nullable=True)
    delta_correctness = Column(Float, nullable=True)
    delta_citation = Column(Float, nullable=True)
    delta_contradiction = Column(Float, nullable=True)
    delta_tool_efficiency = Column(Float, nullable=True)
    delta_budget_compliance = Column(Float, nullable=True)
    delta_critique_agreement = Column(Float, nullable=True)
