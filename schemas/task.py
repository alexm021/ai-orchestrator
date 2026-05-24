# =============================================================================
# schemas/task.py — Task Input Schema
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Every task that enters the system must have a defined, validated shape.
# This is the "front door" of the entire orchestrator.
#
# Without a schema:
#   - You get raw strings, random dicts, missing fields
#   - Bugs appear deep inside agents, far from where the problem started
#   - You can't write tests because you don't know what to expect
#
# With a schema:
#   - Invalid tasks are rejected at the boundary, before any agent touches them
#   - Every component knows exactly what a "task" looks like
#   - Pydantic gives you free validation, serialization, and documentation
#
# FLOW:
#   User sends request → API layer creates TaskInput → Router receives TaskInput
#   TaskInput travels through the ENTIRE system unchanged.
# =============================================================================

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
import uuid


class TaskInput(BaseModel):
    """
    The standard input object for every task in the system.
    Created once at the API boundary, passed to every component.

    Fields:
        task_id     — Unique ID for tracking this task through logs
        user_message — The actual task/question from the user
        priority    — How urgent is this task?
        context     — Optional extra info (user ID, previous results, etc.)
        created_at  — When was this task created?
    """

    # Unique ID — generated automatically if not provided
    # WHY: Every log line, every agent call, every error includes task_id
    #      so you can trace a single task through the entire system
    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this task"
    )

    # The actual task from the user — this is what agents work with
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The task or question to be processed"
    )

    # Priority level — the orchestrator can use this to reorder execution
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Task priority level"
    )

    # Optional context dictionary — anything extra the system might need
    # Examples: {"user_id": "123", "previous_output": "...", "language": "en"}
    context: dict = Field(
        default_factory=dict,
        description="Optional additional context for the task"
    )

    # Timestamp — useful for latency measurement and debugging
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this task was created"
    )

    class Config:
        # Allow serialization of datetime objects
        json_encoders = {datetime: lambda v: v.isoformat()}

    def summary(self) -> str:
        """Returns a short human-readable summary of this task."""
        msg_preview = self.user_message[:80] + "..." if len(self.user_message) > 80 else self.user_message
        return f"[{self.task_id[:8]}] ({self.priority}) {msg_preview}"


# =============================================================================
# Example usage (for documentation purposes):
#
#   task = TaskInput(
#       user_message="Research the top 5 AI companies and write a summary",
#       priority="high",
#       context={"user_id": "user_001", "language": "en"}
#   )
#
#   # task.task_id is auto-generated UUID
#   # task.created_at is auto-set to now
#   # task.summary() → "[a1b2c3d4] (high) Research the top 5 AI companies..."
# =============================================================================
