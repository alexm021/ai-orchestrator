# =============================================================================
# schemas/agent_output.py — Standard Agent Output Schema
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Every agent in the system — Research, Outreach, Coding, QA, etc. —
# returns a different TYPE of content. But they all return it in the
# SAME WRAPPER structure.
#
# This is the "envelope" pattern:
#   - The envelope (AgentOutput) always looks the same
#   - The letter inside (result) is different per agent
#
# WHY IS THIS IMPORTANT?
# The orchestrator doesn't care what specific content an agent produced.
# It cares about:
#   - Did it succeed?
#   - How long did it take?
#   - What's the output to pass to the next agent?
#   - Was there an error?
#
# By standardizing the output wrapper, the orchestrator can handle
# ALL agents the same way. New agents just plug in automatically.
# =============================================================================

from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
from schemas.routing import AgentType


class AgentOutput(BaseModel):
    """
    Standard output wrapper returned by every agent in the system.

    The orchestrator reads this to know:
    - Did this agent succeed? (success field)
    - What did it produce? (result field)
    - How long did it take? (execution_time_ms)
    - What error occurred if it failed? (error field)
    """

    # Which agent produced this output
    agent_name: AgentType = Field(
        ...,
        description="The agent that produced this output"
    )

    # Did the agent complete successfully?
    # False = something went wrong, check the error field
    success: bool = Field(
        ...,
        description="Whether the agent completed successfully"
    )

    # The actual output — different for each agent type
    # Research agent: {"summary": "...", "sources": [...], "key_facts": [...]}
    # Outreach agent: {"subject": "...", "body": "...", "tone": "professional"}
    # Coding agent:   {"code": "...", "language": "python", "explanation": "..."}
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="The agent's output content (structure varies by agent type)"
    )

    # Error message if success=False
    # Should be a clear, actionable message — not a raw Python traceback
    error: Optional[str] = Field(
        default=None,
        description="Error message if success=False"
    )

    # How long the agent took to run, in milliseconds
    # Critical for monitoring — if an agent consistently takes 10s, something is wrong
    execution_time_ms: int = Field(
        default=0,
        description="How long this agent took to execute in milliseconds"
    )

    # How many tokens were used in the LLM call (for cost tracking)
    # Optional because not all operations call an LLM
    tokens_used: Optional[int] = Field(
        default=None,
        description="LLM tokens consumed by this agent"
    )

    # Which LLM model was used — useful for debugging unexpected behavior
    model_used: Optional[str] = Field(
        default=None,
        description="Which LLM model this agent used"
    )

    # When this output was produced — for timeline reconstruction
    completed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this agent completed"
    )

    # -------------------------------------------------------------------------
    # Factory methods — convenient ways to create success/failure outputs
    # -------------------------------------------------------------------------

    @classmethod
    def success_output(
        cls,
        agent_name: AgentType,
        result: dict,
        execution_time_ms: int,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> "AgentOutput":
        """
        Convenience method to create a successful output.

        Usage in any agent:
            return AgentOutput.success_output(
                agent_name=AgentType.RESEARCH,
                result={"summary": "...", "sources": [...]},
                execution_time_ms=1240,
                model_used="gemini-2.5-flash",
                tokens_used=850
            )
        """
        return cls(
            agent_name=agent_name,
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            model_used=model_used,
            tokens_used=tokens_used,
        )

    @classmethod
    def failure_output(
        cls,
        agent_name: AgentType,
        error: str,
        execution_time_ms: int,
    ) -> "AgentOutput":
        """
        Convenience method to create a failure output.

        Usage in any agent:
            return AgentOutput.failure_output(
                agent_name=AgentType.RESEARCH,
                error="Gemini API rate limit exceeded after 3 retries",
                execution_time_ms=3000
            )
        """
        return cls(
            agent_name=agent_name,
            success=False,
            result={},
            error=error,
            execution_time_ms=execution_time_ms,
        )

    def get_content(self) -> str:
        """
        Returns the main text content from result, for easy passing to next agent.
        Agents store their primary output in result["content"] or result["text"].
        """
        return (
            self.result.get("content")
            or self.result.get("text")
            or self.result.get("output")
            or str(self.result)
        )
