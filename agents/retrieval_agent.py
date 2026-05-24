# =============================================================================
# agents/retrieval_agent.py — Retrieval Agent
# =============================================================================
#
# PURPOSE:
# The Retrieval Agent is invoked when the Router detects that a task
# references past context — phrases like "last time", "we discussed",
# "the email you wrote", "previous conversation", etc.
#
# It searches the MemoryManager for semantically similar past tasks
# and returns a structured summary of what it found.
#
# WHAT MAKES THIS AGENT DIFFERENT:
# Most agents call Gemini to generate new content.
# This agent queries ChromaDB — NO LLM CALL (for the search itself).
# It's pure retrieval: embed the query → find closest vectors → return them.
#
# The structured output tells downstream agents:
#   "Here's what we talked about before, use this context."
#
# EXAMPLE FLOW:
# User: "Follow up on that Anthropic email you wrote"
# Router: requires_memory=True → routes to retrieval first
# Retrieval: finds the previous Anthropic outreach in memory
# Outreach: gets the retrieved context, writes a follow-up that references
#           the previous email's subject, tone, and key points
# =============================================================================

from pydantic import BaseModel, Field
from typing import Optional

from agents.base_agent import BaseAgent
from schemas.task import TaskInput
from schemas.agent_output import AgentOutput
from schemas.routing import AgentType
from core.memory import get_memory_manager
from core.logger import get_logger


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

class MemoryRecord(BaseModel):
    """A single retrieved memory entry."""
    task_message: str = Field(..., description="The original task request")
    primary_output: str = Field(..., description="The main output from that task")
    agents_used: list[str] = Field(..., description="Which agents handled it")
    timestamp: str = Field(..., description="When it happened (ISO format)")
    relevance_score: float = Field(..., description="How relevant this is (0-1)")


class RetrievalOutput(BaseModel):
    """Structured output from the Retrieval Agent."""
    memories_found: int = Field(..., description="Number of relevant memories found")
    relevant_context: str = Field(..., description="Human-readable summary of relevant past context")
    memories: list[MemoryRecord] = Field(default_factory=list, description="Full memory records")
    query_used: str = Field(..., description="The search query that was used")
    has_relevant_context: bool = Field(..., description="True if any useful memories were found")


# =============================================================================
# RETRIEVAL AGENT
# =============================================================================

class RetrievalAgent(BaseAgent):
    """
    Retrieval Agent — searches vector memory for relevant past context.

    Unlike other agents, this one does NOT call Gemini to generate content.
    It queries ChromaDB using Gemini embeddings and returns structured results.

    The returned context is passed to subsequent agents via the pipeline's
    context-passing mechanism — they receive "retrieved_memories" in their
    context dict.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RETRIEVAL

    async def execute(
        self,
        task: TaskInput,
        context: Optional[dict] = None,
    ) -> AgentOutput:
        """
        Search memory for tasks relevant to the current request.

        Args:
            task: The TaskInput — we use task.user_message as the search query
            context: Optional context from previous agents (rarely used here)

        Returns:
            AgentOutput with RetrievalOutput as result
        """
        start_time = self._start_timer()

        self.logger.info(
            "retrieval_started",
            task_id=task.task_id,
            query_length=len(task.user_message),
        )

        try:
            memory = get_memory_manager()

            # Semantic search — finds memories about similar topics
            memories = await memory.search(
                query=task.user_message,
                n_results=5,
                min_relevance=0.3,
            )

            if not memories:
                # No relevant memories found — return gracefully
                output = RetrievalOutput(
                    memories_found=0,
                    relevant_context=(
                        "No relevant past conversations found in memory. "
                        "This appears to be a new topic."
                    ),
                    memories=[],
                    query_used=task.user_message[:200],
                    has_relevant_context=False,
                )
            else:
                # Build a human-readable context summary for downstream agents
                # This is what OutreachAgent or other agents will actually read
                context_parts = [
                    f"Found {len(memories)} relevant past interaction(s):\n"
                ]
                for i, m in enumerate(memories, 1):
                    # Format each memory as a mini-briefing
                    ts = m["timestamp"][:10] if m["timestamp"] else "unknown date"
                    agents_str = ", ".join(m["agents_used"])
                    context_parts.append(
                        f"[Memory {i} — {ts}] (relevance: {m['relevance_score']:.0%})\n"
                        f"  Request: {m['task_message'][:150]}\n"
                        f"  Result:  {m['primary_output'][:200]}\n"
                        f"  Agents:  {agents_str}"
                    )

                relevant_context = "\n\n".join(context_parts)

                output = RetrievalOutput(
                    memories_found=len(memories),
                    relevant_context=relevant_context,
                    memories=[MemoryRecord(**m) for m in memories],
                    query_used=task.user_message[:200],
                    has_relevant_context=True,
                )

            self.logger.info(
                "retrieval_completed",
                task_id=task.task_id,
                memories_found=output.memories_found,
                has_context=output.has_relevant_context,
            )

            # Note: model_used is "memory" not a Gemini model —
            # the embedding call doesn't count as agent model usage
            return self._success(
                result=output.model_dump(),
                start_time=start_time,
                model_used="chromadb+text-embedding-004",
            )

        except Exception as e:
            self.logger.error("retrieval_failed", task_id=task.task_id, error=str(e))
            return self._failure(str(e), start_time)
