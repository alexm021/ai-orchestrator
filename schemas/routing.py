# =============================================================================
# schemas/routing.py — Router Agent Output Schema
# =============================================================================
#
# WHY THIS FILE EXISTS:
# The Router Agent returns a DECISION, not content.
# This schema defines exactly what that decision looks like.
#
# This is one of the most important files in the system.
# The RoutingDecision object is what connects the Router to the Orchestrator.
# If this schema is wrong, nothing downstream will work correctly.
#
# WHAT HAPPENS WITH THIS:
#   Router Agent produces → RoutingDecision
#   Orchestrator reads RoutingDecision → knows which agents to call, in what order
#
# WHY PYDANTIC ENUMS:
# We use Enum classes instead of raw strings like "research" or "sequential".
# If you typo "reserach" instead of "research" — Pydantic catches it immediately.
# Raw strings fail silently. Enums fail loudly. We want loud failures.
# =============================================================================

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# =============================================================================
# ENUMS — Valid values for agent types and execution strategies
# =============================================================================

class AgentType(str, Enum):
    """
    All specialized agents available in the system.

    WHY str + Enum:
    By inheriting from str, these values serialize as plain strings.
    So AgentType.RESEARCH serializes to "research" in JSON.
    Pure Enum would serialize as "AgentType.RESEARCH" — ugly and wrong.
    """
    RESEARCH = "research"           # Gathers information, web search, analysis
    SUMMARIZATION = "summarization" # Condenses long text into key points
    OUTREACH = "outreach"           # Writes emails, LinkedIn, cold outreach
    CODING = "coding"               # Writes, reviews, or debugs code
    QA = "qa"                       # Tests and validates quality of outputs
    CONTENT = "content"             # Blog posts, social media, marketing copy
    PLANNING = "planning"           # Breaks complex tasks into actionable steps
    RETRIEVAL = "retrieval"         # Fetches docs from memory/knowledge base
    GENERAL = "general"             # Fallback for uncategorized tasks


class ExecutionStrategy(str, Enum):
    """
    How the orchestrator should run the required agents.

    SINGLE:
        One agent handles the entire task.
        Example: "Summarize this article" → just SummarizationAgent

    SEQUENTIAL:
        Agents run one after another. Output of agent N becomes input for agent N+1.
        Example: "Research Tesla and write a cold email"
        → ResearchAgent runs first → OutreachAgent uses research output

    PARALLEL:
        Multiple agents run simultaneously, results are combined at the end.
        Example: "Review this code for bugs AND security issues"
        → CodingAgent + QAAgent run at the same time (faster)
    """
    SINGLE = "single"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# =============================================================================
# ROUTING DECISION — What the Router Agent produces
# =============================================================================

class RoutingDecision(BaseModel):
    """
    The structured output of the Router Agent.

    Every field here is a decision the router makes about how to handle a task.
    The orchestrator reads this object and executes accordingly.

    IMPORTANT: This schema is sent to Gemini as the response_schema.
    Gemini will return JSON that matches this exact structure.
    Pydantic then validates and parses it into this object.
    """

    # Human-readable classification of the task type
    # Examples: "research_request", "cold_email", "code_review", "content_creation"
    task_type: str = Field(
        ...,
        description="Short label classifying the type of task"
    )

    # The MAIN agent — most important one, always required
    # If only one agent is needed, this is the only one used
    primary_agent: AgentType = Field(
        ...,
        description="The main agent responsible for this task"
    )

    # ALL agents required, including primary
    # For single-agent tasks: ["research"]
    # For multi-agent tasks: ["research", "summarization", "outreach"]
    agents_required: list[AgentType] = Field(
        ...,
        min_length=1,
        description="All agents needed to complete this task, in execution order"
    )

    # How should the orchestrator run these agents?
    execution_strategy: ExecutionStrategy = Field(
        ...,
        description="Whether agents run single, sequential, or parallel"
    )

    # WHY this routing decision was made — critical for debugging
    # Example: "Task requires web research before writing outreach email"
    # This appears in logs so you can understand any routing decision
    reasoning: str = Field(
        ...,
        description="Explanation of why this routing decision was made"
    )

    # How confident is the router in this decision? 0.0 to 1.0
    # Low confidence = maybe the task is ambiguous
    # Orchestrator can use this to decide whether to ask for clarification
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Router confidence in this decision (0.0 to 1.0)"
    )

    # Does this task need to access stored memory/previous context?
    # True = retrieval agent should fetch relevant context first
    requires_memory: bool = Field(
        default=False,
        description="Whether this task needs memory/context retrieval"
    )

    # Optional hint about what the final output should look like
    # Examples: "email", "bullet_list", "code_snippet", "json_report"
    expected_output_format: Optional[str] = Field(
        default=None,
        description="Hint about the expected format of the final output"
    )

    def is_multi_agent(self) -> bool:
        """Returns True if more than one agent is required."""
        return len(self.agents_required) > 1

    def agent_names(self) -> list[str]:
        """Returns agent names as plain strings."""
        return [a.value for a in self.agents_required]

    def summary(self) -> str:
        """Human-readable summary of this routing decision."""
        agents = " -> ".join(self.agent_names())
        return (
            f"[{self.task_type}] "
            f"Strategy: {self.execution_strategy.value} | "
            f"Agents: {agents} | "
            f"Confidence: {self.confidence:.0%}"
        )


# =============================================================================
# Example of what a RoutingDecision looks like:
#
# Task: "Research Tesla's latest earnings and write a cold email to their CFO"
#
# RoutingDecision(
#     task_type="research_and_outreach",
#     primary_agent=AgentType.RESEARCH,
#     agents_required=[AgentType.RESEARCH, AgentType.OUTREACH],
#     execution_strategy=ExecutionStrategy.SEQUENTIAL,
#     reasoning="Task requires factual research before personalized outreach can be written",
#     confidence=0.95,
#     requires_memory=False,
#     expected_output_format="email"
# )
# =============================================================================
