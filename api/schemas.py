# =============================================================================
# api/schemas.py — FastAPI Request & Response Models
# =============================================================================
#
# WHY SEPARATE FROM orchestrator SCHEMAS:
# The orchestrator schemas (schemas/task.py, schemas/routing.py etc.) are
# INTERNAL models used by agents and nodes. They can be complex and contain
# fields that aren't relevant to the API consumer.
#
# API schemas are the PUBLIC CONTRACT — what the caller sends in, and what
# they can expect to receive back. They should be:
#   - Simple and clear
#   - Well-documented (description= on every field)
#   - Stable (you don't break clients by changing internal models)
#
# This separation is the API Gateway Pattern: external interface is decoupled
# from internal implementation. We translate between the two in route handlers.
# =============================================================================

from pydantic import BaseModel, Field
from typing import Optional


# =============================================================================
# REQUEST MODELS
# =============================================================================

class TaskRequest(BaseModel):
    """
    Incoming task request from the API client.

    Example:
        POST /task
        {
            "message": "Research OpenAI and write a cold email to their Head of Product",
            "priority": "high"
        }
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The task description. Be specific — the more context, the better the output.",
        examples=["Research Anthropic AI and write a cold email to their Head of Product"],
    )
    priority: str = Field(
        default="medium",
        pattern="^(low|medium|high)$",
        description="Task priority. Affects execution urgency. One of: low, medium, high.",
    )
    context: dict = Field(
        default_factory=dict,
        description="Optional extra context passed to agents (e.g., company name, tone preference).",
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class PerformanceInfo(BaseModel):
    """Token usage and timing breakdown."""
    total_tokens: int = Field(0, description="Total tokens used across all agents")
    total_execution_ms: int = Field(0, description="Total agent execution time in ms")
    tokens_by_agent: dict = Field(
        default_factory=dict,
        description="Per-agent token breakdown: {'research': 1348, 'outreach': 2575}",
    )


class RoutingInfo(BaseModel):
    """How the router classified and dispatched this task."""
    task_type: Optional[str] = Field(None, description="Detected task category")
    confidence: Optional[float] = Field(None, description="Router confidence score (0-1)")
    primary_agent: Optional[str] = Field(None, description="Main agent responsible")
    reasoning: Optional[str] = Field(None, description="Router's explanation of the decision")


class TaskResponse(BaseModel):
    """
    Full orchestration result returned to the API client.

    On success: status="complete", primary_output contains the main result.
    On failure: status="failed", errors contains the error list.
    """
    task_id: str = Field(..., description="Unique task identifier (UUID)")
    status: str = Field(..., description="'complete' or 'failed'")
    agents_used: list[str] = Field(
        default_factory=list,
        description="Agents that ran, in execution order: ['research', 'outreach']",
    )
    execution_strategy: Optional[str] = Field(
        None,
        description="How agents ran: 'single', 'sequential', or 'parallel'",
    )
    primary_output: Optional[str] = Field(
        None,
        description="The main result text (last agent's primary output)",
    )
    all_outputs: dict = Field(
        default_factory=dict,
        description="Full structured output from every agent that ran",
    )
    performance: PerformanceInfo = Field(
        default_factory=PerformanceInfo,
        description="Token usage and execution timing",
    )
    routing: RoutingInfo = Field(
        default_factory=RoutingInfo,
        description="How the router classified this task",
    )
    total_wall_time_ms: int = Field(
        0,
        description="Total end-to-end time including routing and aggregation",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages if status='failed'",
    )


# =============================================================================
# HEALTH CHECK MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """
    System health status.

    status="healthy"  → all systems operational
    status="degraded" → partial functionality (e.g., memory unavailable)
    status="unhealthy" → critical failure
    """
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    app_name: str = Field(..., description="Application name from config")
    environment: str = Field(..., description="'development' or 'production'")
    memory_count: int = Field(0, description="Number of memories stored in ChromaDB")
    agents_available: list[str] = Field(
        default_factory=list,
        description="List of registered agent names",
    )
    version: str = Field("1.0.0", description="API version")


# =============================================================================
# MEMORY MODELS
# =============================================================================

class MemoryRecord(BaseModel):
    """A single memory entry from ChromaDB."""
    task_message: str = Field(..., description="Original user request")
    primary_output: str = Field(..., description="Main result (truncated to 500 chars)")
    agents_used: list[str] = Field(..., description="Agents that handled this task")
    timestamp: str = Field(..., description="ISO timestamp of when this task ran")
    relevance_score: Optional[float] = Field(
        None,
        description="Semantic relevance to search query (0-1). Only present in search results.",
    )


class MemoriesResponse(BaseModel):
    """List of stored task memories."""
    total: int = Field(..., description="Total memories in the store")
    memories: list[MemoryRecord] = Field(
        default_factory=list,
        description="Memory records, newest first",
    )


class MemorySearchRequest(BaseModel):
    """Request body for memory semantic search."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query — finds semantically similar past tasks",
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )


# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response for 4xx/5xx responses."""
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error description")
    detail: Optional[dict] = Field(None, description="Additional error context")
