# =============================================================================
# orchestrator/state.py — Shared Orchestrator State
# =============================================================================
#
# WHY THIS FILE EXISTS:
# LangGraph nodes are stateless functions. They can't remember what happened
# in previous nodes. The State object is the SHARED MEMORY of the graph.
#
# Every node reads from State and writes back to State.
# The graph guarantees that each node sees the latest State.
#
# ANALOGY:
# Think of State as a shared whiteboard in a conference room.
# Each person (node) reads what's on it, does their work, and writes results.
# The next person comes in and sees everything that was written before them.
#
# DESIGN DECISIONS:
# 1. TypedDict instead of Pydantic — LangGraph natively uses TypedDict for state
# 2. Every field has Optional/default — nodes only update what they change
# 3. Annotated[list, operator.add] — for errors/logs, we APPEND not replace
# 4. We store dicts, not Pydantic objects — safer for serialization
#
# STATE LIFECYCLE:
#   Created fresh for every incoming task
#   Passed through: route_task → execute_agents → aggregate_output
#   Contains full history of the execution when done
# =============================================================================

from typing import TypedDict, Optional, Annotated
import operator


class OrchestratorState(TypedDict):
    """
    The complete shared state that flows through the LangGraph execution.

    Populated incrementally as nodes execute:
      - task: set by caller before graph starts
      - routing_decision: set by node_route
      - agents_to_execute: set by node_route
      - agent_outputs: updated by node_execute (each agent adds its result)
      - final_output: set by node_aggregate
    """

    # -------------------------------------------------------------------------
    # INPUT — Set once before the graph starts, never changed
    # -------------------------------------------------------------------------

    # The incoming task — what the user wants
    task_id: str
    task_message: str
    task_priority: str
    task_context: dict  # Any extra context from the caller

    # -------------------------------------------------------------------------
    # ROUTING — Populated by node_route
    # -------------------------------------------------------------------------

    # The router's decision (stored as dict for LangGraph compatibility)
    # Fields: task_type, primary_agent, agents_required, execution_strategy,
    #         reasoning, confidence, requires_memory, expected_output_format
    routing_decision: Optional[dict]

    # Ordered list of agent names to execute (from routing_decision)
    # Example: ["research", "outreach"]  or  ["coding"]
    agents_to_execute: list[str]

    # Which execution strategy: "single" | "sequential" | "parallel"
    execution_strategy: str

    # -------------------------------------------------------------------------
    # EXECUTION — Updated by node_execute_agents
    # -------------------------------------------------------------------------

    # Map of agent results: {"research": {topic, summary, key_facts, ...}}
    # Updated after each agent completes
    agent_outputs: dict[str, dict]

    # Token usage tracking per agent: {"research": 1603, "outreach": 1871}
    tokens_used: dict[str, int]

    # Execution time per agent in ms: {"research": 6425, "outreach": 5845}
    execution_times_ms: dict[str, int]

    # -------------------------------------------------------------------------
    # ERROR TRACKING
    # -------------------------------------------------------------------------

    # Annotated with operator.add means: state["errors"] + new_errors
    # So when a node returns {"errors": ["something failed"]},
    # it APPENDS to the list instead of replacing it.
    # This preserves the full error history across all nodes.
    errors: Annotated[list[str], operator.add]

    # -------------------------------------------------------------------------
    # OUTPUT — Populated by node_aggregate
    # -------------------------------------------------------------------------

    # The final assembled response returned to the caller
    final_output: Optional[dict]

    # Overall execution status
    # "pending" → "routing" → "executing" → "aggregating" → "complete" | "failed"
    status: str

    # Total execution time from start to finish
    total_execution_ms: Optional[int]


def create_initial_state(
    task_id: str,
    task_message: str,
    task_priority: str = "medium",
    task_context: dict | None = None,
) -> OrchestratorState:
    """
    Factory function to create a fresh initial state for a new task.

    Call this before invoking the graph:
        state = create_initial_state(task_id="abc", task_message="Research Tesla...")
        result = await app.ainvoke(state)

    Every field has a safe default — nodes only need to update what they change.
    """
    return OrchestratorState(
        # Input
        task_id=task_id,
        task_message=task_message,
        task_priority=task_priority,
        task_context=task_context or {},
        # Routing (empty until node_route runs)
        routing_decision=None,
        agents_to_execute=[],
        execution_strategy="single",
        # Execution (empty until node_execute runs)
        agent_outputs={},
        tokens_used={},
        execution_times_ms={},
        # Errors (empty at start)
        errors=[],
        # Output (empty until node_aggregate runs)
        final_output=None,
        status="pending",
        total_execution_ms=None,
    )
