# =============================================================================
# orchestrator/graph.py — LangGraph Orchestration Engine
# =============================================================================
#
# WHY THIS FILE EXISTS:
# This is where everything comes together.
# We wire up nodes + edges into a compiled, runnable graph.
#
# THE GRAPH STRUCTURE:
#
#   START
#     │
#     ▼
#   [node_route] ──── RouterAgent decides which agents to run
#     │
#     ▼ (conditional edge: edge_after_routing)
#     ├─ "node_execute_agents"  (routing succeeded)
#     └─ "node_failed"          (routing failed)
#
#   [node_execute_agents] ──── Runs agents (single/sequential/parallel)
#     │
#     ▼ (conditional edge: edge_after_execution)
#     ├─ "node_aggregate"  (at least some outputs exist)
#     └─ "node_failed"     (zero outputs)
#
#   [node_aggregate] ──── Combines outputs into final response
#     │
#     ▼
#   END
#
#   [node_failed] ──── Returns structured error response
#     │
#     ▼
#   END
#
# HOW TO USE:
#   from orchestrator.graph import run_task
#
#   result = await run_task(
#       task_message="Research Tesla and write a cold email to their VP",
#       task_priority="high"
#   )
#   print(result["primary_output"])
# =============================================================================

import time
from langgraph.graph import StateGraph, END

from orchestrator.state import OrchestratorState, create_initial_state
from orchestrator.nodes import node_route, node_execute_agents, node_aggregate, node_failed
from orchestrator.edges import edge_after_routing, edge_after_execution
from core.logger import get_logger

logger = get_logger("orchestrator.graph")


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph orchestration graph.

    Called ONCE at startup (or when testing).
    Returns a compiled, runnable graph object.

    WHY SEPARATE build_graph() FROM run_task():
    Building the graph is expensive (validates structure, compiles edges).
    We build it once and reuse it for all requests.
    """
    # Create the graph with our state schema
    graph = StateGraph(OrchestratorState)

    # -------------------------------------------------------------------------
    # ADD NODES — Register each function as a named node
    # The string name is what edges reference
    # -------------------------------------------------------------------------
    graph.add_node("node_route", node_route)
    graph.add_node("node_execute_agents", node_execute_agents)
    graph.add_node("node_aggregate", node_aggregate)
    graph.add_node("node_failed", node_failed)

    # -------------------------------------------------------------------------
    # SET ENTRY POINT — Where execution starts
    # -------------------------------------------------------------------------
    graph.set_entry_point("node_route")

    # -------------------------------------------------------------------------
    # ADD EDGES — Define the execution flow
    # -------------------------------------------------------------------------

    # After routing: conditional — go to execute or fail
    graph.add_conditional_edges(
        "node_route",                  # From this node
        edge_after_routing,            # This function decides where to go
        {
            "node_execute_agents": "node_execute_agents",   # If returns "node_execute_agents"
            "node_failed": "node_failed",                   # If returns "node_failed"
        }
    )

    # After execution: conditional — go to aggregate or fail
    graph.add_conditional_edges(
        "node_execute_agents",
        edge_after_execution,
        {
            "node_aggregate": "node_aggregate",
            "node_failed": "node_failed",
        }
    )

    # After aggregation: always go to END
    graph.add_edge("node_aggregate", END)

    # After failure handler: always go to END
    graph.add_edge("node_failed", END)

    # Compile the graph — validates structure, optimizes execution
    compiled = graph.compile()
    logger.info("graph_compiled", nodes=["node_route", "node_execute_agents", "node_aggregate", "node_failed"])
    return compiled


# =============================================================================
# SINGLETON — Build once, reuse everywhere
# =============================================================================
# The compiled graph is created once when this module is first imported.
# Every subsequent call to run_task() reuses the same compiled graph.

_graph = None


def get_graph():
    """
    Returns the singleton compiled graph, building it on first call.
    Thread-safe: Python's GIL ensures this runs only once.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# =============================================================================
# PUBLIC API — The only function external code needs to call
# =============================================================================

async def run_task(
    task_message: str,
    task_priority: str = "medium",
    task_context: dict | None = None,
    task_id: str | None = None,
) -> dict:
    """
    Run a task through the full orchestration pipeline.

    This is the main entry point for the orchestrator.
    Called by the FastAPI API layer (Phase 6) and tests.

    Args:
        task_message: The user's task or question
        task_priority: "low" | "medium" | "high"
        task_context: Optional extra context dict
        task_id: Optional custom ID (auto-generated if not provided)

    Returns:
        Final output dict with:
        - task_id: tracking ID
        - status: "complete" or "failed"
        - primary_output: the main result text
        - all_outputs: individual agent results
        - performance: timing and token usage
        - routing: routing decision details

    Example:
        result = await run_task(
            task_message="Research Stripe and write a cold email to their CTO",
            task_priority="high"
        )
        print(result["primary_output"])  # The final email
        print(result["performance"]["total_tokens"])  # Token cost
    """
    start_time = time.monotonic()

    # Create initial state
    task = create_initial_state(
        task_id=task_id or __import__('uuid').uuid4().__str__(),
        task_message=task_message,
        task_priority=task_priority,
        task_context=task_context or {},
    )

    logger.info(
        "orchestration_started",
        task_id=task["task_id"],
        message_preview=task_message[:80],
        priority=task_priority,
    )

    try:
        # Execute the graph — this runs all nodes in sequence
        # ainvoke = async invocation (non-blocking)
        graph = get_graph()
        result_state = await graph.ainvoke(task)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Extract the final output from the completed state
        final = result_state.get("final_output", {})
        if final:
            final["total_wall_time_ms"] = elapsed_ms

        status = result_state.get("status", "unknown")
        logger.info(
            "orchestration_completed",
            task_id=task["task_id"],
            status=status,
            wall_time_ms=elapsed_ms,
            agents_used=final.get("agents_used", []),
        )

        return final

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.error(
            "orchestration_crashed",
            task_id=task["task_id"],
            error=str(e),
            wall_time_ms=elapsed_ms,
        )
        return {
            "task_id": task["task_id"],
            "status": "failed",
            "errors": [str(e)],
            "primary_output": None,
            "total_wall_time_ms": elapsed_ms,
        }
