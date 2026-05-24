# =============================================================================
# orchestrator/edges.py — Conditional Edge Functions
# =============================================================================
#
# WHY THIS FILE EXISTS:
# In LangGraph, edges between nodes can be CONDITIONAL.
# Instead of always going A → B, you can go A → B or A → C depending on state.
#
# Edge functions read the current state and return a STRING.
# That string is the name of the next node to execute.
#
# This is how the graph makes decisions at runtime:
#   "Did routing succeed? → go to execute"
#   "Did routing fail? → go to failed"
#   "Are all agents done? → go to aggregate"
#
# PATTERN:
#   def edge_name(state: OrchestratorState) -> str:
#       if condition:
#           return "next_node_name"
#       return "other_node_name"
#
# Then in graph.py:
#   graph.add_conditional_edges("node_route", edge_after_routing, {...})
# =============================================================================

from orchestrator.state import OrchestratorState
from core.logger import get_logger

logger = get_logger("orchestrator.edges")


def edge_after_routing(state: OrchestratorState) -> str:
    """
    Conditional edge: what to do after node_route completes.

    If routing succeeded → proceed to execute agents
    If routing failed → go to failure handler

    Returns the name of the next node.
    """
    status = state.get("status", "")
    has_agents = bool(state.get("agents_to_execute"))

    if status == "failed" or not has_agents:
        logger.warning(
            "routing_failed_routing_to_failure",
            task_id=state["task_id"],
            status=status,
            agents=state.get("agents_to_execute"),
        )
        return "node_failed"

    logger.info(
        "routing_succeeded_routing_to_execute",
        task_id=state["task_id"],
        agents=state.get("agents_to_execute"),
        strategy=state.get("execution_strategy"),
    )
    return "node_execute_agents"


def edge_after_execution(state: OrchestratorState) -> str:
    """
    Conditional edge: what to do after node_execute_agents completes.

    If execution succeeded (at least some outputs) → aggregate
    If execution completely failed → failure handler

    Note: Partial failures (some agents succeeded, some failed) still go
    to aggregation — we return what we have and include errors in the output.
    This is more useful than returning nothing just because one agent failed.
    """
    status = state.get("status", "")
    has_outputs = bool(state.get("agent_outputs"))
    errors = state.get("errors", [])

    # If we have ANY outputs, aggregate them even if some agents failed
    if has_outputs:
        if errors:
            logger.warning(
                "partial_execution_proceeding_to_aggregate",
                task_id=state["task_id"],
                successful_agents=list(state.get("agent_outputs", {}).keys()),
                errors=errors,
            )
        return "node_aggregate"

    # No outputs at all — complete failure
    logger.error(
        "execution_failed_routing_to_failure",
        task_id=state["task_id"],
        status=status,
        errors=errors,
    )
    return "node_failed"
