# =============================================================================
# orchestrator/nodes.py — Graph Node Functions + Agent Registry
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Each node in the LangGraph is an async function with a specific signature:
#   async def node_name(state: OrchestratorState) -> dict
#
# The returned dict contains ONLY the fields we want to update in state.
# LangGraph merges this partial update into the full state automatically.
#
# THREE NODES:
#
# 1. node_route
#    Runs RouterAgent, gets RoutingDecision
#    Updates: routing_decision, agents_to_execute, execution_strategy, status
#
# 2. node_execute_agents
#    Runs the actual agent(s) based on execution_strategy:
#    - single/sequential: run agents one by one, pass previous output as context
#    - parallel: run all agents simultaneously with asyncio.gather
#    Updates: agent_outputs, tokens_used, execution_times_ms
#
# 3. node_aggregate
#    Combines all agent outputs into the final response
#    Updates: final_output, status, total_execution_ms
#
# AGENT REGISTRY:
# A dict mapping AgentType → agent instance.
# The orchestrator doesn't import agents directly — it uses the registry.
# WHY: Adding a new agent = add one line to AGENT_REGISTRY. Done.
# The orchestrator code never needs to change.
# =============================================================================

import asyncio
import time
from typing import Any

from core.logger import get_logger
from core.memory import get_memory_manager
from schemas.task import TaskInput
from schemas.routing import AgentType, ExecutionStrategy
from agents.router_agent import RouterAgent
from agents.research_agent import ResearchAgent
from agents.summarization_agent import SummarizationAgent
from agents.outreach_agent import OutreachAgent
from agents.coding_agent import CodingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.base_agent import BaseAgent
from orchestrator.state import OrchestratorState

logger = get_logger("orchestrator.nodes")


# =============================================================================
# AGENT REGISTRY — The plug-in system for all agents
# =============================================================================
# Every agent that the orchestrator can delegate to must be registered here.
# Key = AgentType enum value (string), Value = agent instance
#
# When the router says "use research agent", the orchestrator does:
#   agent = AGENT_REGISTRY["research"]
#   output = await agent.execute(task, context)
#
# This is the OPEN/CLOSED PRINCIPLE:
# Open for extension (add new agents), closed for modification (no changes needed here)
# =============================================================================

# Single shared instances — created once at startup
_router_agent = RouterAgent()
_research_agent = ResearchAgent()
_summarization_agent = SummarizationAgent()
_outreach_agent = OutreachAgent()
_coding_agent = CodingAgent()
_retrieval_agent = RetrievalAgent()

AGENT_REGISTRY: dict[str, BaseAgent] = {
    AgentType.RESEARCH.value: _research_agent,
    AgentType.SUMMARIZATION.value: _summarization_agent,
    AgentType.OUTREACH.value: _outreach_agent,
    AgentType.CODING.value: _coding_agent,
    AgentType.RETRIEVAL.value: _retrieval_agent,
    # Fallback aliases — router may return these for ambiguous tasks
    AgentType.GENERAL.value: _research_agent,       # "general" → research
    "qa": _coding_agent,                             # "qa" → coding (code review)
    "content": _outreach_agent,                      # "content" → outreach (writing)
    "planning": _research_agent,                     # "planning" → research (analysis)
}


# =============================================================================
# NODE 1 — ROUTING
# =============================================================================

async def node_route(state: OrchestratorState) -> dict:
    """
    LangGraph Node: Run the RouterAgent to decide execution plan.

    Reads from state:
        task_id, task_message, task_priority, task_context

    Updates state with:
        routing_decision, agents_to_execute, execution_strategy, status

    If routing fails, sets status="failed" and adds to errors.
    The edge function will then route to a failure endpoint.
    """
    logger.info(
        "node_route_started",
        task_id=state["task_id"],
        message_preview=state["task_message"][:60],
    )

    try:
        # Reconstruct the TaskInput from state
        # (We store primitives in state, not Pydantic objects, for LangGraph compatibility)
        task = TaskInput(
            task_id=state["task_id"],
            user_message=state["task_message"],
            priority=state["task_priority"],
            context=state["task_context"],
        )

        # Get routing decision from router agent
        decision = await _router_agent.route(task)

        logger.info(
            "node_route_completed",
            task_id=state["task_id"],
            task_type=decision.task_type,
            agents=decision.agent_names(),
            strategy=decision.execution_strategy.value,
            confidence=decision.confidence,
        )

        # Return PARTIAL state update — only the fields this node touches
        return {
            "routing_decision": decision.model_dump(),
            "agents_to_execute": decision.agent_names(),
            "execution_strategy": decision.execution_strategy.value,
            "status": "executing",
        }

    except Exception as e:
        logger.error("node_route_failed", task_id=state["task_id"], error=str(e))
        return {
            "status": "failed",
            "errors": [f"Routing failed: {str(e)}"],
        }


# =============================================================================
# NODE 2 — AGENT EXECUTION
# =============================================================================

async def node_execute_agents(state: OrchestratorState) -> dict:
    """
    LangGraph Node: Execute all required agents based on the routing decision.

    This single node handles all three execution strategies:

    SINGLE:
        One agent, run it, done.

    SEQUENTIAL:
        Agents run in order. Each agent receives the previous agent's output
        as context. This is how Research → Outreach works.

    PARALLEL:
        All agents run simultaneously with asyncio.gather().
        None receive each other's output (they run at the same time).
        Results are collected after all finish.

    Reads from state:
        agents_to_execute, execution_strategy, task_*, routing_decision

    Updates state with:
        agent_outputs, tokens_used, execution_times_ms
    """
    strategy = state["execution_strategy"]
    agents = state["agents_to_execute"]

    logger.info(
        "node_execute_started",
        task_id=state["task_id"],
        strategy=strategy,
        agents=agents,
    )

    task = TaskInput(
        task_id=state["task_id"],
        user_message=state["task_message"],
        priority=state["task_priority"],
        context=state["task_context"],
    )

    # Accumulated results
    agent_outputs: dict[str, dict] = {}
    tokens_used: dict[str, int] = {}
    execution_times: dict[str, int] = {}
    errors: list[str] = []

    try:
        if strategy == ExecutionStrategy.PARALLEL.value:
            # ----------------------------------------------------------------
            # PARALLEL EXECUTION
            # Run all agents simultaneously, collect results after all finish
            # ----------------------------------------------------------------
            logger.info("executing_parallel", count=len(agents))

            async def run_agent(agent_name: str) -> tuple[str, Any]:
                """Run a single agent and return (name, output) tuple."""
                agent = AGENT_REGISTRY.get(agent_name)
                if not agent:
                    logger.warning("agent_not_found", agent=agent_name)
                    return agent_name, None
                output = await agent.execute(task)
                return agent_name, output

            # Launch all agents at the same time
            results = await asyncio.gather(
                *[run_agent(name) for name in agents],
                return_exceptions=True
            )

            # Collect results
            for item in results:
                if isinstance(item, Exception):
                    errors.append(f"Agent execution error: {str(item)}")
                    continue
                agent_name, output = item
                if output and output.success:
                    agent_outputs[agent_name] = output.result
                    if output.tokens_used:
                        tokens_used[agent_name] = output.tokens_used
                    execution_times[agent_name] = output.execution_time_ms
                elif output:
                    errors.append(f"{agent_name} failed: {output.error}")

        else:
            # ----------------------------------------------------------------
            # SINGLE or SEQUENTIAL EXECUTION
            # Run agents one by one, passing previous output as context
            # ----------------------------------------------------------------
            accumulated_context: dict = {}

            for i, agent_name in enumerate(agents):
                agent = AGENT_REGISTRY.get(agent_name)
                if not agent:
                    msg = f"Agent '{agent_name}' not found in registry"
                    logger.warning("agent_not_found", agent=agent_name)
                    errors.append(msg)
                    continue

                logger.info(
                    "executing_agent",
                    agent=agent_name,
                    step=f"{i+1}/{len(agents)}",
                    has_context=bool(accumulated_context),
                )

                # Build context for this agent from previous agents' outputs
                # This is the KEY to sequential pipelines:
                # Research output → becomes context for Outreach agent
                context = _build_agent_context(
                    agent_name=agent_name,
                    previous_outputs=accumulated_context,
                    task_context=state["task_context"],
                )

                output = await agent.execute(task, context=context)

                if output.success:
                    agent_outputs[agent_name] = output.result
                    if output.tokens_used:
                        tokens_used[agent_name] = output.tokens_used
                    execution_times[agent_name] = output.execution_time_ms

                    # Add this agent's output to accumulated context for next agent
                    # Key pattern: "research_output", "outreach_output", etc.
                    accumulated_context[f"{agent_name}_output"] = output.result

                    logger.info(
                        "agent_succeeded",
                        agent=agent_name,
                        duration_ms=output.execution_time_ms,
                    )
                else:
                    error_msg = f"{agent_name} failed: {output.error}"
                    errors.append(error_msg)
                    logger.error("agent_failed_in_pipeline", agent=agent_name, error=output.error)
                    # In sequential mode, a failed agent stops the pipeline
                    # because subsequent agents may depend on this output
                    break

        return {
            "agent_outputs": agent_outputs,
            "tokens_used": tokens_used,
            "execution_times_ms": execution_times,
            "errors": errors,
            "status": "aggregating" if not errors else "failed",
        }

    except Exception as e:
        logger.error("node_execute_failed", task_id=state["task_id"], error=str(e))
        return {
            "agent_outputs": agent_outputs,
            "tokens_used": tokens_used,
            "execution_times_ms": execution_times,
            "errors": [f"Execution node failed: {str(e)}"],
            "status": "failed",
        }


def _build_agent_context(
    agent_name: str,
    previous_outputs: dict,
    task_context: dict,
) -> dict:
    """
    Build the context dict for an agent based on what previous agents produced.

    This is the context-passing mechanism for sequential pipelines.

    Rules:
    - OutreachAgent looks for "research_output" key in context
    - SummarizationAgent looks for "content_to_summarize" key in context
    - Any agent can look at "previous_outputs" for full history

    This function knows the conventions that agents expect.
    When you add a new agent, add its context-building logic here.
    """
    context: dict = {}

    # Pass all previous outputs so any agent can access them
    context.update(previous_outputs)

    # Agent-specific context enrichment
    if agent_name == AgentType.OUTREACH.value and "research_output" in previous_outputs:
        # OutreachAgent already handles "research_output" key natively
        pass  # Already in context via context.update(previous_outputs)

    if agent_name == AgentType.SUMMARIZATION.value:
        # If research ran before summarization, summarize the research summary
        if "research_output" in previous_outputs:
            research = previous_outputs["research_output"]
            # Build a combined text from research output for summarization
            facts = "\n".join(f"- {f}" for f in research.get("key_facts", []))
            context["content_to_summarize"] = (
                f"{research.get('summary', '')}\n\nKey facts:\n{facts}"
            )

    # If retrieval ran before this agent, inject the retrieved memories
    # so the agent knows what happened in previous sessions
    if "retrieval_output" in previous_outputs:
        retrieval = previous_outputs["retrieval_output"]
        if retrieval.get("has_relevant_context"):
            context["retrieved_memories"] = retrieval.get("relevant_context", "")

    # Pass any extra context from the original task
    if task_context:
        context["task_context"] = task_context

    return context


# =============================================================================
# NODE 3 — OUTPUT AGGREGATION
# =============================================================================

async def node_aggregate(state: OrchestratorState) -> dict:
    """
    LangGraph Node: Combine all agent outputs into the final response.

    This node assembles everything into a clean, structured response
    that gets returned to the caller (API, test, or user interface).

    The final_output structure:
    {
        "task_id": "...",
        "task_type": "...",
        "status": "complete",
        "agents_used": ["research", "outreach"],
        "execution_strategy": "sequential",
        "primary_output": "The main content (last agent's output)",
        "all_outputs": {"research": {...}, "outreach": {...}},
        "performance": {
            "total_execution_ms": 12345,
            "tokens_by_agent": {"research": 1603, "outreach": 1871},
            "total_tokens": 3474
        },
        "routing": {
            "task_type": "research_and_outreach",
            "confidence": 0.95,
            "reasoning": "..."
        }
    }
    """
    agent_outputs = state.get("agent_outputs", {})
    routing = state.get("routing_decision", {})
    tokens = state.get("tokens_used", {})
    times = state.get("execution_times_ms", {})

    logger.info(
        "node_aggregate_started",
        task_id=state["task_id"],
        agents_completed=list(agent_outputs.keys()),
    )

    # Identify the primary output — last agent's output is the final deliverable
    agents_executed = list(agent_outputs.keys())
    primary_agent = agents_executed[-1] if agents_executed else None
    primary_output = agent_outputs.get(primary_agent, {}) if primary_agent else {}

    # Extract the main content from primary output
    # Agents store their main text in different keys — check common ones
    primary_content = (
        primary_output.get("body")          # OutreachAgent
        or primary_output.get("solution")   # CodingAgent
        or primary_output.get("summary")    # ResearchAgent / SummarizationAgent
        or primary_output.get("content")    # Generic
        or str(primary_output)              # Fallback
    )

    total_tokens = sum(tokens.values())
    total_time_ms = sum(times.values())

    final = {
        "task_id": state["task_id"],
        "task_type": routing.get("task_type", "unknown"),
        "status": "complete",
        "agents_used": agents_executed,
        "execution_strategy": state["execution_strategy"],
        "primary_output": primary_content,
        "all_outputs": agent_outputs,
        "performance": {
            "total_execution_ms": total_time_ms,
            "tokens_by_agent": tokens,
            "total_tokens": total_tokens,
        },
        "routing": {
            "task_type": routing.get("task_type"),
            "confidence": routing.get("confidence"),
            "reasoning": routing.get("reasoning"),
            "primary_agent": routing.get("primary_agent"),
        },
    }

    logger.info(
        "node_aggregate_completed",
        task_id=state["task_id"],
        agents_used=agents_executed,
        total_tokens=total_tokens,
        total_ms=total_time_ms,
    )

    # -------------------------------------------------------------------------
    # AUTO-SAVE TO MEMORY
    # -------------------------------------------------------------------------
    # Save this completed task to long-term vector memory.
    # This is NON-CRITICAL — if it fails, the response still returns normally.
    # Over time, these saves build the memory store that the RetrievalAgent
    # will search when future tasks ask about "what we discussed before."
    # -------------------------------------------------------------------------
    try:
        memory = get_memory_manager()
        await memory.save_task(
            task_id=state["task_id"],
            task_message=state["task_message"],
            primary_output=str(primary_content)[:800] if primary_content else "",
            agents_used=agents_executed,
            status="complete",
        )
    except Exception as mem_err:
        # Never let memory failures affect the response
        logger.warning("memory_save_skipped", error=str(mem_err))

    return {
        "final_output": final,
        "status": "complete",
        "total_execution_ms": total_time_ms,
    }


# =============================================================================
# NODE 4 — FAILURE HANDLER
# =============================================================================

async def node_failed(state: OrchestratorState) -> dict:
    """
    LangGraph Node: Handle failed executions gracefully.

    Instead of crashing, returns a structured error response.
    This is what gets returned to the caller when something goes wrong.
    """
    errors = state.get("errors", [])

    logger.error(
        "orchestration_failed",
        task_id=state["task_id"],
        errors=errors,
    )

    final = {
        "task_id": state["task_id"],
        "status": "failed",
        "errors": errors,
        "agents_used": list(state.get("agent_outputs", {}).keys()),
        "partial_outputs": state.get("agent_outputs", {}),
        "primary_output": None,
    }

    return {
        "final_output": final,
        "status": "failed",
    }
