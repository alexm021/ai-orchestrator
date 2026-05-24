# =============================================================================
# agents/router_agent.py — Router Agent
# =============================================================================
#
# PURPOSE:
# The Router Agent is the FIRST component that runs on every incoming task.
# It does NOT produce content. It produces a DECISION.
#
# Specifically, it answers:
#   - What kind of task is this?
#   - Which agent(s) should handle it?
#   - In what order should they run?
#   - How confident are we in this decision?
#
# TECHNICAL APPROACH:
# We use Gemini's structured output feature (response_mime_type="application/json")
# This forces Gemini to return valid JSON that matches our RoutingDecision schema.
#
# WHY STRUCTURED OUTPUTS FOR ROUTING:
# If we don't constrain the output, Gemini might return:
#   "I think the research agent would be best for this task..."
# That's a string. We can't act on a string programmatically.
#
# With structured output, we always get:
#   {"task_type": "research_request", "primary_agent": "research", ...}
# That's a machine-readable decision. The orchestrator can act on it immediately.
#
# WHY LOW TEMPERATURE (0.1):
# Routing decisions should be CONSISTENT, not creative.
# If "write a cold email" is routed to OutreachAgent, it should ALWAYS
# be routed to OutreachAgent, not 70% of the time.
# Low temperature = deterministic, consistent behavior.
# =============================================================================

import json
from google.genai import types
from core.logger import get_logger
from core.exceptions import RouterError, UnroutableTaskError
from schemas.task import TaskInput
from schemas.routing import RoutingDecision, AgentType, ExecutionStrategy
from schemas.agent_output import AgentOutput
from agents.base_agent import BaseAgent


# =============================================================================
# ROUTER SYSTEM PROMPT
# =============================================================================
# This is the "brain" of the router — the instructions that tell Gemini
# exactly how to analyze tasks and make routing decisions.
#
# PROMPT ENGINEERING PRINCIPLES USED HERE:
# 1. Clear role definition ("You are the Router Agent...")
# 2. Explicit list of available capabilities
# 3. Decision rules stated clearly
# 4. Examples of each agent's responsibility
# 5. Format requirements
# =============================================================================

ROUTER_SYSTEM_PROMPT = """You are the Router Agent for an AI Multi-Agent Orchestrator system.

Your ONLY job is to analyze incoming tasks and decide which specialized agents should handle them.
You do NOT produce content. You produce ROUTING DECISIONS.

## Available Agents

- **research**: Gathers and analyzes information. Use for: market research, competitor analysis,
  fact-finding, topic exploration, data gathering, news research, company analysis.

- **summarization**: Condenses long content into key points. Use for: summarizing articles,
  reports, meeting notes, long documents, or any text that needs to be shortened.

- **outreach**: Creates professional communication. Use for: cold emails, LinkedIn messages,
  follow-up emails, sales scripts, partnership proposals, any email writing.

- **coding**: Handles all code-related tasks. Use for: writing code, debugging, code review,
  explaining code, refactoring, architecture advice, technical documentation.

- **qa**: Validates and tests outputs. Use for: reviewing agent outputs, fact-checking,
  quality control, proofreading, checking for errors or inconsistencies.

- **content**: Creates marketing and long-form content. Use for: blog posts, social media posts,
  ad copy, product descriptions, newsletters, any marketing content.

- **planning**: Breaks down complex tasks into steps. Use for: project planning, strategy creation,
  roadmaps, action plans, task decomposition, workflow design.

- **retrieval**: Fetches relevant context from memory. Use for: tasks that reference previous
  conversations, need historical context, or require past data.

- **general**: Fallback for simple tasks. Use for: direct questions, conversational responses,
  tasks that don't clearly fit other categories.

## Routing Rules

1. **Single agent**: Most tasks need only one agent. Default to SINGLE strategy.
2. **Sequential**: Use when Agent B needs Agent A's output. Order matters.
   Example: research → outreach (must research before writing the email)
3. **Parallel**: Use when agents are independent and can run simultaneously.
   Example: coding + qa (review code AND check quality at the same time)
4. **Confidence**: Be honest about confidence. Score 0.9+ only when the task is unambiguous.
5. **requires_memory**: Set True only when the task references "previous", "last time", "our conversation", or similar.

## Key Decision Examples

- "Write a cold email to Tesla's CEO" → outreach (single)
- "Research Tesla then write a cold email" → research → outreach (sequential)
- "Fix this Python bug" → coding (single)
- "Summarize this 3000-word article" → summarization (single)
- "Create a 3-month content strategy" → planning → content (sequential)
- "What did we discuss last time?" → retrieval (single, requires_memory=true)

Always return a complete routing decision. Never refuse to route a task.
"""


class RouterAgent(BaseAgent):
    """
    The Router Agent — analyzes tasks and returns routing decisions.

    This agent is special: it doesn't inherit the standard execute() flow
    fully because it returns a RoutingDecision, not an AgentOutput.

    It has two methods:
        route()   — Returns RoutingDecision (used by the orchestrator)
        execute() — Wraps route() in AgentOutput (standard interface)
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger("RouterAgent")

    @property
    def agent_type(self) -> AgentType:
        # Router is a special agent — it doesn't appear in AgentType enum
        # because it never processes tasks, only routes them.
        # We return GENERAL as a placeholder for the base class requirement.
        return AgentType.GENERAL

    async def route(self, task: TaskInput) -> RoutingDecision:
        """
        PRIMARY METHOD — Analyze a task and return a RoutingDecision.

        This is what the orchestrator calls on every incoming request.

        Args:
            task: The TaskInput to route

        Returns:
            RoutingDecision with all routing details

        Raises:
            RouterError: If Gemini fails to produce a valid routing decision
            UnroutableTaskError: If the task cannot be classified
        """
        start_time = self._start_timer()

        self.logger.info(
            "routing_started",
            task_id=task.task_id,
            priority=task.priority,
            message_length=len(task.user_message),
        )

        try:
            # Build the prompt with the task details
            # WHY SEPARATE FROM SYSTEM PROMPT:
            # System prompt = instructions (cached, same for every call)
            # User content = the actual task (changes every call)
            # Keeping them separate allows Gemini to cache the system prompt
            # and only process the new task content. Saves tokens and latency.
            user_prompt = self._build_routing_prompt(task)

            # Fallback model chain — try each model in order
            # If primary is rate-limited (429), automatically try the next one.
            # This is basic resilience — the system keeps working even when
            # one model tier hits its quota.
            # Full fallback chain — ordered from fastest/cheapest to most capable
            models_to_try = [
                self.settings.router_model,
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
            ]
            # Remove duplicates while preserving order
            seen = set()
            models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

            response = None
            last_error = None

            for model_name in models_to_try:
                for attempt in range(2):
                    try:
                        response = await self._gemini_client.aio.models.generate_content(
                            model=model_name,
                            contents=user_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=ROUTER_SYSTEM_PROMPT,
                                response_mime_type="application/json",
                                response_schema=RoutingDecision,
                                temperature=0.1,
                                max_output_tokens=512,
                            )
                        )
                        self.logger.info("model_selected", model=model_name, task_id=task.task_id)
                        break
                    except Exception as model_err:
                        import asyncio as _asyncio
                        last_error = model_err
                        error_str = str(model_err)
                        if "429" not in error_str and "RESOURCE_EXHAUSTED" not in error_str:
                            raise
                        wait_sec = self._extract_retry_delay(error_str)
                        if attempt == 0 and wait_sec <= 65:
                            wait = wait_sec + 3
                            self.logger.info("rate_limited_waiting", model=model_name,
                                            wait_seconds=wait, task_id=task.task_id)
                            await _asyncio.sleep(wait)
                        else:
                            self.logger.warning("model_skipped", model=model_name,
                                                retry_suggested_seconds=wait_sec)
                            break
                if response is not None:
                    break

            if response is None:
                raise last_error

            # Parse the JSON response into our Pydantic model
            decision = self._parse_routing_response(response.text, task)

            elapsed = self._elapsed_ms(start_time)
            self.logger.info(
                "routing_completed",
                task_id=task.task_id,
                task_type=decision.task_type,
                primary_agent=decision.primary_agent.value,
                agents=decision.agent_names(),
                strategy=decision.execution_strategy.value,
                confidence=decision.confidence,
                duration_ms=elapsed,
            )

            return decision

        except (RouterError, UnroutableTaskError):
            # Re-raise our own exceptions unchanged
            raise
        except Exception as e:
            elapsed = self._elapsed_ms(start_time)
            self.logger.error(
                "routing_failed",
                task_id=task.task_id,
                error=str(e),
                duration_ms=elapsed,
            )
            raise RouterError(
                message=f"Router agent failed: {str(e)}",
                details={"task_id": task.task_id, "error": str(e)}
            )

    def _build_routing_prompt(self, task: TaskInput) -> str:
        """
        Build the user-facing prompt content for the routing request.

        Includes the task message and any relevant context.
        Keeps this separate from the system prompt for clarity.
        """
        prompt_parts = [
            f"Task to route:\n{task.user_message}",
            f"\nPriority: {task.priority}",
        ]

        # Add context if provided (e.g., previous conversation summary)
        if task.context:
            context_str = "\n".join(f"  {k}: {v}" for k, v in task.context.items())
            prompt_parts.append(f"\nAdditional context:\n{context_str}")

        return "\n".join(prompt_parts)

    def _parse_routing_response(
        self,
        response_text: str,
        task: TaskInput
    ) -> RoutingDecision:
        """
        Parse Gemini's JSON response into a RoutingDecision object.

        WHY NOT USE response.parsed DIRECTLY:
        The google.genai SDK's automatic parsing can sometimes fail on
        complex nested schemas. Parsing manually gives us better error messages
        and control over what happens when parsing fails.

        Args:
            response_text: Raw JSON string from Gemini
            task: Original task (for error context)

        Returns:
            Validated RoutingDecision object

        Raises:
            RouterError: If JSON is invalid or doesn't match schema
        """
        try:
            # Parse JSON string to dict
            data = json.loads(response_text)

            # Validate and create RoutingDecision via Pydantic
            # This validates all fields, enum values, value ranges, etc.
            decision = RoutingDecision.model_validate(data)

            return decision

        except json.JSONDecodeError as e:
            raise RouterError(
                message=f"Router returned invalid JSON: {e}",
                details={"task_id": task.task_id, "raw_response": response_text[:200]}
            )
        except Exception as e:
            raise RouterError(
                message=f"Failed to parse routing decision: {e}",
                details={
                    "task_id": task.task_id,
                    "raw_response": response_text[:200],
                    "error": str(e)
                }
            )

    def _extract_retry_delay(self, error_str: str) -> int:
        """Extract retry delay seconds from a 429 error message. Defaults to 35s."""
        import re
        match = re.search(r"retry[_\s]delay.*?(\d+)s", error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"retryDelay.*?(\d+)", error_str)
        if match:
            return int(match.group(1))
        return 35  # Safe default

    async def execute(
        self,
        task: TaskInput,
        context: dict | None = None
    ) -> AgentOutput:
        """
        Standard AgentOutput interface wrapping route().

        The orchestrator calls route() directly.
        This execute() method exists to satisfy the BaseAgent interface
        and for cases where the router is used in a pipeline.
        """
        start_time = self._start_timer()
        try:
            decision = await self.route(task)
            return self._success(
                result=decision.model_dump(),
                start_time=start_time,
                model_used=self.settings.router_model,
            )
        except Exception as e:
            return self._failure(str(e), start_time)
