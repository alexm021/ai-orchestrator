# =============================================================================
# agents/base_agent.py — Abstract Base Class for All Agents
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Every specialized agent (Research, Outreach, Coding...) shares common needs:
#   - A logger
#   - Access to settings
#   - A standard execute() interface
#   - Timing measurement
#
# Instead of copy-pasting this setup into every agent,
# we put it here ONCE and every agent inherits it.
#
# This is the "Template Method" design pattern:
#   - BaseAgent defines the STRUCTURE (logger, settings, interface)
#   - Each subclass fills in the CONTENT (what the agent actually does)
#
# REAL BENEFIT:
# When we add retry logic, observability, or timeout handling in Phase 4,
# we add it HERE — and all agents automatically get it.
# We never touch individual agent files.
# =============================================================================

from abc import ABC, abstractmethod
from typing import Optional
import time

from core.config import settings
from core.logger import get_logger
from schemas.task import TaskInput
from schemas.agent_output import AgentOutput
from schemas.routing import AgentType
from google import genai


class BaseAgent(ABC):
    """
    Abstract base class that all specialized agents inherit from.

    To create a new agent:
        class ResearchAgent(BaseAgent):
            @property
            def agent_type(self) -> AgentType:
                return AgentType.RESEARCH

            async def execute(self, task: TaskInput, context: dict) -> AgentOutput:
                # your agent logic here
                pass
    """

    def __init__(self):
        # Every agent gets its own logger with its name
        # Logs will show: "research_agent | event=task_started"
        self.logger = get_logger(self.__class__.__name__)

        # Every agent has access to the global settings
        self.settings = settings

        # Shared Gemini client — created once, reused across calls
        # WHY: Creating a new client on every call is wasteful.
        # This client is async-capable via client.aio
        self._gemini_client = genai.Client(api_key=settings.gemini_api_key)

    # -------------------------------------------------------------------------
    # ABSTRACT PROPERTIES — Every subclass MUST implement these
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """
        Returns the AgentType enum for this agent.
        Used for logging, routing, and identification.
        """
        pass

    # -------------------------------------------------------------------------
    # ABSTRACT METHODS — Every subclass MUST implement these
    # -------------------------------------------------------------------------

    @abstractmethod
    async def execute(
        self,
        task: TaskInput,
        context: Optional[dict] = None
    ) -> AgentOutput:
        """
        The main execution method for this agent.

        Args:
            task: The TaskInput object containing the user's request
            context: Optional dict with extra context (previous agent outputs, etc.)

        Returns:
            AgentOutput with success=True and result, or success=False and error
        """
        pass

    # -------------------------------------------------------------------------
    # CONCRETE METHODS — Available to all subclasses for free
    # -------------------------------------------------------------------------

    def _start_timer(self) -> float:
        """
        Start a timer for measuring execution time.
        Call this at the beginning of execute().

        Returns:
            float: The start timestamp
        """
        return time.monotonic()

    def _elapsed_ms(self, start_time: float) -> int:
        """
        Calculate elapsed time in milliseconds since start_timer() was called.

        Args:
            start_time: The float returned by _start_timer()

        Returns:
            int: Elapsed time in milliseconds
        """
        return int((time.monotonic() - start_time) * 1000)

    def _success(
        self,
        result: dict,
        start_time: float,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> AgentOutput:
        """
        Convenience method to return a successful AgentOutput.
        Handles timing automatically.

        Usage at end of execute():
            return self._success(
                result={"summary": "...", "sources": [...]},
                start_time=start_time,
                model_used=self.settings.agent_model,
                tokens_used=response.usage_metadata.total_token_count
            )
        """
        elapsed = self._elapsed_ms(start_time)
        self.logger.info(
            "agent_completed",
            agent=self.agent_type.value,
            success=True,
            duration_ms=elapsed,
            tokens=tokens_used,
        )
        return AgentOutput.success_output(
            agent_name=self.agent_type,
            result=result,
            execution_time_ms=elapsed,
            model_used=model_used,
            tokens_used=tokens_used,
        )

    def _failure(self, error: str, start_time: float) -> AgentOutput:
        """
        Convenience method to return a failed AgentOutput.
        Handles timing and logging automatically.

        Usage in exception handlers:
            except Exception as e:
                return self._failure(str(e), start_time)
        """
        elapsed = self._elapsed_ms(start_time)
        self.logger.error(
            "agent_failed",
            agent=self.agent_type.value,
            error=error,
            duration_ms=elapsed,
        )
        return AgentOutput.failure_output(
            agent_name=self.agent_type,
            error=error,
            execution_time_ms=elapsed,
        )

    async def _gemini_generate(self, model: str, contents: str, config) -> object:
        """
        Gemini API call with smart retry + model fallback on 429.

        Strategy:
        - Per-minute rate limit → wait the suggested delay, retry same model once
        - Per-day limit → immediately skip to next model (don't waste time waiting)
        - All models exhausted → raise the last error
        """
        import asyncio as _asyncio
        import re

        fallback_models = [model, "gemini-2.5-flash", "gemini-2.5-pro"]
        seen = set()
        fallback_models = [m for m in fallback_models if not (m in seen or seen.add(m))]

        last_error = None

        for attempt_model in fallback_models:
            # Try each model up to 2 times: first attempt + one retry after waiting
            for attempt in range(2):
                try:
                    return await self._gemini_client.aio.models.generate_content(
                        model=attempt_model,
                        contents=contents,
                        config=config,
                    )
                except Exception as e:
                    last_error = e
                    error_str = str(e)

                    if "429" not in error_str and "RESOURCE_EXHAUSTED" not in error_str:
                        raise  # Non-rate-limit error — fail immediately

                    wait_sec = self._parse_retry_delay(error_str)

                    if attempt == 0 and wait_sec <= 65:
                        # Per-minute rate limit — wait the suggested time and retry once
                        wait = wait_sec + 3  # Add 3s buffer
                        self.logger.info(
                            "rate_limited_waiting",
                            model=attempt_model,
                            wait_seconds=wait,
                            agent=self.agent_type.value,
                        )
                        await _asyncio.sleep(wait)
                        # Loop back to retry same model (attempt=1)
                    else:
                        # Either second attempt failed OR wait time is too long (daily limit)
                        # Skip to next model
                        self.logger.warning(
                            "model_skipped",
                            model=attempt_model,
                            agent=self.agent_type.value,
                            retry_suggested_seconds=wait_sec,
                        )
                        break  # Move to next model

        raise last_error

    def _parse_retry_delay(self, error_str: str) -> int:
        """Parse retry delay in seconds from Gemini 429 error message."""
        import re
        # Match patterns like: "retry in 32.58s" or "'retryDelay': '54s'"
        patterns = [
            r"retry in (\d+)\.",          # "retry in 32.58s"
            r"retry in (\d+)s",           # "retry in 32s"
            r"retryDelay.*?(\d+)s",       # "'retryDelay': '54s'"
            r"retryDelay.*?'(\d+)'",      # "'retryDelay': '54'"
            r"retry_delay.*?(\d+)",       # various other formats
        ]
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if val > 0:
                    return val
        return 30  # Safe default: 30 seconds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type.value})"
