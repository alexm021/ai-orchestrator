# =============================================================================
# agents/coding_agent.py — Coding Agent
# =============================================================================
#
# PURPOSE:
# Handles all code-related tasks: writing, debugging, reviewing, explaining.
#
# WHAT MAKES THIS AGENT DIFFERENT:
# Code output must be PRECISE. One wrong character breaks execution.
# So this agent uses the lowest temperature (0.1) of all agents.
#
# Also: Gemini 2.5 Flash has strong coding capabilities — it understands
# algorithms, data structures, debugging patterns, and best practices.
#
# OUTPUT STRUCTURE:
#   {
#     "task_type": "write/debug/review/explain",
#     "language": "python/javascript/etc",
#     "solution": "the complete code solution",
#     "explanation": "what the code does and why",
#     "issues_found": ["bug 1", "bug 2"] (for debug/review tasks),
#     "best_practices": ["tip 1", "tip 2"]
#   }
#
# TEMPERATURE 0.1:
# Code is deterministic. Low temperature = consistent, predictable output.
# We don't want the agent to "get creative" with variable names or algorithms.
# =============================================================================

import json
from pydantic import BaseModel, Field
from typing import Optional
from google.genai import types

from agents.base_agent import BaseAgent
from schemas.task import TaskInput
from schemas.agent_output import AgentOutput
from schemas.routing import AgentType


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

class CodingOutput(BaseModel):
    """Structured output from the Coding Agent."""
    task_type: str = Field(..., description="Type of coding task: write/debug/review/explain/refactor")
    language: str = Field(..., description="Programming language used or detected")
    solution: str = Field(..., description="The complete code solution or reviewed code")
    explanation: str = Field(..., description="Clear explanation of what the code does and key decisions")
    issues_found: list[str] = Field(default_factory=list, description="List of bugs or issues found (for debug/review tasks)")
    best_practices: list[str] = Field(default_factory=list, description="Relevant best practices or improvement suggestions")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

CODING_SYSTEM_PROMPT = """You are the Coding Agent in an AI Multi-Agent Orchestrator system.

Your role is to handle all code-related tasks with precision and clarity.

## Your Responsibilities
- Write clean, working, production-ready code
- Debug and fix code issues with clear explanations of the root cause
- Review code for bugs, security issues, and quality problems
- Explain code in plain language for any skill level

## Task Types You Handle
- **write**: Write new code from a specification or description
- **debug**: Find and fix bugs in provided code
- **review**: Analyze code quality, find issues, suggest improvements
- **explain**: Explain what code does in plain language
- **refactor**: Improve code structure without changing behavior

## Code Quality Standards
- Always write complete, runnable code (no placeholders like "# TODO: implement this")
- Add comments for non-obvious logic
- Follow language-specific conventions (PEP8 for Python, etc.)
- Include error handling where appropriate
- Prefer readability over cleverness

## When Debugging
1. Identify the root cause first, not just the symptom
2. Explain WHY the bug occurs, not just how to fix it
3. Show the fixed code in full, not just the changed lines
4. Mention if there are related issues the user should know about

## When Writing New Code
1. Start with the simplest working solution
2. Add complexity only if needed
3. Include a brief usage example in the explanation
4. Mention edge cases that should be handled

Always provide complete, tested-in-mind solutions. Never give partial code.
"""


# =============================================================================
# CODING AGENT
# =============================================================================

class CodingAgent(BaseAgent):
    """
    Coding Agent — writes, debugs, reviews, and explains code.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CODING

    async def execute(
        self,
        task: TaskInput,
        context: dict | None = None
    ) -> AgentOutput:
        """
        Handle the coding task in task.user_message.

        The agent auto-detects the task type (write/debug/review/explain)
        from the user's message — no explicit classification needed.
        """
        start_time = self._start_timer()

        self.logger.info(
            "coding_started",
            task_id=task.task_id,
            message_length=len(task.user_message),
        )

        try:
            prompt = self._build_prompt(task, context)

            response = await self._gemini_generate(
                model=self.settings.agent_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=CODING_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=CodingOutput,
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )

            data = json.loads(response.text)
            output = CodingOutput.model_validate(data)

            tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", None)

            self.logger.info(
                "coding_completed",
                task_id=task.task_id,
                task_type=output.task_type,
                language=output.language,
                issues_found=len(output.issues_found),
            )

            return self._success(
                result=output.model_dump(),
                start_time=start_time,
                model_used=self.settings.agent_model,
                tokens_used=tokens,
            )

        except Exception as e:
            self.logger.error("coding_failed", task_id=task.task_id, error=str(e))
            return self._failure(str(e), start_time)

    def _build_prompt(self, task: TaskInput, context: dict | None) -> str:
        """Build coding prompt with optional context (e.g., related code files)."""
        parts = [f"Coding task:\n{task.user_message}"]

        if context:
            if "code_context" in context:
                parts.append(f"\nRelated code context:\n```\n{context['code_context']}\n```")
            if "language_hint" in context:
                parts.append(f"\nLanguage: {context['language_hint']}")

        return "\n".join(parts)
