# =============================================================================
# agents/research_agent.py — Research Agent
# =============================================================================
#
# PURPOSE:
# The Research Agent analyzes topics and returns structured information.
# It does NOT search the web in Phase 3 (we add real tools in Phase 5).
# It uses Gemini's built-in knowledge to produce research-quality output.
#
# WHY GEMINI'S KNOWLEDGE IS ENOUGH FOR NOW:
# Gemini 2.5 Flash has a knowledge cutoff of early 2025 and knows about
# most companies, technologies, concepts, and market trends.
# For a portfolio demo, this is sufficient to prove the agent architecture.
# Web search tools (Serper API, Tavily) will be added in Phase 5.
#
# OUTPUT STRUCTURE:
#   {
#     "topic": "what was researched",
#     "summary": "3-4 sentence executive summary",
#     "key_facts": ["fact 1", "fact 2", ...],
#     "insights": ["insight 1", "insight 2", ...],
#     "confidence": 0.0-1.0,
#     "limitations": "what this research doesn't cover"
#   }
#
# TEMPERATURE 0.3:
# Research needs to be mostly factual (low temp) but benefits from slight
# creativity in synthesizing insights (slightly above 0).
# =============================================================================

import json
from pydantic import BaseModel, Field
from google.genai import types

from agents.base_agent import BaseAgent
from schemas.task import TaskInput
from schemas.agent_output import AgentOutput
from schemas.routing import AgentType


# =============================================================================
# OUTPUT SCHEMA — What this agent returns inside AgentOutput.result
# =============================================================================

class ResearchOutput(BaseModel):
    """
    Structured output from the Research Agent.
    Sent to Gemini as response_schema — forces structured JSON back.
    """
    topic: str = Field(..., description="The main topic that was researched")
    summary: str = Field(..., description="3-4 sentence executive summary of findings")
    key_facts: list[str] = Field(..., min_length=2, description="List of concrete, verifiable facts")
    insights: list[str] = Field(..., min_length=1, description="Analytical insights and implications")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the research quality")
    limitations: str = Field(..., description="What this research does not cover or may be outdated on")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent in an AI Multi-Agent Orchestrator system.

Your role is to analyze topics and produce structured, high-quality research output.

## Your Responsibilities
- Synthesize accurate information about the requested topic
- Identify concrete, verifiable facts (not opinions)
- Extract meaningful insights and implications
- Be honest about the confidence level and limitations of your knowledge
- Provide actionable, relevant information

## Output Quality Standards
- Facts must be specific and concrete, not vague generalizations
- Insights must go beyond facts — they should reveal patterns, implications, or non-obvious connections
- Summary should be executive-level: clear, concise, immediately useful
- Confidence should reflect how well-known this topic is and how current your information is

## Confidence Scoring Guide
- 0.9-1.0: Well-established topic, widely documented, unlikely to change
- 0.7-0.9: Good coverage but some details may be outdated
- 0.5-0.7: Topic is known but information may be incomplete or evolving
- Below 0.5: Limited information, user should verify independently

Always return complete, accurate research. Never fabricate facts.
"""


# =============================================================================
# RESEARCH AGENT
# =============================================================================

class ResearchAgent(BaseAgent):
    """
    Research Agent — analyzes topics and returns structured research output.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RESEARCH

    async def execute(
        self,
        task: TaskInput,
        context: dict | None = None
    ) -> AgentOutput:
        """
        Research the topic in the task message.

        Args:
            task: TaskInput with the research topic in user_message
            context: Optional — if provided, enriches the research query

        Returns:
            AgentOutput with result containing ResearchOutput fields
        """
        start_time = self._start_timer()

        self.logger.info(
            "research_started",
            task_id=task.task_id,
            topic_length=len(task.user_message),
        )

        try:
            # Build research prompt — include any provided context
            prompt = self._build_prompt(task, context)

            # Call Gemini with structured output
            response = await self._gemini_generate(
                model=self.settings.agent_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RESEARCH_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ResearchOutput,
                    temperature=0.3,
                    max_output_tokens=8192,
                ),
            )

            # Parse and validate the response
            data = json.loads(response.text)
            output = ResearchOutput.model_validate(data)

            # Extract token usage if available
            tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", None)

            self.logger.info(
                "research_completed",
                task_id=task.task_id,
                topic=output.topic,
                facts_count=len(output.key_facts),
                confidence=output.confidence,
            )

            return self._success(
                result=output.model_dump(),
                start_time=start_time,
                model_used=self.settings.agent_model,
                tokens_used=tokens,
            )

        except Exception as e:
            self.logger.error("research_failed", task_id=task.task_id, error=str(e))
            return self._failure(str(e), start_time)

    def _build_prompt(self, task: TaskInput, context: dict | None) -> str:
        """Build the research prompt, incorporating any available context."""
        parts = [f"Research request:\n{task.user_message}"]

        # If context has previous research or additional info, include it
        if context:
            if "additional_context" in context:
                parts.append(f"\nAdditional context:\n{context['additional_context']}")
            if "focus_areas" in context:
                parts.append(f"\nFocus especially on: {context['focus_areas']}")

        return "\n".join(parts)
