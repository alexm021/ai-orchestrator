# =============================================================================
# agents/summarization_agent.py — Summarization Agent
# =============================================================================
#
# PURPOSE:
# Condenses long text into structured, scannable summaries.
# Used both as a standalone agent and as a post-processing step in pipelines.
#
# EXAMPLE PIPELINE USE:
# Research Agent returns 500 words of findings
# → Orchestrator passes it to Summarization Agent
# → Returns 5 bullet points for the final user response
#
# OUTPUT STRUCTURE:
#   {
#     "summary": "1-2 sentence TL;DR",
#     "key_points": ["point 1", "point 2", ...],
#     "main_topic": "one-phrase topic label",
#     "tone": "formal/technical/casual/etc",
#     "compression_ratio": "e.g. 85% reduction"
#   }
#
# TEMPERATURE 0.2:
# Summarization needs to stay faithful to the source text.
# Very low temperature = minimal paraphrasing, maximum fidelity.
# =============================================================================

import json
from pydantic import BaseModel, Field
from google.genai import types

from agents.base_agent import BaseAgent
from schemas.task import TaskInput
from schemas.agent_output import AgentOutput
from schemas.routing import AgentType


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

class SummarizationOutput(BaseModel):
    """Structured output from the Summarization Agent."""
    summary: str = Field(..., description="1-2 sentence TL;DR of the entire content")
    key_points: list[str] = Field(..., min_length=2, description="Most important points as bullet items")
    main_topic: str = Field(..., description="One-phrase label for the main topic")
    tone: str = Field(..., description="Tone of the original content: formal/technical/casual/journalistic")
    compression_ratio: str = Field(..., description="Approximate compression e.g. '80% shorter'")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SUMMARIZATION_SYSTEM_PROMPT = """You are the Summarization Agent in an AI Multi-Agent Orchestrator system.

Your role is to condense content into clear, accurate, and scannable summaries.

## Your Responsibilities
- Preserve the most important information, discard filler
- Never add information that wasn't in the original text
- Use plain, accessible language unless the original is highly technical
- Structure output for maximum readability

## Key Points Quality Standards
- Each point must be self-contained and understandable without the others
- Lead with the most important point
- Be specific — avoid vague points like "AI is important"
- 3-7 key points is ideal. More than 7 defeats the purpose of summarization.

## Tone Detection Guide
- formal: academic papers, legal documents, official reports
- technical: engineering docs, scientific papers, API documentation
- business: corporate reports, business news, market analysis
- casual: blog posts, social media, informal articles
- journalistic: news articles, press releases

Always stay faithful to the source. Never editorialize or add opinions.
"""


# =============================================================================
# SUMMARIZATION AGENT
# =============================================================================

class SummarizationAgent(BaseAgent):
    """
    Summarization Agent — condenses text into structured bullet summaries.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SUMMARIZATION

    async def execute(
        self,
        task: TaskInput,
        context: dict | None = None
    ) -> AgentOutput:
        """
        Summarize the text in the task message (or from context if piped).

        Context-awareness:
            If context contains "content_to_summarize", use that.
            This allows Research Agent → Summarization Agent pipelines.
            Otherwise, summarize the task.user_message directly.
        """
        start_time = self._start_timer()

        # Determine what text to summarize
        # Priority: explicit context content > task message
        if context and "content_to_summarize" in context:
            text_to_summarize = context["content_to_summarize"]
            source = "pipeline_context"
        else:
            text_to_summarize = task.user_message
            source = "user_message"

        self.logger.info(
            "summarization_started",
            task_id=task.task_id,
            source=source,
            text_length=len(text_to_summarize),
        )

        try:
            prompt = f"Summarize the following content:\n\n{text_to_summarize}"

            response = await self._gemini_generate(
                model=self.settings.agent_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=SummarizationOutput,
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )

            data = json.loads(response.text)
            output = SummarizationOutput.model_validate(data)

            tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", None)

            self.logger.info(
                "summarization_completed",
                task_id=task.task_id,
                topic=output.main_topic,
                points_count=len(output.key_points),
                compression=output.compression_ratio,
            )

            return self._success(
                result=output.model_dump(),
                start_time=start_time,
                model_used=self.settings.agent_model,
                tokens_used=tokens,
            )

        except Exception as e:
            self.logger.error("summarization_failed", task_id=task.task_id, error=str(e))
            return self._failure(str(e), start_time)
