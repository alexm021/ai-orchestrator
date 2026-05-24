# =============================================================================
# agents/outreach_agent.py — Outreach Agent
# =============================================================================
#
# PURPOSE:
# Writes professional outreach communication: cold emails, LinkedIn messages,
# follow-ups, partnership proposals.
#
# CONTEXT-AWARENESS — KEY FEATURE:
# This agent is designed to consume ResearchAgent output as context.
# When given research about a company/person, it writes PERSONALIZED outreach.
# This is the first real example of agent-to-agent data flow in the system.
#
# WITHOUT RESEARCH CONTEXT:
#   "Write a cold email to Tesla's VP of Sales"
#   → Generic email based on general knowledge of Tesla
#
# WITH RESEARCH CONTEXT (research agent ran first):
#   context["research_output"] = {topic, summary, key_facts, insights}
#   → Personalized email referencing specific Tesla initiatives and facts
#   → Much higher response rate in practice
#
# OUTPUT STRUCTURE:
#   {
#     "subject": "email subject line",
#     "body": "full email body",
#     "tone": "professional/casual/executive",
#     "call_to_action": "what we want the recipient to do",
#     "personalization_notes": "what was personalized and why"
#   }
#
# TEMPERATURE 0.7:
# Writing tasks need creativity. Higher temperature = more natural, engaging copy.
# Not too high (above 0.9) — we still need professional, coherent output.
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

class OutreachOutput(BaseModel):
    """Structured output from the Outreach Agent."""
    subject: str = Field(..., description="Email subject line — compelling and specific")
    body: str = Field(..., description="Full email body text, ready to send")
    tone: str = Field(..., description="Tone used: professional/executive/casual/direct")
    call_to_action: str = Field(..., description="The specific action requested from the recipient")
    personalization_notes: str = Field(..., description="What was personalized and what research was used")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

OUTREACH_SYSTEM_PROMPT = """You are the Outreach Agent in an AI Multi-Agent Orchestrator system.

Your role is to write high-converting professional outreach communications.

## Your Responsibilities
- Write emails that get opened, read, and responded to
- Personalize content using any research context provided
- Keep emails concise — under 150 words for cold outreach
- Always include a clear, low-friction call to action
- Mirror the appropriate professional tone for the recipient

## Email Quality Standards
SUBJECT LINE:
- Under 50 characters
- Specific, not generic ("Your Q3 expansion into DACH" not "Partnership opportunity")
- No all-caps, no spam trigger words, no excessive punctuation

BODY:
- Opening: reference something specific about them (not "I hope this finds you well")
- Value proposition: one sentence on what you offer and why it matters to THEM
- Social proof: one credibility signal (client, result, metric) — keep it brief
- CTA: single, specific ask with low commitment ("15-min call" not "partnership discussion")
- Signature placeholder: [Your Name] / [Company] / [Contact]

TONE GUIDE:
- C-suite (CEO, CTO, VP): Executive tone — respect time, lead with business impact
- Sales/Marketing: Peer tone — results-focused, direct
- Technical (Engineers, CTOs): Technical tone — skip the fluff, show you understand the domain
- Founders/Startups: Casual but sharp — they value authenticity over polish

## When Research Context Is Provided
Use the research data to:
1. Reference a specific initiative, product, or milestone they have
2. Connect your offer directly to their current situation
3. Show you've done your homework — this dramatically increases response rates

Never write generic emails. Every email must feel handwritten for this specific recipient.
"""


# =============================================================================
# OUTREACH AGENT
# =============================================================================

class OutreachAgent(BaseAgent):
    """
    Outreach Agent — writes personalized professional outreach emails.

    Consumes ResearchAgent output as context for personalization.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.OUTREACH

    async def execute(
        self,
        task: TaskInput,
        context: dict | None = None
    ) -> AgentOutput:
        """
        Write outreach email based on task and optional research context.

        Args:
            task: TaskInput describing who to write to and what to say
            context: Optional dict. If it contains "research_output" key,
                     that research will be used for personalization.
                     Example:
                     {
                         "research_output": {
                             "topic": "Tesla Inc",
                             "summary": "Tesla is expanding into...",
                             "key_facts": ["..."],
                             "insights": ["..."]
                         }
                     }
        """
        start_time = self._start_timer()

        has_research = bool(context and "research_output" in context)
        self.logger.info(
            "outreach_started",
            task_id=task.task_id,
            has_research_context=has_research,
        )

        try:
            prompt = self._build_prompt(task, context)

            response = await self._gemini_generate(
                model=self.settings.agent_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=OUTREACH_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=OutreachOutput,
                    temperature=0.7,
                    max_output_tokens=8192,
                ),
            )

            data = json.loads(response.text)
            output = OutreachOutput.model_validate(data)

            tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", None)

            self.logger.info(
                "outreach_completed",
                task_id=task.task_id,
                tone=output.tone,
                subject_length=len(output.subject),
                body_length=len(output.body),
                personalized=has_research,
            )

            return self._success(
                result=output.model_dump(),
                start_time=start_time,
                model_used=self.settings.agent_model,
                tokens_used=tokens,
            )

        except Exception as e:
            self.logger.error("outreach_failed", task_id=task.task_id, error=str(e))
            return self._failure(str(e), start_time)

    def _build_prompt(self, task: TaskInput, context: dict | None) -> str:
        """
        Build the outreach prompt, enriching with research context if available.

        This is where agent-to-agent data flow happens:
        ResearchAgent output → injected into OutreachAgent prompt.
        """
        parts = [f"Outreach request:\n{task.user_message}"]

        if context and "research_output" in context:
            research = context["research_output"]
            # Format research data cleanly for the prompt
            parts.append("\n--- RESEARCH CONTEXT (use for personalization) ---")
            parts.append(f"Topic: {research.get('topic', 'N/A')}")
            parts.append(f"Summary: {research.get('summary', 'N/A')}")

            if research.get("key_facts"):
                facts = "\n".join(f"  - {f}" for f in research["key_facts"])
                parts.append(f"Key Facts:\n{facts}")

            if research.get("insights"):
                insights = "\n".join(f"  - {i}" for i in research["insights"])
                parts.append(f"Insights:\n{insights}")

            parts.append("--- END RESEARCH CONTEXT ---")
            parts.append("\nUse the above research to personalize the outreach significantly.")

        return "\n".join(parts)
