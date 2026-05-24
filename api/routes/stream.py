# =============================================================================
# api/routes/stream.py -- POST /task/stream (Server-Sent Events)
# =============================================================================
#
# WHY STREAMING:
# The regular POST /task waits 5-15 seconds then returns everything at once.
# Streaming sends events as they happen, so the client sees progress immediately.
# This is how ChatGPT works -- tokens appear as they're generated.
#
# SSE EVENT SEQUENCE:
#   1. {"type": "status",      "message": "Routing task..."}
#   2. {"type": "routing",     "agents": [...], "strategy": "...", ...}
#   3. {"type": "agent_start", "agent": "coding", "step": "1/1"}
#   4. {"type": "chunk",       "agent": "coding", "text": "def hello..."}
#      ... (many chunk events, one per Gemini token batch)
#   5. {"type": "agent_done",  "agent": "coding"}
#   6. {"type": "done",        "task_id": "...", "status": "complete"}
#
# CLIENT USAGE (JavaScript):
#   const es = new EventSource('/api/v1/task/stream?message=...');
#   es.onmessage = (e) => {
#     const event = JSON.parse(e.data);
#     if (event.type === 'chunk') appendText(event.text);
#     if (event.type === 'done')  es.close();
#   };
#
# NOTE:
# The streaming endpoint uses free-text Gemini output (no response_schema).
# Structured JSON output requires the full response before parsing.
# For streaming, we prioritize token-by-token text delivery.
# =============================================================================

import json
import uuid
import asyncio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from api.auth import require_api_key
from api.limiter import limiter
from api.schemas import TaskRequest
from core.config import settings
from core.logger import get_logger

router = APIRouter()
logger = get_logger("api.stream")


# Per-agent system prompts for streaming mode (same quality, no response_schema)
_AGENT_PROMPTS = {
    "coding": (
        "You are an expert software engineer. Write clean, working, production-ready code. "
        "Include a brief explanation of what the code does and any important edge cases."
    ),
    "research": (
        "You are a research analyst. Provide comprehensive, well-structured analysis. "
        "Include key facts, important insights, and a clear summary."
    ),
    "summarization": (
        "You are a summarization expert. Condense the topic into clear, structured output. "
        "Use bullet points for key takeaways and end with actionable next steps."
    ),
    "outreach": (
        "You are an expert in professional communication and sales. "
        "Write a compelling, personalized message that is concise and has a clear call to action."
    ),
    "general": (
        "You are a helpful, knowledgeable AI assistant. "
        "Provide a clear, thorough response."
    ),
}


async def _event_generator(task_message: str, priority: str, context: dict, request_id: str):
    """
    Async generator that yields JSON-encoded SSE event strings.

    Each yielded string becomes one SSE message on the client.
    The generator handles the full lifecycle: routing -> execution -> done.
    """
    from google import genai
    from google.genai import types
    from agents.router_agent import RouterAgent
    from schemas.task import TaskInput

    task_id = str(uuid.uuid4())
    gemini = genai.Client(api_key=settings.gemini_api_key)

    logger.info("stream_started", task_id=task_id, request_id=request_id)

    try:
        # ---------------------------------------------------------------
        # STEP 1: ROUTING
        # ---------------------------------------------------------------
        yield json.dumps({"type": "status", "message": "Routing task..."})

        task = TaskInput(
            task_id=task_id,
            user_message=task_message,
            priority=priority,
            context=context,
        )

        router_agent = RouterAgent()
        decision = await router_agent.route(task)

        agents = decision.agent_names()

        yield json.dumps({
            "type": "routing",
            "task_id": task_id,
            "agents": agents,
            "strategy": decision.execution_strategy.value,
            "confidence": round(decision.confidence, 2),
            "reasoning": decision.reasoning,
        })

        logger.info("stream_routed", task_id=task_id, agents=agents)

        # ---------------------------------------------------------------
        # STEP 2: EXECUTE AGENTS (STREAMING)
        # Each agent streams tokens directly to the client.
        # Sequential: previous agent's full output is passed as context.
        # ---------------------------------------------------------------
        accumulated_context = ""

        for i, agent_name in enumerate(agents):
            # Skip retrieval in streaming mode (no LLM output to stream)
            if agent_name == "retrieval":
                yield json.dumps({"type": "agent_start", "agent": "retrieval", "step": f"{i+1}/{len(agents)}"})
                yield json.dumps({"type": "agent_done", "agent": "retrieval"})
                continue

            yield json.dumps({
                "type": "agent_start",
                "agent": agent_name,
                "step": f"{i+1}/{len(agents)}",
            })

            system_prompt = _AGENT_PROMPTS.get(agent_name, _AGENT_PROMPTS["general"])

            # Build prompt — inject previous agent's output as context
            if accumulated_context:
                prompt = (
                    f"Context from previous analysis:\n{accumulated_context}\n\n"
                    f"Now complete this task:\n{task_message}"
                )
            else:
                prompt = task_message

            full_output = ""

            # Stream tokens from Gemini
            async for chunk in await gemini.aio.models.generate_content_stream(
                model=settings.agent_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            ):
                if chunk.text:
                    full_output += chunk.text
                    yield json.dumps({
                        "type": "chunk",
                        "agent": agent_name,
                        "text": chunk.text,
                    })

            accumulated_context = full_output
            yield json.dumps({"type": "agent_done", "agent": agent_name})

        # ---------------------------------------------------------------
        # STEP 3: DONE
        # ---------------------------------------------------------------
        yield json.dumps({
            "type": "done",
            "task_id": task_id,
            "status": "complete",
            "agents_used": agents,
        })

        logger.info("stream_completed", task_id=task_id, agents=agents)

    except asyncio.CancelledError:
        # Client disconnected — normal, not an error
        logger.info("stream_cancelled", task_id=task_id)

    except Exception as e:
        logger.error("stream_failed", task_id=task_id, error=str(e))
        yield json.dumps({"type": "error", "message": str(e)})


@router.post(
    "/task/stream",
    summary="Stream a task (Server-Sent Events)",
    description=(
        "Submit a task and receive results as a real-time stream.\n\n"
        "Returns `text/event-stream` — each event is a JSON object:\n"
        "- `status` — progress message\n"
        "- `routing` — which agents will run\n"
        "- `agent_start` — an agent has started\n"
        "- `chunk` — a text fragment from the agent (stream these to the user)\n"
        "- `agent_done` — an agent finished\n"
        "- `done` — all agents finished\n"
        "- `error` — something went wrong\n\n"
        "**Rate limit:** 10 requests/minute per IP"
    ),
    response_class=EventSourceResponse,
)
@limiter.limit("10/minute")
async def stream_task(
    request: Request,
    request_body: TaskRequest,
    _: None = Depends(require_api_key),
):
    """
    Stream a task through the multi-agent orchestrator.

    Tokens are delivered as they're generated — no waiting for the full response.
    Connect with EventSource (JavaScript) or any SSE client.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "stream_task_received",
        request_id=request_id,
        message_length=len(request_body.message),
    )

    return EventSourceResponse(
        _event_generator(
            task_message=request_body.message,
            priority=request_body.priority,
            context=request_body.context,
            request_id=request_id,
        )
    )
