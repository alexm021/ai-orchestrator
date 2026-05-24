# =============================================================================
# api/routes/task.py — POST /task endpoint (Production-Ready)
# =============================================================================

import asyncio
import time
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.auth import require_api_key
from api.schemas import TaskRequest, TaskResponse, PerformanceInfo, RoutingInfo
from orchestrator.graph import run_task
from core.logger import get_logger

router = APIRouter()
logger = get_logger("api.task")

# Maximum time a task is allowed to run before we cut it off.
# Free tier Gemini can have 60s rate-limit waits — give enough headroom.
# Production with billing: can lower to 60s.
TASK_TIMEOUT_SECONDS = 180


def _is_quota_error(result: dict) -> bool:
    """Check if a failed task result was caused by API quota exhaustion."""
    errors = result.get("errors", [])
    return any("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) for e in errors)


@router.post(
    "/task",
    summary="Run an orchestration task",
    description=(
        "Submit a task to the multi-agent orchestrator. The system automatically "
        "routes it to the right agent(s) and returns structured results.\n\n"
        "**Status codes:**\n"
        "- `200` Task completed successfully\n"
        "- `422` Invalid request (missing/bad fields)\n"
        "- `503` AI service temporarily unavailable (quota exceeded — retry later)\n"
        "- `504` Task timed out\n"
        "- `500` Internal server error"
    ),
    responses={
        200: {"description": "Task completed successfully"},
        422: {"description": "Validation error"},
        503: {"description": "AI quota exceeded — retry after cooldown"},
        504: {"description": "Task execution timed out"},
        500: {"description": "Internal server error"},
    },
)
async def create_task(request_body: TaskRequest, request: Request, _: None = Depends(require_api_key)):
    """
    Execute a task through the multi-agent orchestrator.

    Returns HTTP 200 on success with the full result.
    Returns HTTP 503 if the AI quota is exhausted (retry in a few minutes).
    Returns HTTP 504 if the task takes longer than 3 minutes.
    """
    wall_start = time.monotonic()
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "api_task_received",
        request_id=request_id,
        message_length=len(request_body.message),
        priority=request_body.priority,
    )

    try:
        # ---------------------------------------------------------------
        # TIMEOUT WRAPPER
        # asyncio.wait_for() cancels the coroutine if it doesn't complete
        # within TASK_TIMEOUT_SECONDS. Without this, a task waiting on
        # rate-limited API calls could hang for 10+ minutes, blocking the
        # event loop for other requests.
        # ---------------------------------------------------------------
        result = await asyncio.wait_for(
            run_task(
                task_message=request_body.message,
                task_priority=request_body.priority,
                task_context=request_body.context,
            ),
            timeout=TASK_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        wall_ms = int((time.monotonic() - wall_start) * 1000)
        logger.error(
            "api_task_timeout",
            request_id=request_id,
            wall_ms=wall_ms,
            timeout_s=TASK_TIMEOUT_SECONDS,
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "timeout",
                "message": (
                    f"Task exceeded the {TASK_TIMEOUT_SECONDS}s time limit. "
                    "The AI service may be rate-limited. Please retry in a few minutes."
                ),
                "request_id": request_id,
            },
        )

    except Exception as e:
        wall_ms = int((time.monotonic() - wall_start) * 1000)
        logger.error(
            "api_task_exception",
            request_id=request_id,
            error=str(e),
            wall_ms=wall_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "orchestration_error",
                "message": "An unexpected error occurred during task execution.",
                "request_id": request_id,
            },
        )

    wall_ms = int((time.monotonic() - wall_start) * 1000)

    # ---------------------------------------------------------------
    # QUOTA ERROR → HTTP 503
    # When Gemini returns 429 (quota exhausted), the orchestrator
    # returns status="failed" with the 429 error in the errors list.
    # We detect this and translate it to HTTP 503 + Retry-After header.
    # 503 = "Service Unavailable" — the standard code for "try again later"
    # Retry-After tells the client how long to wait.
    # ---------------------------------------------------------------
    if result.get("status") == "failed" and _is_quota_error(result):
        logger.warning(
            "api_task_quota_exceeded",
            request_id=request_id,
            wall_ms=wall_ms,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "60"},
            content={
                "error": "quota_exceeded",
                "message": (
                    "The AI service quota is temporarily exhausted. "
                    "Please retry in 60 seconds."
                ),
                "request_id": request_id,
            },
        )

    logger.info(
        "api_task_completed",
        request_id=request_id,
        task_id=result.get("task_id"),
        status=result.get("status"),
        agents=result.get("agents_used", []),
        wall_ms=wall_ms,
    )

    # Build clean response
    perf = result.get("performance", {})
    routing = result.get("routing", {})

    return TaskResponse(
        task_id=result.get("task_id", "unknown"),
        status=result.get("status", "failed"),
        agents_used=result.get("agents_used", []),
        execution_strategy=result.get("execution_strategy"),
        primary_output=result.get("primary_output"),
        all_outputs=result.get("all_outputs", {}),
        performance=PerformanceInfo(
            total_tokens=perf.get("total_tokens", 0),
            total_execution_ms=perf.get("total_execution_ms", 0),
            tokens_by_agent=perf.get("tokens_by_agent", {}),
        ),
        routing=RoutingInfo(
            task_type=routing.get("task_type"),
            confidence=routing.get("confidence"),
            primary_agent=routing.get("primary_agent"),
            reasoning=routing.get("reasoning"),
        ),
        total_wall_time_ms=result.get("total_wall_time_ms", wall_ms),
        errors=result.get("errors", []),
    )
