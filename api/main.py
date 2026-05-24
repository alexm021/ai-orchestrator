# =============================================================================
# api/main.py — FastAPI Application (Production-Ready)
# =============================================================================

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from core.config import settings
from core.logger import setup_logging, get_logger
from api.routes import task, health, memory, stream
from api.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from api.limiter import limiter

logger = get_logger("api.main")


# =============================================================================
# LIFESPAN — startup and shutdown hooks
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-warm everything. Shutdown: log gracefully."""

    # --- STARTUP ---
    setup_logging()
    logger.info(
        "api_starting",
        app=settings.app_name,
        env=settings.app_env,
        host=settings.api_host,
        port=settings.api_port,
    )

    # Validate critical config — fail fast with a clear message
    # Better to crash at startup than to serve broken requests
    if not settings.gemini_api_key or len(settings.gemini_api_key) < 10:
        logger.error("startup_failed", reason="GEMINI_API_KEY is missing or invalid")
        raise RuntimeError("GEMINI_API_KEY is not configured. Check your .env file.")

    # Pre-warm LangGraph — compile graph once at startup, not on first request
    try:
        from orchestrator.graph import get_graph
        get_graph()
        logger.info("langgraph_prewarmed")
    except Exception as e:
        logger.warning("langgraph_prewarm_failed", error=str(e))

    # Pre-initialize ChromaDB connection
    try:
        from core.memory import get_memory_manager
        mem = get_memory_manager()
        logger.info("memory_prewarmed", memories=mem.count())
    except Exception as e:
        logger.warning("memory_prewarm_failed", error=str(e))

    logger.info(
        "api_ready",
        docs_url=f"http://localhost:{settings.api_port}/docs",
        task_timeout_s=settings.agent_timeout_seconds * 4,
    )

    yield  # ← Server runs here

    # --- SHUTDOWN ---
    logger.info("api_shutting_down", app=settings.app_name)


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="AI Multi-Agent Orchestrator",
    description=(
        "A production-grade multi-agent AI system.\n\n"
        "Submit any task in plain text — the orchestrator automatically routes it "
        "to the right AI agents, executes them in sequence or parallel, "
        "and returns structured JSON results.\n\n"
        "**Available agents:** research, summarization, outreach, coding, retrieval\n\n"
        "**Example:** `POST /api/v1/task` with "
        "`{\"message\": \"Research Anthropic then write a cold email to their Head of Product\"}`"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Attach the rate limiter to the app so slowapi can track request counts
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# MIDDLEWARE
# NOTE: Middleware is applied in REVERSE registration order.
# RequestID must run FIRST (outermost), so register it LAST.
# Order of execution: RequestID → RequestLogging → CORS → Handler
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)  # Registered last = runs first


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================
# Without these, FastAPI returns HTML error pages for unhandled exceptions.
# In an API, everything must be JSON — no exceptions.
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override FastAPI's default validation error format.
    Returns a cleaner, consistent JSON structure.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error.get("loc", []))
        errors.append({"field": field, "message": error.get("msg", "Invalid value")})

    logger.warning(
        "validation_error",
        request_id=request_id,
        path=request.url.path,
        errors=errors,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": errors,
            "request_id": request_id,
        },
    )


@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
    """Task exceeded the maximum allowed execution time."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "request_timeout",
        request_id=request_id,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=504,
        content={
            "error": "timeout",
            "message": "Task execution timed out. Try a simpler task or retry later.",
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for any unhandled exception.
    Logs the full error, returns clean JSON (no stack traces to clients).

    WHY HIDE STACK TRACES:
    Stack traces reveal your internal structure to potential attackers.
    Log them server-side (where only you can see), never send to clients.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled_exception",
        request_id=request_id,
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Please try again.",
            "request_id": request_id,
        },
    )


# =============================================================================
# ROUTES
# =============================================================================

app.include_router(task.router,   prefix="/api/v1", tags=["Tasks"])
app.include_router(stream.router, prefix="/api/v1", tags=["Tasks"])
app.include_router(health.router, prefix="/api/v1", tags=["System"])
app.include_router(memory.router, prefix="/api/v1", tags=["Memory"])


# =============================================================================
# ROOT
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")
