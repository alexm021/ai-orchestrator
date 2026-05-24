# =============================================================================
# api/middleware.py — Production Middleware Stack
# =============================================================================
#
# Middleware runs on EVERY request, before and after the route handler.
# Think of it as a pipeline: Request → MW1 → MW2 → Handler → MW2 → MW1 → Response
#
# WHY MIDDLEWARE INSTEAD OF PUTTING THIS IN EACH ROUTE:
# DRY principle. Request logging, timing, request IDs — every endpoint needs
# these. If we put them in each route, we'd copy-paste 20 lines everywhere,
# and forget one when we add a new route. Middleware = write once, applies
# to all routes automatically.
#
# MIDDLEWARE WE'RE ADDING:
#
# 1. RequestIDMiddleware
#    Assigns a unique short ID to every request.
#    This ID appears in ALL logs generated during that request.
#    When something breaks in production, you search for the request ID
#    and instantly see every log line from that specific request.
#    Also returned in the response as X-Request-ID header.
#
# 2. RequestLoggingMiddleware
#    Logs: method, path, status code, and response time for every request.
#    This is your access log — the standard way to audit traffic.
#    Without this, you're blind to what's hitting your API.
# =============================================================================

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.logger import get_logger

logger = get_logger("api.middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique ID to every incoming request.

    HOW IT WORKS:
    1. Generate a short UUID (8 chars — short enough to read, unique enough to track)
    2. Attach it to request.state so route handlers can access it
    3. Add it to the response as X-Request-ID header
       (so the client can include it in bug reports)

    EXAMPLE LOG OUTPUT:
        info  api_request_completed  request_id=a3f7b2c1  method=POST  path=/api/v1/task  status=200  ms=3421
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate a short, readable request ID
        # Full UUID4 = "550e8400-e29b-41d4-a716-446655440000" (36 chars, hard to read)
        # We take the first 8 chars: "550e8400" — still unique enough for logs
        request_id = str(uuid.uuid4())[:8]

        # Attach to request state — accessible in route handlers via request.state.request_id
        request.state.request_id = request_id

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        # The client can use this to reference a specific request when reporting issues
        response.headers["X-Request-ID"] = request_id

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request with method, path, status, and response time.

    This is your ACCESS LOG — the standard tool for:
    - Monitoring traffic patterns
    - Debugging slow endpoints
    - Auditing what clients are calling
    - Spotting anomalies (sudden spike in 500s, etc.)

    WHY NOT USE UVICORN'S BUILT-IN ACCESS LOG:
    Uvicorn's access log doesn't include our structlog format or request IDs.
    This middleware gives us a consistent log format across the whole app.

    SKIP PATTERNS:
    /health is called constantly by load balancers — logging every single
    health check creates noise. We skip it unless it returns an error.
    """

    # Paths to skip logging for (too noisy at high frequency)
    SKIP_PATHS = {"/api/v1/health", "/health", "/"}

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.monotonic()
        request_id = getattr(request.state, "request_id", "unknown")
        path = request.url.path
        method = request.method

        # Process the request
        response = await call_next(request)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        status = response.status_code

        # Skip successful health checks — they're too frequent to be useful
        if path in self.SKIP_PATHS and status < 400:
            return response

        # Log level based on status code:
        # 2xx/3xx → info (normal traffic)
        # 4xx → warning (client errors, worth noting)
        # 5xx → error (server errors, need attention)
        if status >= 500:
            logger.error(
                "api_request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                ms=elapsed_ms,
            )
        elif status >= 400:
            logger.warning(
                "api_request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                ms=elapsed_ms,
            )
        else:
            logger.info(
                "api_request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                ms=elapsed_ms,
            )

        return response
