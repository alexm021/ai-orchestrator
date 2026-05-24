# =============================================================================
# api/auth.py — API Key Authentication Dependency
# =============================================================================
#
# HOW IT WORKS:
# FastAPI dependency injection — add `_: None = Depends(require_api_key)`
# to any route and it will automatically check the X-API-Key header.
#
# BEHAVIOR:
# - API_KEY not set in .env → auth disabled (dev-friendly, open access)
# - API_KEY set → header required on every protected request
# - Wrong/missing key → HTTP 401 Unauthorized
#
# USAGE IN ROUTES:
#   from api.auth import require_api_key
#   @router.post("/task")
#   async def create_task(body: TaskRequest, _: None = Depends(require_api_key)):
#       ...
# =============================================================================

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from core.config import settings
from core.logger import get_logger

logger = get_logger("api.auth")

# Tells FastAPI/Swagger to expect X-API-Key in the header.
# auto_error=False means we handle the missing key ourselves (better error msg).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """
    FastAPI dependency that enforces API key authentication.

    - If API_KEY is not configured → passes through (dev mode)
    - If API_KEY is configured and header matches → passes through
    - If API_KEY is configured and header is missing/wrong → 401
    """
    # Auth disabled — no API_KEY configured (local dev)
    if not settings.api_key:
        return

    if api_key is None:
        logger.warning("auth_missing_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Missing X-API-Key header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        logger.warning("auth_invalid_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Invalid API key.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )
