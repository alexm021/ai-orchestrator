# =============================================================================
# api/routes/health.py — GET /health endpoint
# =============================================================================
#
# The health check is the FIRST thing a production deployment checks.
# Load balancers, Kubernetes, and CI/CD pipelines hit /health to decide:
#   - Is this instance ready to accept traffic?
#   - Should it be restarted?
#   - Is the deployment healthy?
#
# A good health check is:
#   - FAST (< 100ms, no external API calls)
#   - MEANINGFUL (tells you the actual system state)
#   - SAFE (read-only, never mutates state)
#
# We check:
#   1. Can we read from the config? (catches bad env setup)
#   2. Is ChromaDB accessible? (catches storage issues)
#   3. Are agents registered? (catches import errors)
# =============================================================================

from fastapi import APIRouter
from api.schemas import HealthResponse
from core.config import settings
from core.memory import get_memory_manager
from orchestrator.nodes import AGENT_REGISTRY
from core.logger import get_logger

router = APIRouter()
logger = get_logger("api.health")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description=(
        "Returns the current health status of the orchestrator. "
        "Checks config, ChromaDB memory store, and agent registry. "
        "Safe to poll frequently — no API calls made."
    ),
)
async def health_check() -> HealthResponse:
    """
    Quick system health check.

    Checks three things without making any external API calls:
    1. Config loaded (Gemini API key present)
    2. ChromaDB accessible (can query memory count)
    3. Agents registered (registry is non-empty)

    Returns status="healthy" if all checks pass,
            status="degraded" if memory is unavailable but agents work,
            status="unhealthy" if critical systems are down.
    """
    checks = {
        "config": False,
        "memory": False,
        "agents": False,
    }
    memory_count = 0

    # Check 1: Config
    try:
        _ = settings.gemini_api_key
        checks["config"] = True
    except Exception:
        pass

    # Check 2: ChromaDB memory store
    try:
        memory = get_memory_manager()
        memory_count = memory.count()
        checks["memory"] = True
    except Exception as e:
        logger.warning("health_memory_unavailable", error=str(e))

    # Check 3: Agent registry
    try:
        agents = list(AGENT_REGISTRY.keys())
        checks["agents"] = len(agents) > 0
    except Exception:
        agents = []

    # Determine overall status
    if checks["config"] and checks["agents"]:
        if checks["memory"]:
            overall = "healthy"
        else:
            overall = "degraded"  # Works, but no memory
    else:
        overall = "unhealthy"

    logger.info(
        "health_check",
        status=overall,
        checks=checks,
        memory_count=memory_count,
    )

    return HealthResponse(
        status=overall,
        app_name=settings.app_name,
        environment=settings.app_env,
        memory_count=memory_count,
        agents_available=sorted(agents),
        version="1.0.0",
    )
