# =============================================================================
# core/logger.py — Structured Logging Setup
# =============================================================================
#
# WHY STRUCTURED LOGGING:
# Standard print() or logging.info("agent started") is useless in production.
# You can't search it, filter it, or feed it to monitoring tools.
#
# Structured logging means every log is a JSON object with consistent fields:
# {
#   "event": "agent_started",
#   "agent": "research",
#   "task_id": "abc-123",
#   "timestamp": "2025-01-01T10:00:00Z",
#   "level": "info"
# }
#
# This lets you later:
# - Filter logs by agent name
# - Measure average execution time per agent
# - Alert when error rate exceeds threshold
# - Build dashboards from log data
#
# We use structlog — the industry standard for Python structured logging.
# =============================================================================

import logging
import structlog
from core.config import settings


def setup_logging() -> None:
    """
    Configure structlog for the entire application.
    Called ONCE at startup in main.py.

    After this runs, every module can do:
        logger = structlog.get_logger()
        logger.info("event_name", key=value, key2=value2)
    """

    # Convert our string log level ("INFO", "DEBUG") to the logging constant
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure the standard Python logging underneath structlog
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    # Choose output format based on environment
    # Development: pretty colored output for humans
    # Production: JSON output for log aggregation tools (Datadog, Splunk, etc.)
    if settings.app_env == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            # Add log level to every event (info, warning, error)
            structlog.stdlib.add_log_level,
            # Add timestamp to every event
            structlog.processors.TimeStamper(fmt="iso"),
            # Add the app name so you know which service produced this log
            structlog.processors.CallsiteParameterAdder(
                [structlog.processors.CallsiteParameter.FILENAME,
                 structlog.processors.CallsiteParameter.LINENO]
            ),
            # Final renderer — human-readable in dev, JSON in prod
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "orchestrator") -> structlog.stdlib.BoundLogger:
    """
    Returns a logger bound with the component name.

    Usage:
        from core.logger import get_logger
        logger = get_logger("research_agent")
        logger.info("task_started", task_id="abc", model="gemini-1.5-pro")
        logger.error("api_failed", error=str(e), retry=1)

    The 'name' appears in every log line so you know exactly where it came from.
    """
    return structlog.get_logger(name)
