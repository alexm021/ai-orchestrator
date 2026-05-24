# =============================================================================
# core/config.py — Centralized Configuration
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Every part of the system needs settings (API keys, model names, timeouts).
# Instead of reading .env files everywhere, we create ONE settings object.
# All other files import from here. If a setting changes, we change it once.
#
# HOW IT WORKS:
# Pydantic Settings reads environment variables automatically.
# It validates types (str, int, bool) and raises errors if something is missing.
# This means we catch config errors at startup, not during execution.
#
# PATTERN: Singleton settings object loaded once, imported everywhere.
# =============================================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Central configuration object for the entire orchestrator system.

    Pydantic reads these values from environment variables (.env file).
    Field(...) means required — the app will refuse to start if missing.
    Field(default=...) means optional — uses the default if not set.
    """

    # -------------------------------------------------------------------------
    # LLM Provider API Keys
    # -------------------------------------------------------------------------
    # These are required — the system cannot function without them
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key (optional fallback)")
    api_key: str = Field(default="", description="API key for endpoint auth (empty = auth disabled)")

    # -------------------------------------------------------------------------
    # Model Selection
    # -------------------------------------------------------------------------
    # WHY SEPARATE MODELS:
    # - Router model = fast + cheap (gemini-2.0-flash)
    #   The router runs on every single request. Speed matters.
    # - Agent model = smart + thorough (gemini-1.5-pro)
    #   Agents do complex reasoning. Quality matters.
    # - Fast model = ultra-fast for simple tasks
    router_model: str = Field(default="gemini-2.5-flash")
    agent_model: str = Field(default="gemini-2.5-flash")
    fast_model: str = Field(default="gemini-2.5-flash")
    groq_model: str = Field(default="llama-3.1-8b-instant")

    # -------------------------------------------------------------------------
    # App Identity
    # -------------------------------------------------------------------------
    app_name: str = Field(default="ai-orchestrator")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # -------------------------------------------------------------------------
    # API Server
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # -------------------------------------------------------------------------
    # Memory / ChromaDB
    # -------------------------------------------------------------------------
    chroma_persist_dir: str = Field(default="./data/chromadb")
    chroma_collection_name: str = Field(default="orchestrator_memory")

    # -------------------------------------------------------------------------
    # Agent Behavior Controls
    # -------------------------------------------------------------------------
    # WHY THESE EXIST:
    # In production, you need guardrails. Without them:
    # - An agent can loop forever (no retry limit)
    # - One slow API call blocks everything (no timeout)
    # - Too many parallel agents crash memory (no concurrency limit)
    max_agent_retries: int = Field(default=3)
    agent_timeout_seconds: int = Field(default=30)
    max_concurrent_agents: int = Field(default=5)

    class Config:
        """
        Pydantic Settings configuration.
        env_file tells it to read from .env automatically.
        case_sensitive=False means GEMINI_API_KEY matches gemini_api_key.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# =============================================================================
# SINGLETON PATTERN
# =============================================================================
# @lru_cache means this function is only executed ONCE.
# Every import of get_settings() returns the SAME object in memory.
# WHY: We don't want to re-read the .env file hundreds of times per request.
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Returns the singleton settings instance.
    Called once at startup, cached forever.

    Usage in any file:
        from core.config import get_settings
        settings = get_settings()
        print(settings.gemini_api_key)
    """
    return Settings()


# Convenience alias — allows `from core.config import settings` anywhere
settings = get_settings()
