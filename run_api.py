"""
AI Multi-Agent Orchestrator — API Server Entry Point

Usage:
    python run_api.py             # Development mode (auto-reload)
    python run_api.py --prod      # Production mode (multi-worker, no reload)

The API will be available at:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/api/v1/health
    http://localhost:8000/api/v1/task
"""
import os
import sys
import uvicorn
from core.config import settings


def main():
    # Railway (and most cloud platforms) inject PORT as an env var.
    # Fall back to settings value for local dev.
    port = int(os.environ.get("PORT", settings.api_port))
    host = settings.api_host
    prod_mode = "--prod" in sys.argv or settings.app_env == "production"

    # Ensure the data directory exists (needed for ChromaDB on first run)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    if prod_mode:
        print(f"Starting {settings.app_name} in PRODUCTION mode")
        print(f"Listening on {host}:{port}")
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=2,
            log_level="warning",
            access_log=False,
        )
    else:
        print(f"Starting {settings.app_name} in DEVELOPMENT mode")
        print(f"API docs: http://localhost:{port}/docs")
        print(f"Press Ctrl+C to stop\n")
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )


if __name__ == "__main__":
    main()
