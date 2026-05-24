# =============================================================================
# Dockerfile -- AI Multi-Agent Orchestrator
# =============================================================================
#
# Multi-stage build:
#   Stage 1 (builder): install dependencies into a venv
#   Stage 2 (runtime): copy only the venv + app code, no build tools
#
# WHY MULTI-STAGE:
# Build tools (gcc, pip, wheel) are needed to compile some packages
# (chromadb has C extensions) but not needed at runtime.
# Multi-stage keeps the final image small and secure.
#
# Build:   docker build -t ai-orchestrator .
# Run:     docker run -p 8000:8000 --env-file .env ai-orchestrator
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1 -- Builder: install all Python dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install system deps needed to compile chromadb / onnxruntime C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first -- Docker caches this layer
# If requirements.txt doesn't change, pip install is skipped on rebuild
COPY requirements.txt .

# Install into a virtual env inside the image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# STAGE 2 -- Runtime: lean final image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create a non-root user -- running as root inside containers is a security risk
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy the venv from builder (no gcc/build tools in final image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create data directory for ChromaDB persistence
RUN mkdir -p ./data/chromadb && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Document which port the app listens on
EXPOSE 8000

# Health check -- Docker will restart the container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Start the server in production mode
CMD ["python", "run_api.py", "--prod"]
