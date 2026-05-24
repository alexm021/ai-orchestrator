# =============================================================================
# api/limiter.py -- Rate Limiter (shared instance)
# =============================================================================
#
# WHY RATE LIMITING:
# Without it, a single client can spam POST /task indefinitely,
# burning through Gemini quota in minutes. Rate limiting:
# - Protects the Gemini API quota
# - Prevents abuse / denial-of-service
# - Keeps the service fair for all users
#
# LIMITS (per IP address):
#   POST /task              -> 10 requests/minute  (expensive: calls Gemini)
#   POST /memories/search   -> 20 requests/minute  (moderate: calls Gemini embeddings)
#   GET  /memories          -> 30 requests/minute  (cheap: DB read only)
#   GET  /health            -> unlimited            (public, no AI calls)
#
# HOW SLOWAPI WORKS:
# slowapi is a port of Flask-Limiter for FastAPI/Starlette.
# It uses an in-memory store by default (resets on restart).
# For production at scale, swap the storage_uri to Redis.
#
# IDENTIFYING CLIENTS:
# We identify by real IP. The get_remote_address function reads
# X-Forwarded-For first (set by Railway's edge proxy), then falls
# back to the direct connection IP.
# =============================================================================

from slowapi import Limiter
from slowapi.util import get_remote_address

# Single shared Limiter instance -- imported by main.py and all routes
# storage_uri="memory://" = in-process store (good for single-instance deploy)
# For multi-worker / Redis: storage_uri="redis://localhost:6379"
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],          # No global default -- set per-route
    storage_uri="memory://",
)
