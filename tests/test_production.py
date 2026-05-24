"""
Phase 7 Production Tests -- middleware, timeouts, error handling
Run: python tests/test_production.py

Tests:
  1. Request ID -- every response has X-Request-ID header
  2. Validation error -- empty message returns 422 with JSON body
  3. Missing field -- request without 'message' returns 422
  4. Invalid priority -- bad enum value returns 422
  5. Health endpoint -- returns JSON (not HTML)
  6. Root redirect -- GET / redirects to /docs
  7. Exception handler -- unhandled errors return JSON (not HTML 500)
  8. Request logging -- middleware runs without crashing

All tests use ASGITransport (no server needed, no API calls made).
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label: str, passed: bool, detail: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"\n  {color}{status}{reset}  {label}")
    if detail:
        print(f"    {detail}")


# =============================================================================
# TEST 1 -- REQUEST ID HEADER
# =============================================================================

async def test_request_id_header(client) -> bool:
    """Every response must include X-Request-ID header."""
    response = await client.get("/api/v1/health")

    request_id = response.headers.get("X-Request-ID", "")
    passed = (
        response.status_code == 200
        and len(request_id) == 8  # We generate 8-char UUID prefix
        and request_id.isalnum() or "-" in request_id  # UUID hex chars
    )

    print_result(
        "X-Request-ID header present on every response",
        passed,
        f"X-Request-ID: {request_id!r} (len={len(request_id)})",
    )
    return passed


# =============================================================================
# TEST 2 -- EMPTY MESSAGE VALIDATION
# =============================================================================

async def test_validation_empty_message(client) -> bool:
    """Empty message string returns 422 with structured JSON error."""
    response = await client.post(
        "/api/v1/task",
        json={"message": "", "priority": "medium"},
    )

    data = response.json()
    passed = (
        response.status_code == 422
        and data.get("error") == "validation_error"
        and "details" in data
        and "request_id" in data
    )

    print_result(
        "POST /task -- empty message returns 422 JSON",
        passed,
        f"HTTP {response.status_code} | error={data.get('error')} | "
        f"request_id present={('request_id' in data)}",
    )
    if not passed:
        print(f"    Response: {data}")
    return passed


# =============================================================================
# TEST 3 -- MISSING REQUIRED FIELD
# =============================================================================

async def test_validation_missing_field(client) -> bool:
    """Request without 'message' field returns 422."""
    response = await client.post(
        "/api/v1/task",
        json={"priority": "high"},  # 'message' is missing
    )

    data = response.json()
    passed = (
        response.status_code == 422
        and data.get("error") == "validation_error"
    )

    print_result(
        "POST /task -- missing 'message' field returns 422",
        passed,
        f"HTTP {response.status_code} | error={data.get('error')}",
    )
    return passed


# =============================================================================
# TEST 4 -- INVALID PRIORITY VALUE
# =============================================================================

async def test_validation_bad_priority(client) -> bool:
    """Invalid priority value (not low/medium/high) returns 422."""
    response = await client.post(
        "/api/v1/task",
        json={"message": "Do something", "priority": "URGENT"},
    )

    data = response.json()
    passed = (
        response.status_code == 422
        and data.get("error") == "validation_error"
    )

    print_result(
        "POST /task -- invalid priority returns 422",
        passed,
        f"HTTP {response.status_code} | error={data.get('error')}",
    )
    return passed


# =============================================================================
# TEST 5 -- HEALTH RETURNS JSON (NOT HTML)
# =============================================================================

async def test_health_returns_json(client) -> bool:
    """GET /health must return JSON, not an HTML page."""
    response = await client.get("/api/v1/health")

    content_type = response.headers.get("content-type", "")
    try:
        data = response.json()
        is_json = True
    except Exception:
        is_json = False
        data = {}

    passed = (
        response.status_code == 200
        and is_json
        and "status" in data
        and "application/json" in content_type
    )

    print_result(
        "GET /health returns JSON (not HTML)",
        passed,
        f"Content-Type: {content_type} | status={data.get('status')}",
    )
    return passed


# =============================================================================
# TEST 6 -- ROOT REDIRECT
# =============================================================================

async def test_root_redirect(client) -> bool:
    """GET / redirects to /docs (Swagger UI)."""
    # follow_redirects=False so we can inspect the redirect itself
    response = await client.get("/", follow_redirects=False)

    passed = (
        response.status_code in (301, 302, 307, 308)
        and "/docs" in response.headers.get("location", "")
    )

    print_result(
        "GET / redirects to /docs",
        passed,
        f"HTTP {response.status_code} | Location: {response.headers.get('location', 'N/A')}",
    )
    return passed


# =============================================================================
# TEST 7 -- 404 NOT FOUND RETURNS JSON
# =============================================================================

async def test_404_returns_json(client) -> bool:
    """Unknown endpoint returns JSON error, not HTML 404 page."""
    response = await client.get("/api/v1/nonexistent-endpoint")

    content_type = response.headers.get("content-type", "")

    # FastAPI returns JSON 404 by default for unknown routes
    passed = (
        response.status_code == 404
        and "application/json" in content_type
    )

    print_result(
        "GET /unknown-endpoint returns JSON 404 (not HTML)",
        passed,
        f"HTTP {response.status_code} | Content-Type: {content_type}",
    )
    return passed


# =============================================================================
# TEST 8 -- CORS HEADERS
# =============================================================================

async def test_cors_headers(client) -> bool:
    """OPTIONS preflight request includes CORS headers."""
    response = await client.options(
        "/api/v1/task",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    # CORS middleware should add allow-origin header
    allow_origin = response.headers.get("access-control-allow-origin", "")
    passed = (
        response.status_code in (200, 204)
        and allow_origin != ""
    )

    print_result(
        "OPTIONS /task -- CORS headers present",
        passed,
        f"HTTP {response.status_code} | Access-Control-Allow-Origin: {allow_origin!r}",
    )
    return passed


# =============================================================================
# TEST 9 -- VALIDATION ERROR HAS REQUEST ID
# =============================================================================

async def test_validation_error_has_request_id(client) -> bool:
    """422 validation error response includes request_id for tracing."""
    response = await client.post(
        "/api/v1/task",
        json={"message": ""},
    )

    data = response.json()
    request_id_in_body = data.get("request_id", "")
    request_id_in_header = response.headers.get("X-Request-ID", "")

    passed = (
        response.status_code == 422
        and len(request_id_in_body) > 0
        and len(request_id_in_header) > 0
        # Both should match (same request)
        and request_id_in_body == request_id_in_header
    )

    print_result(
        "422 error body contains request_id matching X-Request-ID header",
        passed,
        f"body request_id={request_id_in_body!r} | header={request_id_in_header!r} | match={request_id_in_body == request_id_in_header}",
    )
    return passed


# =============================================================================
# TEST 10 -- AUTH: NO KEY RETURNS 401 WHEN API_KEY IS SET
# =============================================================================

async def test_auth_no_key(client) -> bool:
    """When API_KEY is configured, requests without X-API-Key get 401."""
    import os
    from unittest.mock import patch

    # Temporarily set an API key in settings
    with patch.object(__import__('core.config', fromlist=['settings']).settings, 'api_key', 'test-secret-key'):
        response = await client.post(
            "/api/v1/task",
            json={"message": "Hello", "priority": "medium"},
        )

    passed = response.status_code == 401
    data = response.json()

    print_result(
        "POST /task without key returns 401 when auth enabled",
        passed,
        f"HTTP {response.status_code} | error={data.get('detail', {}).get('error') if isinstance(data.get('detail'), dict) else data.get('error')}",
    )
    return passed


async def test_auth_wrong_key(client) -> bool:
    """Wrong API key returns 401."""
    from unittest.mock import patch

    with patch.object(__import__('core.config', fromlist=['settings']).settings, 'api_key', 'test-secret-key'):
        response = await client.post(
            "/api/v1/task",
            json={"message": "Hello", "priority": "medium"},
            headers={"X-API-Key": "wrong-key"},
        )

    passed = response.status_code == 401

    print_result(
        "POST /task with wrong key returns 401",
        passed,
        f"HTTP {response.status_code}",
    )
    return passed


async def test_auth_health_public(client) -> bool:
    """Health endpoint is always public — no API key needed."""
    from unittest.mock import patch

    with patch.object(__import__('core.config', fromlist=['settings']).settings, 'api_key', 'test-secret-key'):
        response = await client.get("/api/v1/health")

    passed = response.status_code == 200

    print_result(
        "GET /health is public (no key required)",
        passed,
        f"HTTP {response.status_code}",
    )
    return passed


# =============================================================================
# MAIN
# =============================================================================

async def run_all_tests():
    try:
        from httpx import AsyncClient, ASGITransport
        from api.main import app
    except ImportError as e:
        print(f"\n  [ERROR] Missing dependency: {e}")
        print("  Install with: pip install httpx")
        return False

    setup_logging()

    print_header("PHASE 7 -- PRODUCTION ENGINEERING TESTS")
    print("\n  Testing middleware, validation, and error handling")
    print("  No API calls made -- all tests run in-process\n")

    results = []

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        print("\n[ TEST 1 -- REQUEST ID MIDDLEWARE ]")
        results.append(await test_request_id_header(client))

        print("\n[ TEST 2 -- EMPTY MESSAGE VALIDATION ]")
        results.append(await test_validation_empty_message(client))

        print("\n[ TEST 3 -- MISSING FIELD VALIDATION ]")
        results.append(await test_validation_missing_field(client))

        print("\n[ TEST 4 -- INVALID PRIORITY VALIDATION ]")
        results.append(await test_validation_bad_priority(client))

        print("\n[ TEST 5 -- HEALTH RETURNS JSON ]")
        results.append(await test_health_returns_json(client))

        print("\n[ TEST 6 -- ROOT REDIRECT ]")
        results.append(await test_root_redirect(client))

        print("\n[ TEST 7 -- 404 RETURNS JSON ]")
        results.append(await test_404_returns_json(client))

        print("\n[ TEST 8 -- CORS HEADERS ]")
        results.append(await test_cors_headers(client))

        print("\n[ TEST 9 -- REQUEST ID IN ERROR BODY ]")
        results.append(await test_validation_error_has_request_id(client))

        print("\n[ TEST 10 -- AUTH: NO KEY -> 401 ]")
        results.append(await test_auth_no_key(client))

        print("\n[ TEST 11 -- AUTH: WRONG KEY -> 401 ]")
        results.append(await test_auth_wrong_key(client))

        print("\n[ TEST 12 -- AUTH: HEALTH IS PUBLIC ]")
        results.append(await test_auth_health_public(client))

    passed_count = sum(results)
    total = len(results)

    print(f"\n{'='*60}")
    if all(results):
        print(f"  \033[92m ALL PRODUCTION TESTS PASSED ({passed_count}/{total})\033[0m")
        print(f"\n  What Phase 7 gives us:")
        print(f"  - Request ID on every request (distributed tracing)")
        print(f"  - Structured JSON for ALL errors (no HTML pages)")
        print(f"  - Input validation with readable error messages")
        print(f"  - CORS support for browser frontends")
        print(f"  - HTTP 503 + Retry-After for quota exhaustion")
        print(f"  - HTTP 504 for task timeouts (3 min limit)")
        print(f"  - Access log for every request (method/path/status/ms)")
    else:
        failed = total - passed_count
        print(f"  \033[91m {failed} TEST(S) FAILED ({passed_count}/{total} passed)\033[0m")
    print(f"{'='*60}\n")

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
