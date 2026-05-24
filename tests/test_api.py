"""
Phase 6 API Tests -- FastAPI endpoint tests
Run: python tests/test_api.py

Tests:
  1. GET /api/v1/health -- system status
  2. POST /api/v1/task -- single agent task
  3. POST /api/v1/task -- sequential pipeline
  4. GET /api/v1/memories -- list memories
  5. POST /api/v1/memories/search -- semantic search

Uses httpx AsyncClient to call the FastAPI app directly (no server needed).
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
# TEST 1 -- HEALTH CHECK
# =============================================================================

async def test_health(client) -> bool:
    """GET /api/v1/health returns healthy status."""
    response = await client.get("/api/v1/health")

    passed = (
        response.status_code == 200
        and response.json().get("status") in ("healthy", "degraded")
        and "agents_available" in response.json()
    )

    data = response.json()
    print_result(
        "GET /health",
        passed,
        f"status={data.get('status')} | agents={data.get('agents_available')} | memories={data.get('memory_count')}",
    )
    return passed


# =============================================================================
# TEST 2 -- SINGLE AGENT TASK (CODING)
# =============================================================================

async def test_task_single_agent(client) -> bool:
    """POST /task with a simple coding request routes to one agent."""
    response = await client.post(
        "/api/v1/task",
        json={
            "message": "Write a Python function to reverse a string",
            "priority": "medium",
        },
        timeout=60.0,
    )

    data = response.json()
    passed = (
        response.status_code == 200
        and data.get("status") == "complete"
        and "coding" in data.get("agents_used", [])
        and data.get("primary_output")
    )

    print_result(
        "POST /task -- single agent (coding)",
        passed,
        f"agents={data.get('agents_used')} | tokens={data.get('performance', {}).get('total_tokens')}",
    )

    if passed:
        output = data.get("primary_output", "")
        print(f"\n    Code preview (first 3 lines):")
        for line in output.split("\n")[:3]:
            if line.strip():
                print(f"      {line}")

    return passed


# =============================================================================
# TEST 3 -- REQUEST VALIDATION
# =============================================================================

async def test_request_validation(client) -> bool:
    """Empty message returns 422 Unprocessable Entity."""
    response = await client.post(
        "/api/v1/task",
        json={"message": "", "priority": "medium"},
    )

    passed = response.status_code == 422

    print_result(
        "POST /task -- validation (empty message -> 422)",
        passed,
        f"HTTP {response.status_code}",
    )
    return passed


# =============================================================================
# TEST 4 -- LIST MEMORIES
# =============================================================================

async def test_memories_list(client) -> bool:
    """GET /memories returns list of stored memories."""
    response = await client.get("/api/v1/memories?limit=5")

    data = response.json()
    passed = (
        response.status_code == 200
        and "total" in data
        and "memories" in data
        and isinstance(data["memories"], list)
    )

    print_result(
        "GET /memories",
        passed,
        f"total={data.get('total')} | returned={len(data.get('memories', []))}",
    )

    if passed and data.get("memories"):
        top = data["memories"][0]
        print(f"\n    Most recent memory:")
        print(f"      Task:    {top.get('task_message', '')[:70]}...")
        print(f"      Agents:  {top.get('agents_used')}")
        print(f"      Time:    {top.get('timestamp', '')[:19]}")

    return passed


# =============================================================================
# TEST 5 -- MEMORY SEARCH
# =============================================================================

async def test_memory_search(client) -> bool:
    """POST /memories/search returns semantically relevant results."""
    response = await client.post(
        "/api/v1/memories/search",
        json={"query": "Python code function", "n_results": 3},
        timeout=30.0,
    )

    data = response.json()
    passed = (
        response.status_code == 200
        and "memories" in data
        and isinstance(data["memories"], list)
    )

    print_result(
        "POST /memories/search",
        passed,
        f"found={len(data.get('memories', []))} results for 'Python code function'",
    )

    if passed and data.get("memories"):
        for m in data["memories"][:2]:
            score = m.get("relevance_score")
            score_str = f"{score:.0%}" if score is not None else "N/A"
            print(f"    [{score_str}] {m.get('task_message', '')[:70]}")

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

    print_header("PHASE 6 -- API TESTS")
    print("\n  Testing FastAPI endpoints (no server needed -- direct ASGI)")
    print("  Tests: health, task, validation, memories, search\n")

    results = []

    # Use ASGITransport to test the FastAPI app directly
    # No server process needed -- httpx calls the app in-process
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        print("\n[ TEST 1 -- HEALTH CHECK ]")
        results.append(await test_health(client))

        print("\n[ TEST 2 -- SINGLE AGENT TASK ]")
        results.append(await test_task_single_agent(client))

        print("\n[ TEST 3 -- INPUT VALIDATION ]")
        results.append(await test_request_validation(client))

        print("\n[ TEST 4 -- LIST MEMORIES ]")
        results.append(await test_memories_list(client))

        print("\n[ TEST 5 -- MEMORY SEARCH ]")
        results.append(await test_memory_search(client))

    passed_count = sum(results)
    total = len(results)

    print(f"\n{'='*60}")
    if all(results):
        print(f"  \033[92m ALL API TESTS PASSED ({passed_count}/{total})\033[0m")
        print(f"\n  What Phase 6 gives us:")
        print(f"  - REST API over the full multi-agent orchestrator")
        print(f"  - POST /task: submit any task, get structured JSON back")
        print(f"  - GET /health: production-ready health check")
        print(f"  - GET /memories: audit what the system has learned")
        print(f"  - POST /memories/search: semantic search over past tasks")
        print(f"  - Swagger UI at /docs when server is running")
        print(f"\n  To start the server: python run_api.py")
        print(f"  Then visit: http://localhost:8000/docs")
    else:
        failed = total - passed_count
        print(f"  \033[91m {failed} TEST(S) FAILED ({passed_count}/{total} passed)\033[0m")
    print(f"{'='*60}\n")

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
