# =============================================================================
# tests/test_router.py — Router Agent Tests
# =============================================================================
#
# WHAT WE'RE TESTING:
# The Router Agent must correctly classify 5 fundamentally different tasks.
# Each task should route to a different primary agent.
# This proves the routing logic is working, not just returning the same answer.
#
# WHY TEST THIS WAY:
# We're NOT mocking the Gemini API here. We're doing a LIVE integration test.
# Reason: Router quality depends entirely on the LLM's understanding.
# Mocking would test nothing real.
#
# In a production system you'd have:
# - Unit tests with mocked API (fast, no cost)
# - Integration tests against real API (slower, costs tokens) — like these
#
# HOW TO RUN:
#   cd D:\Orchestrator
#   .\venv\Scripts\Activate.ps1
#   python tests/test_router.py
# =============================================================================

import asyncio
import sys
import os

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging
from schemas.task import TaskInput
from schemas.routing import AgentType, ExecutionStrategy
from agents.router_agent import RouterAgent


# =============================================================================
# TEST CASES
# Each entry: (task_message, expected_primary_agent, test_description)
# =============================================================================

TEST_CASES = [
    (
        "Research the top 5 AI companies in 2025 and their main products",
        AgentType.RESEARCH,
        "Pure research task — should go to research agent"
    ),
    (
        "Write a cold email to the VP of Sales at Stripe introducing our payment analytics tool",
        AgentType.OUTREACH,
        "Cold email writing — should go to outreach agent"
    ),
    (
        "Fix this Python bug: my for loop is not iterating over the list correctly. "
        "Code: for i in range(len(mylist)): print(mylist[i+1])",
        AgentType.CODING,
        "Code debugging task — should go to coding agent"
    ),
    (
        "Summarize this article in 5 bullet points: Artificial intelligence is transforming "
        "industries at an unprecedented pace. Companies across sectors from healthcare to "
        "finance are integrating AI into their core operations. The technology promises "
        "efficiency gains but also raises concerns about job displacement and data privacy.",
        AgentType.SUMMARIZATION,
        "Text summarization — should go to summarization agent"
    ),
    (
        "Create a 3-month content marketing plan for our SaaS product launch",
        [AgentType.PLANNING, AgentType.CONTENT],  # Either is valid
        "Content strategy — should go to planning or content agent"
    ),
]


async def run_single_test(
    router: RouterAgent,
    task_message: str,
    expected_agent,
    description: str,
    test_number: int
) -> bool:
    """
    Run a single routing test and return True if it passed.
    """
    print(f"\n  Test {test_number}: {description}")
    print(f"  Task: \"{task_message[:70]}{'...' if len(task_message) > 70 else ''}\"")

    task = TaskInput(user_message=task_message)

    try:
        decision = await router.route(task)

        # Check if expected agent matches
        if isinstance(expected_agent, list):
            passed = decision.primary_agent in expected_agent
            expected_str = " or ".join(a.value for a in expected_agent)
        else:
            passed = decision.primary_agent == expected_agent
            expected_str = expected_agent.value

        # Print results
        status = "[PASS]" if passed else "[FAIL]"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"

        print(f"  {color}{status}{reset}")
        print(f"    task_type:    {decision.task_type}")
        print(f"    primary:      {decision.primary_agent.value}  (expected: {expected_str})")
        print(f"    agents:       {' -> '.join(decision.agent_names())}")
        print(f"    strategy:     {decision.execution_strategy.value}")
        print(f"    confidence:   {decision.confidence:.0%}")
        print(f"    reasoning:    {decision.reasoning[:100]}...")

        if not passed:
            print(f"    [!] Expected {expected_str}, got {decision.primary_agent.value}")

        return passed

    except Exception as e:
        print(f"  \033[91m[ERROR]\033[0m Exception during routing: {e}")
        return False


async def run_all_tests():
    """
    Run all routing test cases and print a summary.
    """
    setup_logging()

    print("\n" + "=" * 60)
    print("   ROUTER AGENT — INTEGRATION TESTS")
    print("=" * 60)
    print(f"\n  Running {len(TEST_CASES)} test cases against Gemini API...")
    print("  (Each test makes one API call — takes a few seconds)\n")

    router = RouterAgent()
    results = []

    for i, (message, expected, description) in enumerate(TEST_CASES, 1):
        passed = await run_single_test(router, message, expected, description, i)
        results.append(passed)
        # Small delay between API calls to avoid rate limiting
        if i < len(TEST_CASES):
            await asyncio.sleep(2)

    # Summary
    passed_count = sum(results)
    total = len(results)
    all_passed = all(results)

    print("\n" + "=" * 60)
    if all_passed:
        print(f"  \033[92m ALL TESTS PASSED ({passed_count}/{total})\033[0m")
        print("  Router agent is working correctly.")
        print("  Ready for Phase 3 — Specialized Agents.")
    else:
        failed = total - passed_count
        print(f"  \033[91m {failed} TEST(S) FAILED ({passed_count}/{total} passed)\033[0m")
        print("  Check the routing logic or system prompt.")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
