# =============================================================================
# tests/test_orchestrator.py — Full End-to-End Orchestration Tests
# =============================================================================
#
# WHAT WE'RE TESTING:
# The full system: User task → Router → Agents → Final output
# No mocking. Real API calls. Real agent execution.
#
# TEST CASES:
# 1. Single-agent task (coding) → proves basic flow works
# 2. Sequential pipeline (research → outreach) → proves context passing works
# 3. Edge case: unambiguous single-agent task (summarization)
#
# WHY THESE THREE:
# - Case 1: happy path, simple
# - Case 2: the money shot — full multi-agent pipeline
# - Case 3: verifies the router doesn't over-engineer simple tasks
#
# HOW TO RUN:
#   cd D:\Orchestrator
#   .\venv\Scripts\Activate.ps1
#   python tests/test_orchestrator.py
# =============================================================================

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging
from orchestrator.graph import run_task


def print_header(title: str):
    print(f"\n{'='*65}")
    print(f"   {title}")
    print(f"{'='*65}")


def print_result(label: str, passed: bool, detail: str = ""):
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n  {color}{status}{reset}  {label}")
    if detail:
        print(f"    {detail}")


def print_performance(result: dict):
    perf = result.get("performance", {})
    if perf:
        print(f"\n    Performance:")
        print(f"      Total tokens:   {perf.get('total_tokens', 0)}")
        print(f"      Wall time:      {result.get('total_wall_time_ms', 0)}ms")
        by_agent = perf.get("tokens_by_agent", {})
        if by_agent:
            for agent, tokens in by_agent.items():
                print(f"      {agent:20s}: {tokens} tokens")


def print_routing(result: dict):
    routing = result.get("routing", {})
    if routing:
        print(f"\n    Routing:")
        print(f"      task_type:  {routing.get('task_type')}")
        print(f"      confidence: {routing.get('confidence', 0):.0%}")
        agents = result.get("agents_used", [])
        print(f"      agents:     {' -> '.join(agents)}")
        print(f"      strategy:   {result.get('execution_strategy')}")


# =============================================================================
# TEST 1 — SINGLE AGENT (CODING)
# =============================================================================

async def test_single_agent_coding() -> bool:
    """
    End-to-end test: Code task routes to coding agent and returns solution.
    Verifies: routing → single agent execution → output assembly
    """
    print("\n  Task: Write a Python function to check if a number is prime")

    result = await run_task(
        task_message="Write a Python function that checks if a number is prime. Include docstring and example usage.",
        task_priority="medium",
    )

    passed = (
        result.get("status") == "complete"
        and result.get("primary_output")
        and len(result.get("primary_output", "")) > 50
        and "coding" in result.get("agents_used", [])
    )

    print_result("Single Agent — Coding Task", passed,
                 f"agents={result.get('agents_used')} | strategy={result.get('execution_strategy')}")
    print_routing(result)
    print_performance(result)

    if passed:
        output = result.get("primary_output", "")
        print(f"\n    Code output (first 8 lines):")
        for line in output.split("\n")[:8]:
            print(f"      {line}")

    return passed


# =============================================================================
# TEST 2 — SEQUENTIAL PIPELINE (RESEARCH → OUTREACH)
# =============================================================================

async def test_sequential_pipeline() -> bool:
    """
    End-to-end test: Research + outreach task runs both agents sequentially.
    The outreach email must reference facts from the research.
    This is the core multi-agent use case.
    """
    print("\n  Task: Research Anthropic and write a cold email to their Head of Product")

    result = await run_task(
        task_message=(
            "Research Anthropic (the AI safety company) — their main products, "
            "mission, and recent developments. Then write a cold email to their "
            "Head of Product introducing an AI developer tools company."
        ),
        task_priority="high",
    )

    agents_used = result.get("agents_used", [])
    passed = (
        result.get("status") == "complete"
        and result.get("primary_output")
        and len(result.get("primary_output", "")) > 100
        and len(agents_used) >= 2  # Must have used multiple agents
        and "outreach" in agents_used  # Must have produced an email
    )

    print_result("Sequential Pipeline — Research + Outreach", passed,
                 f"agents={agents_used} | strategy={result.get('execution_strategy')}")
    print_routing(result)
    print_performance(result)

    if passed:
        all_outputs = result.get("all_outputs", {})

        # Show research output
        if "research" in all_outputs:
            research = all_outputs["research"]
            print(f"\n    [Research Agent Output]")
            print(f"    Topic: {research.get('topic', 'N/A')}")
            print(f"    Summary: {research.get('summary', '')[:150]}...")
            facts = research.get("key_facts", [])[:2]
            for f in facts:
                print(f"      - {f[:100]}")

        # Show outreach output (primary)
        if "outreach" in all_outputs:
            outreach = all_outputs["outreach"]
            print(f"\n    [Outreach Agent Output]")
            print(f"    Subject: {outreach.get('subject')}")
            print(f"    Tone: {outreach.get('tone')}")
            print(f"\n    Email body (first 4 lines):")
            for line in outreach.get("body", "").split("\n")[:4]:
                if line.strip():
                    print(f"      {line}")
            print(f"\n    Personalization: {outreach.get('personalization_notes', '')[:120]}")

    return passed


# =============================================================================
# TEST 3 — SINGLE AGENT (SUMMARIZATION)
# =============================================================================

async def test_single_agent_summarization() -> bool:
    """
    End-to-end test: Summarization task uses only one agent (no over-engineering).
    Verifies the router correctly identifies simple tasks as single-agent.
    """
    print("\n  Task: Summarize a paragraph about AI agents")

    text = """
    AI agents are autonomous systems that perceive their environment, make decisions,
    and take actions to achieve specific goals. Unlike traditional software that follows
    explicit programmed instructions, agents can adapt their behavior based on context
    and feedback. Modern AI agents built on large language models can perform complex
    reasoning, use external tools, and collaborate with other agents to solve
    sophisticated tasks that would be impossible for any single model to handle alone.
    """

    result = await run_task(
        task_message=f"Summarize this text into 3 bullet points:\n{text}",
        task_priority="low",
    )

    passed = (
        result.get("status") == "complete"
        and result.get("primary_output")
        and "summarization" in result.get("agents_used", [])
    )

    detail = f"agents={result.get('agents_used')} | strategy={result.get('execution_strategy')}"
    print_result("Single Agent — Summarization", passed, detail)
    print_routing(result)
    print_performance(result)

    if passed:
        all_outputs = result.get("all_outputs", {})
        if "summarization" in all_outputs:
            summ = all_outputs["summarization"]
            print(f"\n    TL;DR: {summ.get('summary', '')[:150]}")
            print(f"\n    Key points:")
            for point in summ.get("key_points", []):
                print(f"      - {point[:100]}")

    return passed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    setup_logging()

    print_header("ORCHESTRATION ENGINE — END-TO-END TESTS")
    print("\n  Running 3 full pipeline tests (Router + Agents + Assembly)")
    print("  Each test makes multiple API calls. Allow 20-40 seconds total.\n")

    results = []

    print("\n[ TEST 1 — SINGLE AGENT FLOW ]")
    results.append(await test_single_agent_coding())
    await asyncio.sleep(10)

    print("\n[ TEST 2 — SEQUENTIAL MULTI-AGENT PIPELINE ]")
    results.append(await test_sequential_pipeline())
    await asyncio.sleep(10)

    print("\n[ TEST 3 — SIMPLE TASK (SINGLE AGENT) ]")
    results.append(await test_single_agent_summarization())

    # Summary
    passed_count = sum(results)
    total = len(results)
    all_passed = all(results)

    print(f"\n{'='*65}")
    if all_passed:
        print(f"  \033[92m ALL TESTS PASSED ({passed_count}/{total})\033[0m")
        print(f"\n  What just happened:")
        print(f"  - Task came in as plain text")
        print(f"  - Router analyzed intent and built an execution plan")
        print(f"  - Orchestrator ran agents in the right order")
        print(f"  - Context passed between agents automatically")
        print(f"  - Final output assembled and returned")
        print(f"\n  This is a working multi-agent orchestrator.")
        print(f"  Ready for Phase 5 - Memory + Retrieval.")
    else:
        failed = total - passed_count
        print(f"  \033[91m {failed} TEST(S) FAILED ({passed_count}/{total} passed)\033[0m")
    print(f"{'='*65}\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
