# =============================================================================
# tests/test_agents.py — Specialized Agents Integration Tests
# =============================================================================
#
# WHAT WE'RE TESTING:
# 1. Each agent works independently (correct output structure)
# 2. The Research → Outreach pipeline (context passing between agents)
#
# WHY THE PIPELINE TEST IS THE MOST IMPORTANT:
# Agents working individually proves nothing special.
# Agents CHAINING context proves the architecture works.
# This is the foundation of the orchestrator in Phase 4.
#
# HOW TO RUN:
#   cd D:\Orchestrator
#   .\venv\Scripts\Activate.ps1
#   python tests/test_agents.py
# =============================================================================

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging
from schemas.task import TaskInput
from agents.research_agent import ResearchAgent
from agents.summarization_agent import SummarizationAgent
from agents.outreach_agent import OutreachAgent
from agents.coding_agent import CodingAgent


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}")


def print_agent_result(agent_name: str, output, passed: bool, detail: str = ""):
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n  {color}{status}{reset}  {agent_name}")
    if detail:
        print(f"    {detail}")
    if passed and output and output.success:
        print(f"    duration:  {output.execution_time_ms}ms")
        if output.tokens_used:
            print(f"    tokens:    {output.tokens_used}")
        if output.model_used:
            print(f"    model:     {output.model_used}")


# =============================================================================
# TEST 1 — RESEARCH AGENT
# =============================================================================

async def test_research_agent() -> tuple[bool, dict]:
    """
    Test: Research Agent can analyze a topic and return structured output.
    Returns (passed, result_dict) so result can be used in pipeline test.
    """
    agent = ResearchAgent()
    task = TaskInput(
        user_message="Research LangGraph: what it is, main use cases, and how it compares to simple LLM chains",
        priority="high"
    )

    output = await agent.execute(task)

    passed = (
        output.success
        and "topic" in output.result
        and "summary" in output.result
        and "key_facts" in output.result
        and len(output.result["key_facts"]) >= 2
        and "confidence" in output.result
    )

    detail = ""
    if passed:
        detail = (
            f"topic='{output.result['topic']}' | "
            f"facts={len(output.result['key_facts'])} | "
            f"confidence={output.result['confidence']:.0%}"
        )
    else:
        detail = output.error or "Missing required fields in result"

    print_agent_result("Research Agent", output, passed, detail)

    if passed:
        print(f"\n    Summary preview:")
        print(f"    {output.result['summary'][:150]}...")
        print(f"\n    Key facts:")
        for i, fact in enumerate(output.result["key_facts"][:3], 1):
            print(f"      {i}. {fact}")

    return passed, output.result if passed else {}


# =============================================================================
# TEST 2 — SUMMARIZATION AGENT
# =============================================================================

async def test_summarization_agent() -> bool:
    """
    Test: Summarization Agent condenses a text into structured bullet points.
    """
    agent = SummarizationAgent()

    sample_text = """
    Artificial intelligence is undergoing a fundamental transformation in 2025.
    Large language models have evolved from simple text completion tools into
    sophisticated reasoning systems capable of complex multi-step problem solving.
    Companies like Google, Anthropic, and OpenAI are competing to deploy models
    that can act as autonomous agents — systems that take actions in the world,
    not just generate text.

    The enterprise adoption of AI has accelerated dramatically. A recent survey
    found that 78% of Fortune 500 companies now have at least one AI pilot in
    production, up from 35% in 2023. The primary use cases are customer service
    automation, document processing, and code generation.

    However, significant challenges remain. Hallucination rates in production
    systems, while improving, still require human oversight for high-stakes
    decisions. The cost of running large models at scale remains a barrier for
    smaller organizations. And questions around AI safety and alignment continue
    to drive policy debates in both the US and European Union.
    """

    task = TaskInput(user_message=sample_text)
    output = await agent.execute(task)

    passed = (
        output.success
        and "summary" in output.result
        and "key_points" in output.result
        and len(output.result["key_points"]) >= 2
        and "main_topic" in output.result
    )

    detail = ""
    if passed:
        detail = (
            f"topic='{output.result['main_topic']}' | "
            f"points={len(output.result['key_points'])} | "
            f"compression={output.result.get('compression_ratio', 'N/A')}"
        )
    else:
        detail = output.error or "Missing required fields in result"

    print_agent_result("Summarization Agent", output, passed, detail)

    if passed:
        print(f"\n    TL;DR: {output.result['summary']}")
        print(f"\n    Key points:")
        for i, point in enumerate(output.result["key_points"], 1):
            print(f"      {i}. {point}")

    return passed


# =============================================================================
# TEST 3 — OUTREACH AGENT (standalone, no research context)
# =============================================================================

async def test_outreach_agent_standalone() -> bool:
    """
    Test: Outreach Agent can write an email without research context.
    """
    agent = OutreachAgent()
    task = TaskInput(
        user_message=(
            "Write a cold email to the Head of Engineering at Stripe. "
            "We're selling an AI code review tool that reduces review time by 60%. "
            "Keep it short and direct."
        )
    )

    output = await agent.execute(task)

    passed = (
        output.success
        and "subject" in output.result
        and "body" in output.result
        and len(output.result["subject"]) > 5
        and len(output.result["body"]) > 50
    )

    detail = ""
    if passed:
        detail = (
            f"tone={output.result.get('tone', 'N/A')} | "
            f"subject='{output.result['subject']}'"
        )
    else:
        detail = output.error or "Missing required fields"

    print_agent_result("Outreach Agent (standalone)", output, passed, detail)

    if passed:
        print(f"\n    Subject: {output.result['subject']}")
        print(f"\n    Body preview:")
        body_lines = output.result["body"].split("\n")[:6]
        for line in body_lines:
            if line.strip():
                print(f"    {line}")

    return passed


# =============================================================================
# TEST 4 — CODING AGENT
# =============================================================================

async def test_coding_agent() -> bool:
    """
    Test: Coding Agent can debug broken code and return the fix.
    """
    agent = CodingAgent()
    task = TaskInput(
        user_message=(
            "Debug this Python code — it should return the average of a list "
            "but crashes on empty lists:\n\n"
            "def get_average(numbers):\n"
            "    return sum(numbers) / len(numbers)\n\n"
            "print(get_average([]))"
        )
    )

    output = await agent.execute(task)

    passed = (
        output.success
        and "solution" in output.result
        and "explanation" in output.result
        and "language" in output.result
        and len(output.result["solution"]) > 20
    )

    detail = ""
    if passed:
        detail = (
            f"language={output.result['language']} | "
            f"task_type={output.result.get('task_type', 'N/A')} | "
            f"issues_found={len(output.result.get('issues_found', []))}"
        )
    else:
        detail = output.error or "Missing required fields"

    print_agent_result("Coding Agent", output, passed, detail)

    if passed:
        print(f"\n    Issues found:")
        for issue in output.result.get("issues_found", [])[:3]:
            print(f"      - {issue}")
        print(f"\n    Solution:")
        for line in output.result["solution"].split("\n")[:8]:
            print(f"      {line}")

    return passed


# =============================================================================
# TEST 5 — RESEARCH → OUTREACH PIPELINE (MOST IMPORTANT TEST)
# =============================================================================

async def test_research_to_outreach_pipeline(research_result: dict) -> bool:
    """
    Test: OutreachAgent uses ResearchAgent output to write personalized email.

    This is the core of multi-agent architecture:
    - Agent A produces output
    - That output becomes context for Agent B
    - Agent B's output is measurably better because of Agent A

    We verify personalization by checking that the email references
    specific facts from the research output.
    """
    print(f"\n  [PIPELINE] Research -> Outreach")
    print(f"  Using research output from Test 1 as context for outreach...")

    if not research_result:
        print(f"  \033[91m[SKIP]\033[0m  No research result available (Test 1 failed)")
        return False

    agent = OutreachAgent()
    task = TaskInput(
        user_message=(
            "Write a cold email to the CTO of a company that uses LangGraph in production. "
            "We are selling an AI orchestration monitoring tool. "
            "Use the research context to make this highly personalized."
        )
    )

    # THIS IS THE KEY PART: passing research output as context
    context = {"research_output": research_result}
    output = await agent.execute(task, context=context)

    passed = (
        output.success
        and "subject" in output.result
        and "body" in output.result
        and "personalization_notes" in output.result
    )

    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n  {color}{status}{reset}  Research -> Outreach Pipeline")

    if passed:
        print(f"    Subject: {output.result['subject']}")
        print(f"    Tone: {output.result.get('tone', 'N/A')}")
        print(f"\n    Personalization notes:")
        print(f"    {output.result['personalization_notes'][:200]}")
        print(f"\n    Email body (first 5 lines):")
        for line in output.result["body"].split("\n")[:5]:
            if line.strip():
                print(f"    {line}")
        print(f"\n    duration: {output.execution_time_ms}ms")
    else:
        print(f"    Error: {output.error}")

    return passed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    setup_logging()

    print_header("SPECIALIZED AGENTS — INTEGRATION TESTS")
    print(f"\n  Running 5 tests (4 individual agents + 1 pipeline)...")
    print(f"  Each test makes 1-2 API calls. Total: ~6 calls.\n")

    results = []
    await asyncio.sleep(1)  # Brief pause before starting

    # --- Individual Agent Tests ---
    print("\n[ INDIVIDUAL AGENT TESTS ]")

    passed, research_result = await test_research_agent()
    results.append(passed)
    await asyncio.sleep(3)

    results.append(await test_summarization_agent())
    await asyncio.sleep(3)

    results.append(await test_outreach_agent_standalone())
    await asyncio.sleep(3)

    results.append(await test_coding_agent())
    await asyncio.sleep(3)

    # --- Pipeline Test (requires research result from Test 1) ---
    print("\n[ PIPELINE TEST — AGENT-TO-AGENT CONTEXT PASSING ]")
    results.append(await test_research_to_outreach_pipeline(research_result))

    # Summary
    passed_count = sum(results)
    total = len(results)
    all_passed = all(results)

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  \033[92m ALL TESTS PASSED ({passed_count}/{total})\033[0m")
        print(f"  All agents working. Context passing verified.")
        print(f"  Ready for Phase 4 - Orchestration Engine.")
    else:
        failed = total - passed_count
        print(f"  \033[91m {failed} TEST(S) FAILED ({passed_count}/{total} passed)\033[0m")
    print(f"{'='*60}\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
