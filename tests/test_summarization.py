"""Quick isolated test for the Summarization Agent."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging
from orchestrator.graph import run_task


async def main():
    setup_logging()

    result = await run_task(
        task_message=(
            "Summarize this text into 3 bullet points:\n\n"
            "    AI agents are autonomous systems that perceive their environment, "
            "make decisions, and take actions to achieve specific goals. Unlike "
            "traditional software that follows explicit programmed instructions, "
            "agents can adapt their behavior based on context and feedback. Modern "
            "AI agents built on large language models can perform complex reasoning, "
            "use external tools, and collaborate with other agents to solve "
            "sophisticated tasks that would be impossible for any single model to handle alone."
        ),
        task_priority="low",
    )

    agents = result.get("agents_used", [])
    status = result.get("status")
    passed = status == "complete" and "summarization" in agents

    print(f"\nStatus:  {status}")
    print(f"Agents:  {agents}")
    print(f"Strategy: {result.get('execution_strategy')}")

    if passed:
        summ = result.get("all_outputs", {}).get("summarization", {})
        print(f"\nTL;DR:   {summ.get('summary', '')}")
        print(f"Topic:   {summ.get('main_topic', '')}")
        print(f"Tone:    {summ.get('tone', '')}")
        print("\nKey points:")
        for p in summ.get("key_points", []):
            print(f"  - {p}")
        perf = result.get("performance", {})
        print(f"\nTokens:  {perf.get('total_tokens', 0)}")
        print(f"Time:    {result.get('total_wall_time_ms', 0)}ms")
        print("\n[PASS] Test 3 - Summarization PASSED")
    else:
        print(f"\nErrors: {result.get('errors')}")
        print("[FAIL] Test 3 - Summarization FAILED")

    return passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
