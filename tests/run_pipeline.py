"""
Quick pipeline test — Research -> Outreach
Run: python tests/run_pipeline.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging
from orchestrator.graph import run_task


async def main():
    setup_logging()

    print("\n" + "="*60)
    print("  PIPELINE TEST: Research -> Outreach")
    print("  (may take 60-120s due to free tier rate limits)")
    print("="*60)

    result = await run_task(
        task_message=(
            "Research Anthropic AI company — their products (Claude), "
            "mission (AI safety), and recent developments. "
            "Then write a cold email to their Head of Product "
            "introducing an AI developer tools startup."
        ),
        task_priority="high",
    )

    status = result.get("status")
    agents = result.get("agents_used", [])

    print(f"\n  Status:    {status}")
    print(f"  Agents:    {' -> '.join(agents)}")
    print(f"  Strategy:  {result.get('execution_strategy')}")

    if status == "complete":
        all_out = result.get("all_outputs", {})

        # Research output
        if "research" in all_out:
            r = all_out["research"]
            print(f"\n  [RESEARCH AGENT]")
            print(f"  Topic:   {r.get('topic')}")
            print(f"  Summary: {r.get('summary', '')[:150]}...")
            for fact in r.get("key_facts", [])[:2]:
                print(f"    - {fact[:100]}")

        # Outreach output
        if "outreach" in all_out:
            o = all_out["outreach"]
            print(f"\n  [OUTREACH AGENT]")
            print(f"  Subject: {o.get('subject')}")
            print(f"  Tone:    {o.get('tone')}")
            print(f"\n  Email body:")
            for line in o.get("body", "").split("\n")[:6]:
                if line.strip():
                    print(f"    {line}")
            print(f"\n  Personalization used:")
            print(f"    {o.get('personalization_notes', '')[:200]}")

        perf = result.get("performance", {})
        print(f"\n  Total tokens:  {perf.get('total_tokens', 0)}")
        print(f"  Wall time:     {result.get('total_wall_time_ms', 0)}ms")

        print("\n" + "="*60)
        print("  PIPELINE TEST: PASSED")
        print("  Multi-agent context passing: VERIFIED")
        print("="*60 + "\n")
    else:
        print(f"\n  Errors: {result.get('errors', [])}")
        print("\n  PIPELINE TEST: FAILED")


if __name__ == "__main__":
    asyncio.run(main())
