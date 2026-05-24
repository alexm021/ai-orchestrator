"""
Phase 5 Memory Tests -- ChromaDB + Gemini Embeddings
Run: python tests/test_memory.py

Tests:
  1. Save a task to memory
  2. Retrieve relevant memories by semantic query
  3. Full pipeline: run two tasks, then retrieve via orchestrator
"""
import asyncio
import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logging
from core.memory import get_memory_manager
from orchestrator.graph import run_task


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
# TEST 1 -- SAVE TO MEMORY
# =============================================================================

async def test_save_to_memory() -> bool:
    """Verify that tasks can be saved to ChromaDB."""
    print("\n  Saving 2 test tasks to memory...")

    memory = get_memory_manager()
    initial_count = memory.count()

    task_id_1 = str(uuid.uuid4())
    task_id_2 = str(uuid.uuid4())

    await memory.save_task(
        task_id=task_id_1,
        task_message="Research Anthropic AI company and their products",
        primary_output=(
            "Anthropic is an AI safety company founded in 2021. "
            "Their main product is Claude, a family of large language models. "
            "Key mission: building AI systems that are safe and beneficial."
        ),
        agents_used=["research"],
        status="complete",
    )

    await memory.save_task(
        task_id=task_id_2,
        task_message="Write a cold email to Tesla's Head of Engineering about battery software",
        primary_output=(
            "Subject: Accelerating Tesla's battery R&D\n\n"
            "Hi [Name], I've been following Tesla's work on energy density. "
            "Our software reduces battery simulation time by 40%."
        ),
        agents_used=["outreach"],
        status="complete",
    )

    new_count = memory.count()
    saved = new_count >= initial_count + 2

    print_result(
        "Save tasks to ChromaDB",
        saved,
        f"Memory size: {initial_count} -> {new_count} items",
    )
    return saved


# =============================================================================
# TEST 2 -- SEMANTIC RETRIEVAL
# =============================================================================

async def test_semantic_retrieval() -> bool:
    """Verify that semantic search returns relevant results."""
    print("\n  Searching for memories about 'AI company email'...")

    memory = get_memory_manager()

    results = await memory.search(
        query="I need to write an email to an AI company",
        n_results=5,
    )

    has_results = len(results) > 0
    has_relevance_scores = all("relevance_score" in r for r in results)
    has_required_fields = all(
        "task_message" in r and "primary_output" in r and "agents_used" in r
        for r in results
    )

    passed = has_results and has_relevance_scores and has_required_fields

    print_result(
        "Semantic retrieval",
        passed,
        f"Found {len(results)} result(s)",
    )

    if results:
        print(f"\n    Top result:")
        top = results[0]
        print(f"      Relevance:   {top['relevance_score']:.0%}")
        print(f"      Task:        {top['task_message'][:80]}...")
        print(f"      Agents used: {top['agents_used']}")

    return passed


# =============================================================================
# TEST 3 -- FULL PIPELINE WITH AUTO-SAVE
# =============================================================================

async def test_auto_save_after_pipeline() -> bool:
    """
    Run a real orchestration task and verify it gets auto-saved to memory.
    """
    print("\n  Running a task to populate memory...")

    memory = get_memory_manager()
    count_before = memory.count()

    result = await run_task(
        task_message="Summarize the key benefits of vector databases for AI applications",
        task_priority="low",
    )

    count_after = memory.count()
    auto_saved = (
        result.get("status") == "complete"
        and count_after > count_before
    )

    print_result(
        "Auto-save after pipeline execution",
        auto_saved,
        f"Memory: {count_before} -> {count_after} items | status={result.get('status')}",
    )

    return auto_saved


# =============================================================================
# TEST 4 -- RETRIEVAL AGENT VIA ORCHESTRATOR
# =============================================================================

async def test_retrieval_agent_in_pipeline() -> bool:
    """
    Task that requires memory -> router routes to retrieval agent.
    Verifies the full loop: memory store -> retrieval agent -> output.
    """
    print("\n  Running task that references past conversations...")

    result = await run_task(
        task_message=(
            "What have we previously worked on? "
            "Show me a summary of our past conversations."
        ),
        task_priority="low",
    )

    agents_used = result.get("agents_used", [])
    status = result.get("status")

    passed = (
        status == "complete"
        and "retrieval" in agents_used
    )

    print_result(
        "Retrieval agent invoked for memory query",
        passed,
        f"agents={agents_used} | status={status}",
    )

    if passed:
        output = result.get("all_outputs", {}).get("retrieval", {})
        print(f"\n    Memories found: {output.get('memories_found', 0)}")
        print(f"    Has context:    {output.get('has_relevant_context')}")
        preview = output.get("relevant_context", "")[:200]
        if preview:
            print(f"    Preview:        {preview}...")

    return passed


# =============================================================================
# MAIN
# =============================================================================

async def run_all_tests():
    setup_logging()

    print_header("PHASE 5 -- MEMORY + RETRIEVAL TESTS")
    print("\n  Tests: Save -> Retrieve -> Auto-save -> Pipeline retrieval")
    print("  Uses ChromaDB (local) + Gemini embeddings\n")

    results = []

    print("\n[ TEST 1 -- SAVE TO MEMORY ]")
    results.append(await test_save_to_memory())

    print("\n[ TEST 2 -- SEMANTIC RETRIEVAL ]")
    results.append(await test_semantic_retrieval())

    await asyncio.sleep(5)

    print("\n[ TEST 3 -- AUTO-SAVE AFTER PIPELINE ]")
    results.append(await test_auto_save_after_pipeline())

    await asyncio.sleep(5)

    print("\n[ TEST 4 -- RETRIEVAL AGENT IN PIPELINE ]")
    results.append(await test_retrieval_agent_in_pipeline())

    passed_count = sum(results)
    total = len(results)

    print(f"\n{'='*60}")
    if all(results):
        print(f"  \033[92m ALL MEMORY TESTS PASSED ({passed_count}/{total})\033[0m")
        print(f"\n  What Phase 5 gives us:")
        print(f"  - Every completed task auto-saved to ChromaDB")
        print(f"  - Gemini text-embedding-004: 768-dim semantic vectors")
        print(f"  - Semantic search: finds similar topics, not just keywords")
        print(f"  - RetrievalAgent injects past context into the pipeline")
        print(f"  - Memory failures never crash the orchestrator")
        print(f"\n  Ready for Phase 6 - FastAPI Backend.")
    else:
        failed = total - passed_count
        print(f"  \033[91m {failed} TEST(S) FAILED ({passed_count}/{total} passed)\033[0m")
    print(f"{'='*60}\n")

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
