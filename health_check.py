# =============================================================================
# health_check.py — System Startup Verification
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Before building anything, verify that every component of the foundation works.
# This script tests:
#   1. Config loading (can we read .env?)
#   2. Logger setup (does structlog work?)
#   3. Gemini API connectivity (is the API key valid?)
#   4. Groq API connectivity (is the fallback key valid?)
#   5. ChromaDB initialization (can we create a vector store?)
#   6. Custom exceptions (do they instantiate correctly?)
#
# Run this EVERY TIME you add a new dependency or change configuration.
# A green health check = safe to continue building.
# A red health check = fix the foundation before adding more code.
#
# Usage:
#   cd D:\Orchestrator
#   .\venv\Scripts\Activate.ps1
#   python health_check.py
# =============================================================================

import sys
import asyncio
import traceback


def print_header():
    print("\n" + "="*60)
    print("   AI MULTI-AGENT ORCHESTRATOR — HEALTH CHECK")
    print("="*60 + "\n")


def print_result(check_name: str, passed: bool, detail: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    color = "\033[92m" if passed else "\033[91m"  # green or red
    reset = "\033[0m"
    detail_str = f"  -> {detail}" if detail else ""
    print(f"  {color}{status}{reset}  {check_name}{detail_str}")


def check_1_config():
    """Test: Can we load and validate settings from .env?"""
    try:
        from core.config import settings
        assert settings.gemini_api_key, "GEMINI_API_KEY is empty"
        assert settings.groq_api_key, "GROQ_API_KEY is empty"
        assert settings.router_model, "ROUTER_MODEL is empty"
        assert settings.agent_model, "AGENT_MODEL is empty"
        print_result("Config loading", True,
                     f"env={settings.app_env}, router_model={settings.router_model}")
        return True
    except Exception as e:
        print_result("Config loading", False, str(e))
        return False


def check_2_logger():
    """Test: Does the logging system initialize without errors?"""
    try:
        from core.logger import setup_logging, get_logger
        setup_logging()
        logger = get_logger("health_check")
        logger.info("health_check_started", check="logger")
        print_result("Logging system", True, "structlog initialized")
        return True
    except Exception as e:
        print_result("Logging system", False, str(e))
        return False


def check_3_exceptions():
    """Test: Do custom exceptions work correctly?"""
    try:
        from core.exceptions import (
            AgentExecutionError,
            RouterError,
            LLMProviderError,
            MaxRetriesExceededError,
        )
        # Test that each exception can be instantiated and caught properly
        err = AgentExecutionError("test error", agent_name="test_agent")
        assert err.agent_name == "test_agent"

        err2 = LLMProviderError("rate limit", provider="gemini", status_code=429)
        assert err2.provider == "gemini"
        assert err2.status_code == 429

        err3 = MaxRetriesExceededError("gave up", agent_name="research", retries=3)
        assert err3.retries == 3

        print_result("Custom exceptions", True, "all exception classes valid")
        return True
    except Exception as e:
        print_result("Custom exceptions", False, str(e))
        return False


async def check_4_gemini():
    """Test: Can we reach the Gemini API and get a response?"""
    try:
        # Using new google.genai SDK (google.generativeai is deprecated)
        from google import genai
        from core.config import settings

        client = genai.Client(api_key=settings.gemini_api_key)

        # Try models in order — flash lite is cheapest, good for health checks
        models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Reply with exactly one word: HEALTHY"
                )
                response_text = response.text.strip()
                print_result("Gemini API", True,
                             f"model={model_name}, response='{response_text}'")
                return True
            except Exception as model_err:
                last_error = model_err
                continue

        raise last_error

    except Exception as e:
        # Show only first 200 chars of error — full error is very verbose
        print_result("Gemini API", False, str(e)[:200])
        return False


async def check_5_groq():
    """Test: Can we reach the Groq API and get a response?"""
    try:
        from groq import Groq
        from core.config import settings

        client = Groq(api_key=settings.groq_api_key)

        # llama3-8b-8192 is decommissioned — using llama-3.1-8b-instant
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Reply with exactly one word: HEALTHY"}],
            max_tokens=10,
        )

        response_text = response.choices[0].message.content.strip()
        print_result("Groq API", True,
                     f"model=llama-3.1-8b-instant, response='{response_text}'")
        return True
    except Exception as e:
        print_result("Groq API", False, str(e))
        return False


def check_6_chromadb():
    """Test: Can we initialize ChromaDB and create a collection?"""
    try:
        import chromadb
        from core.config import settings

        # Create a persistent ChromaDB client
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        # Create or get a test collection
        collection = client.get_or_create_collection(
            name="health_check_test"
        )

        # Add a test document
        collection.add(
            documents=["health check test document"],
            ids=["health_check_001"]
        )

        # Query it back
        results = collection.query(
            query_texts=["health check"],
            n_results=1
        )

        assert len(results["documents"][0]) > 0, "No results returned"

        # Clean up the test collection
        client.delete_collection("health_check_test")

        print_result("ChromaDB", True,
                     f"persist_dir={settings.chroma_persist_dir}")
        return True
    except Exception as e:
        print_result("ChromaDB", False, str(e))
        return False


def check_7_imports():
    """Test: Do all critical packages import without errors?"""
    packages = {
        "fastapi": "FastAPI",
        "pydantic": "Pydantic v2",
        "langgraph": "LangGraph",
        "langchain_core": "LangChain Core",
        "langchain_google_genai": "LangChain Google GenAI",
        "structlog": "Structlog",
        "httpx": "HTTPX",
        "chromadb": "ChromaDB",
    }

    all_passed = True
    for module, display_name in packages.items():
        try:
            __import__(module)
        except ImportError as e:
            print_result(f"Import: {display_name}", False, str(e))
            all_passed = False

    if all_passed:
        print_result("All package imports", True,
                     f"{len(packages)} packages verified")
    return all_passed


async def run_all_checks():
    """Run all health checks and return overall status."""
    print_header()

    results = []

    print("[ SYSTEM CHECKS ]")
    results.append(check_7_imports())

    print("\n[ CONFIGURATION ]")
    config_ok = check_1_config()
    results.append(config_ok)

    print("\n[ LOGGING ]")
    results.append(check_2_logger())

    print("\n[ EXCEPTIONS ]")
    results.append(check_3_exceptions())

    print("\n[ API CONNECTIVITY ]")
    results.append(await check_4_gemini())
    results.append(await check_5_groq())

    print("\n[ MEMORY LAYER ]")
    results.append(check_6_chromadb())

    # Final summary
    passed = sum(results)
    total = len(results)
    all_green = all(results)

    print("\n" + "="*60)
    if all_green:
        print(f"  \033[92m ALL CHECKS PASSED ({passed}/{total})\033[0m")
        print("  Foundation is solid. Ready for Phase 2 - Router Agent.")
    else:
        failed = total - passed
        print(f"  \033[91m {failed} CHECK(S) FAILED ({passed}/{total} passed)\033[0m")
        print("  Fix the failing checks before proceeding.")
    print("="*60 + "\n")

    return all_green


if __name__ == "__main__":
    success = asyncio.run(run_all_checks())
    sys.exit(0 if success else 1)
