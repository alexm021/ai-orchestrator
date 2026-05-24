# =============================================================================
# core/exceptions.py — Custom Exception Hierarchy
# =============================================================================
#
# WHY CUSTOM EXCEPTIONS:
# Python's built-in exceptions (ValueError, RuntimeError) tell you WHAT broke.
# Custom exceptions tell you WHERE and WHY in your system it broke.
#
# Without custom exceptions:
#   except Exception as e:
#       print(f"Something broke: {e}")  # useless
#
# With custom exceptions:
#   except AgentExecutionError as e:
#       logger.error("agent_failed", agent=e.agent_name, task=e.task_id)
#       # now you know exactly which agent failed on which task
#
# This is the foundation of error handling across the entire system.
# Every layer (router, orchestrator, agents) raises its own specific exception.
# =============================================================================


class OrchestratorBaseError(Exception):
    """
    Base class for all custom exceptions in this system.
    Every other exception inherits from this.

    WHY: Lets you catch ALL orchestrator errors with one except clause:
        except OrchestratorBaseError as e:
            handle_any_orchestrator_error(e)
    """
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        # details = optional context dict (task_id, agent_name, etc.)
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(OrchestratorBaseError):
    """
    Raised when the system configuration is invalid or incomplete.
    Example: Missing API key, invalid model name, bad environment variable.
    """
    pass


# =============================================================================
# Agent Errors
# =============================================================================

class AgentError(OrchestratorBaseError):
    """
    Base class for all agent-related errors.
    Includes the agent name so you always know which agent failed.
    """
    def __init__(self, message: str, agent_name: str, details: dict | None = None):
        super().__init__(message, details)
        self.agent_name = agent_name


class AgentExecutionError(AgentError):
    """
    Raised when an agent fails to complete its task.
    Example: LLM returned invalid output, tool call failed.
    """
    pass


class AgentTimeoutError(AgentError):
    """
    Raised when an agent exceeds its time limit.
    Example: API call took longer than AGENT_TIMEOUT_SECONDS.
    """
    pass


class AgentValidationError(AgentError):
    """
    Raised when an agent's output fails Pydantic validation.
    Example: LLM returned malformed JSON, missing required field.
    """
    pass


# =============================================================================
# Router Errors
# =============================================================================

class RouterError(OrchestratorBaseError):
    """
    Raised when the router agent fails to classify or route a task.
    """
    pass


class UnroutableTaskError(RouterError):
    """
    Raised when the router cannot determine which agent should handle a task.
    Example: Ambiguous or completely unsupported task type.
    """
    pass


# =============================================================================
# Orchestration Errors
# =============================================================================

class OrchestrationError(OrchestratorBaseError):
    """
    Raised when the orchestration engine fails to coordinate agents.
    Example: Agent dependency chain broke, state became invalid.
    """
    pass


class MaxRetriesExceededError(OrchestrationError):
    """
    Raised when an agent has been retried MAX_AGENT_RETRIES times and still fails.
    The orchestrator gives up and returns this to the caller.
    """
    def __init__(self, message: str, agent_name: str, retries: int, details: dict | None = None):
        super().__init__(message, details)
        self.agent_name = agent_name
        self.retries = retries


# =============================================================================
# Memory Errors
# =============================================================================

class MemoryError(OrchestratorBaseError):
    """
    Raised when the memory layer (ChromaDB) fails.
    Example: Cannot write to vector store, collection not found.
    """
    pass


# =============================================================================
# LLM Provider Errors
# =============================================================================

class LLMProviderError(OrchestratorBaseError):
    """
    Raised when the LLM API call fails.
    Example: Rate limit hit, API key invalid, network timeout.
    """
    def __init__(self, message: str, provider: str, status_code: int | None = None, details: dict | None = None):
        super().__init__(message, details)
        self.provider = provider          # "gemini" or "groq"
        self.status_code = status_code    # HTTP status if available


class RateLimitError(LLMProviderError):
    """
    Raised specifically when we hit API rate limits.
    The orchestrator uses this to trigger backoff + retry logic.
    """
    pass
