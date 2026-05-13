"""Domain-specific exceptions shared across sandbox components."""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for sandbox-specific failures."""


class MissingEntrypointError(SandboxError):
    """Raised when an entrypoint name has not been registered."""


class BudgetExceededError(RuntimeError):
    """Raised when a tool call exceeds the configured session budget."""


class SessionBudgetExhaustedError(RuntimeError):
    """Raised when execution exhausts the session hard limit."""


class ConcurrencyLimitError(RuntimeError):
    """Raised when a token exceeds its permitted parallel call allowance."""


class ToolInvocationTimeoutError(RuntimeError):
    """Raised when a caller-selected tool invocation deadline expires."""


class ToolProviderError(RuntimeError):
    """Raised when a tool's backing provider fails after retry exhaustion."""


__all__ = [
    "SandboxError",
    "MissingEntrypointError",
    "BudgetExceededError",
    "SessionBudgetExhaustedError",
    "ConcurrencyLimitError",
    "ToolInvocationTimeoutError",
    "ToolProviderError",
]
