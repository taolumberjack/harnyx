"""Domain-specific exception types."""

from __future__ import annotations

from caster_commons.errors import BudgetExceededError, ConcurrencyLimitError

__all__ = ["BudgetExceededError", "ConcurrencyLimitError"]
