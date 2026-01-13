"""Port describing access to reference claims batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from caster_commons.domain.claim import MinerTaskClaim


class ClaimsProviderPort(Protocol):
    """Supplies the set of reference claims used for a miner-task batch."""

    def fetch(self, *, batch_id: UUID | None = None) -> Sequence[MinerTaskClaim]:
        """Return the ordered collection of claims for the supplied batch."""


__all__ = ["ClaimsProviderPort"]
