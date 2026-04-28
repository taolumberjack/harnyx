"""Port definitions for interacting with the Bittensor subtensor network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CommitmentRecord:
    """Miner commitment metadata stored on-chain."""

    block: int
    data: str


@dataclass(frozen=True, slots=True)
class MetagraphSnapshot:
    """Lightweight snapshot of the active subnet metagraph."""

    uids: Sequence[int]
    hotkeys: Sequence[str]


@dataclass(frozen=True, slots=True)
class ValidatorNodeInfo:
    """Information about the validator's on-chain identity."""

    uid: int
    version_key: int | None


class WeightSubmissionCadenceStatus(StrEnum):
    """Result status for validator weight-submission cadence checks."""

    OPEN = "open"
    UNREGISTERED = "unregistered"
    METADATA_UNAVAILABLE = "metadata_unavailable"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class WeightSubmissionCadence:
    """Chain-owned weight-submission cadence state for the active validator."""

    status: WeightSubmissionCadenceStatus
    validator_uid: int | None
    commit_reveal_enabled: bool
    current_block: int | None
    last_update_block: int | None
    blocks_since_last_update: int | None
    weights_rate_limit: int | None

    @property
    def can_submit(self) -> bool:
        """Return True when the validator should attempt weight submission now."""

        return self.status is WeightSubmissionCadenceStatus.OPEN


class SubtensorClientPort(Protocol):
    """Abstract client responsible for subtensor interactions."""

    def connect(self) -> None:
        """Ensure the underlying client is ready for use (idempotent)."""

    def close(self) -> None:
        """Release any held resources."""

    def fetch_metagraph(self) -> MetagraphSnapshot:
        """Return the active subnet metagraph."""

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        """Return the latest commitment for ``uid`` when available."""

    def publish_commitment(self, data: str, *, blocks_until_reveal: int = 1) -> CommitmentRecord:
        """Publish a new commitment for the validator."""

    def current_block(self) -> int:
        """Return the current chain block height."""

    def last_update_block(self, uid: int) -> int | None:
        """Return the block height of the most recent weight update for ``uid``."""

    def weight_submission_cadence(self, netuid: int) -> WeightSubmissionCadence:
        """Return chain-owned cadence state for validator weight submission."""

    def validator_info(self) -> ValidatorNodeInfo:
        """Return validator node metadata (UID, version key, etc.)."""

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        """Submit normalized weights and return the transaction reference/hash."""

    def fetch_weight(self, uid: int) -> float:
        """Return this validator's current on-chain weight for target ``uid`` (0.0 when absent)."""

    def tempo(self, netuid: int) -> int:
        """Return the epoch tempo (blocks per epoch) for ``netuid``."""

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        """Return the block height where the next epoch starts for ``netuid``."""


__all__ = [
    "CommitmentRecord",
    "MetagraphSnapshot",
    "SubtensorClientPort",
    "ValidatorNodeInfo",
    "WeightSubmissionCadence",
    "WeightSubmissionCadenceStatus",
]
