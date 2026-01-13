"""Shared parsing helpers for validator infrastructure."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter

from caster_commons.domain.claim import MinerTaskClaim
from caster_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec


class _ScriptArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: int
    artifact_id: UUID
    content_hash: str
    size_bytes: int


class _BatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: UUID
    entrypoint: str
    cutoff_at_iso: str = Field(validation_alias=AliasChoices("cutoff_at_iso", "cutoff_at"))
    created_at_iso: str = Field(validation_alias=AliasChoices("created_at_iso", "created_at"))
    claims: tuple[MinerTaskClaim, ...]
    candidates: tuple[_ScriptArtifactPayload, ...]
    champion_uid: int | None = None
    status: str | None = None
    status_message: str | None = None


_BATCH_PAYLOAD_ADAPTER = TypeAdapter(_BatchPayload)


def parse_batch(payload: Mapping[str, object]) -> MinerTaskBatchSpec:
    """Normalize raw batch payloads into MinerTaskBatchSpec."""
    parsed = _BATCH_PAYLOAD_ADAPTER.validate_python(payload)
    candidates = tuple(
        ScriptArtifactSpec(
            uid=item.uid,
            artifact_id=item.artifact_id,
            content_hash=item.content_hash,
            size_bytes=item.size_bytes,
        )
        for item in parsed.candidates
    )

    return MinerTaskBatchSpec(
        batch_id=parsed.batch_id,
        entrypoint=parsed.entrypoint,
        cutoff_at_iso=parsed.cutoff_at_iso,
        created_at_iso=parsed.created_at_iso,
        claims=parsed.claims,
        candidates=candidates,
    )


__all__ = ["parse_batch"]
