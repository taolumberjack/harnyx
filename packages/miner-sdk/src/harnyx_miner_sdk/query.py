"""Query request/response contracts for miners."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MINER_SDK_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)
_MAX_RESPONSE_CHARS = 80_000
_MAX_RESPONSE_CITATIONS = 200
_MAX_RESPONSE_EVIDENCE_SEGMENTS = 400


class Query(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1)


class CitationSlice(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.end <= self.start:
            raise ValueError("citation slice end must be greater than start")
        return self


class CitationRef(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    receipt_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    slices: list[CitationSlice] = Field(default_factory=list)


class Response(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1, max_length=_MAX_RESPONSE_CHARS)
    citations: list[CitationRef] | None = Field(default=None, max_length=_MAX_RESPONSE_CITATIONS)

    @model_validator(mode="after")
    def validate_total_evidence_segments(self) -> Self:
        total_segments = sum(len(citation.slices) if citation.slices else 1 for citation in self.citations or ())
        if total_segments > _MAX_RESPONSE_EVIDENCE_SEGMENTS:
            raise ValueError("response citations exceed 400 materialized evidence segments")
        return self


__all__ = ["CitationRef", "CitationSlice", "Query", "Response"]
