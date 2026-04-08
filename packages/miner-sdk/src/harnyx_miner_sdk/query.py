"""Query request/response contracts for miners."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_MINER_SDK_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)
_MAX_RESPONSE_CITATIONS = 50


class Query(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1)


class CitationRef(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    receipt_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)


class Response(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1)
    citations: list[CitationRef] | None = Field(default=None, max_length=_MAX_RESPONSE_CITATIONS)


__all__ = ["CitationRef", "Query", "Response"]
