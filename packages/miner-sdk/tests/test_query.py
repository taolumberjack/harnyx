from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.query import CitationRef, Query, Response


def test_query_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Query.model_validate({"text": "   "})


def test_query_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Query.model_validate({"text": "hello", "other": "nope"})


def test_response_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate({"text": ""})


def test_response_accepts_optional_citation_refs() -> None:
    response = Response.model_validate(
        {
            "text": "hello",
            "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
        }
    )

    assert response == Response(
        text="hello",
        citations=[CitationRef(receipt_id="receipt-1", result_id="result-1")],
    )


def test_response_rejects_more_than_fifty_citations() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(
            {
                "text": "hello",
                "citations": [
                    {"receipt_id": f"receipt-{index}", "result_id": f"result-{index}"}
                    for index in range(51)
                ],
            }
        )
