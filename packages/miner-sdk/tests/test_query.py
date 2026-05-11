from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.query import CitationRef, CitationSlice, Query, Response


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
    assert response.citations is not None
    assert response.citations[0].slices == []


def test_response_accepts_targeted_citation_slices() -> None:
    response = Response.model_validate(
        {
            "text": "hello",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": 0, "end": 120}],
                }
            ],
        }
    )

    assert response == Response(
        text="hello",
        citations=[
            CitationRef(
                receipt_id="receipt-1",
                result_id="result-1",
                slices=[CitationSlice(start=0, end=120)],
            )
        ],
    )


def test_response_rejects_more_than_two_hundred_citations() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(
            {
                "text": "hello",
                "citations": [
                    {"receipt_id": f"receipt-{index}", "result_id": f"result-{index}"}
                    for index in range(201)
                ],
            }
        )


def test_response_rejects_more_than_four_hundred_materialized_segments() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(
            {
                "text": "hello",
                "citations": [
                    {
                        "receipt_id": "receipt-1",
                        "result_id": "result-1",
                        "slices": [{"start": index * 100, "end": (index + 1) * 100} for index in range(401)],
                    }
                ],
            }
        )


def test_response_rejects_text_longer_than_eighty_thousand_chars() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate({"text": "x" * 80_001})


def test_citation_slice_requires_end_after_start() -> None:
    with pytest.raises(ValidationError):
        CitationSlice(start=10, end=10)
