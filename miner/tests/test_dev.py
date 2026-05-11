from __future__ import annotations

import json

from harnyx_miner.dev import _serialize_response
from harnyx_miner_sdk.query import CitationRef, CitationSlice, Response


def test_serialize_response_emits_json_object() -> None:
    payload = json.loads(_serialize_response(Response(text="hello")))

    assert payload == {"text": "hello"}


def test_serialize_response_preserves_raw_citation_refs() -> None:
    payload = json.loads(
        _serialize_response(
            Response(
                text="hello",
                citations=[CitationRef(receipt_id="receipt-1", result_id="result-1")],
            )
        )
    )

    assert payload == {
        "text": "hello",
        "citations": [{"receipt_id": "receipt-1", "result_id": "result-1", "slices": []}],
    }


def test_serialize_response_preserves_targeted_citation_slices() -> None:
    payload = json.loads(
        _serialize_response(
            Response(
                text="hello",
                citations=[
                    CitationRef(
                        receipt_id="receipt-1",
                        result_id="result-1",
                        slices=[CitationSlice(start=0, end=120)],
                    )
                ],
            )
        )
    )

    assert payload == {
        "text": "hello",
        "citations": [{"receipt_id": "receipt-1", "result_id": "result-1", "slices": [{"start": 0, "end": 120}]}],
    }
