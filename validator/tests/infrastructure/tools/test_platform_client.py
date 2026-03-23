from __future__ import annotations

import re
from uuid import uuid4

import bittensor as bt
import httpx
import pytest

import harnyx_validator.infrastructure.tools.feed_search_provider as feed_search_provider_module
from harnyx_commons.bittensor import build_canonical_request
from harnyx_validator.infrastructure.tools.feed_search_provider import HttpFeedSearchToolProvider
from harnyx_validator.infrastructure.tools.platform_client import HttpPlatformClient

_HEADER_PATTERN = re.compile(
    r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-f]+)"$'
)


def _keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _assert_signed(request: httpx.Request, keypair: bt.Keypair) -> None:
    header = request.headers.get("Authorization")
    assert header is not None
    match = _HEADER_PATTERN.match(header)
    assert match is not None
    assert match.group("ss58") == keypair.ss58_address
    path = request.url.raw_path.decode()
    query = request.url.query
    if query:
        path = f"{path}?{query}"
    body = request.content or b""
    canonical = build_canonical_request(request.method, path, body)
    signature = bytes.fromhex(match.group("sig"))
    assert keypair.verify(canonical, signature)


def test_get_champion_weights_returns_weights() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == "/v1/weights":
            payload = {
                "weights": {"42": 0.7, "7": 0.3},
                "champion_uid": 42,
            }
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    weights = client.get_champion_weights()

    assert weights.weights == {42: 0.7, 7: 0.3}
    assert weights.champion_uid == 42


def test_get_miner_task_batch_parses_tasks_and_artifacts() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    champion_artifact_id = uuid4()
    budget_usd = 0.123

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        expected_path = f"/v1/miner-task-batches/batch/{batch_id}"
        if request.method == "GET" and request.url.path == expected_path:
            payload = {
                "batch_id": str(batch_id),
                "cutoff_at": "2025-10-17T12:00:00Z",
                "created_at": "2025-10-17T12:00:00Z",
                "tasks": [
                    {
                        "task_id": str(task_id),
                        "query": {"text": "smoke"},
                        "reference_answer": {"text": "ok"},
                        "budget_usd": budget_usd,
                    },
                ],
                "artifacts": [
                    {
                        "uid": 7,
                        "artifact_id": str(artifact_id),
                        "content_hash": "abc",
                        "size_bytes": 1,
                    }
                ],
                "champion_artifact_id": str(champion_artifact_id),
                "completed_at": "2025-10-17T12:05:00Z",
                "failed_at": None,
            }
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    batch = client.get_miner_task_batch(batch_id)

    assert batch.batch_id == batch_id
    assert batch.tasks[0].task_id == task_id
    assert batch.tasks[0].budget_usd == pytest.approx(budget_usd)
    assert batch.tasks[0].query.text == "smoke"
    assert batch.tasks[0].reference_answer.text == "ok"
    assert batch.artifacts[0].artifact_id == artifact_id


@pytest.mark.anyio
async def test_feed_search_provider_offloads_signed_header_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def _record_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        assert request.method == "POST"
        assert request.url.path == "/v1/feeds/search"
        return httpx.Response(status_code=200, json={"items": []})

    monkeypatch.setattr(feed_search_provider_module.asyncio, "to_thread", _record_to_thread)
    keypair = _keypair()
    provider = HttpFeedSearchToolProvider(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    payload = await provider.search_items(
        feed_id=uuid4(),
        enqueue_seq=1,
        search_queries=("demo",),
        num_hit=2,
    )

    assert payload == {"items": []}
    assert len(to_thread_calls) == 1
    called_func, called_args, called_kwargs = to_thread_calls[0]
    assert getattr(called_func, "__self__", None) is provider
    assert getattr(called_func, "__func__", None) is HttpFeedSearchToolProvider._signed_header
    assert called_args[0] == "POST"
    assert called_args[1] == "/v1/feeds/search"
    assert called_args[2]
    assert called_kwargs == {}
