from __future__ import annotations

from pathlib import Path

import pytest

import harnyx_miner.submit as submit_module


def _write_agent(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_upload_agent_rejects_missing_query_before_wallet_or_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = _write_agent(
        tmp_path / "agent.py",
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "@entrypoint('other')\n"
        "async def other(request: dict[str, object]) -> dict[str, object]:\n"
        "    return request\n",
    )
    monkeypatch.setattr(
        submit_module.bt,
        "wallet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wallet should not be reached")),
    )
    monkeypatch.setattr(
        submit_module.httpx,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("http client should not be reached")),
    )

    with pytest.raises(RuntimeError, match="agent did not register entrypoint 'query'"):
        submit_module._upload_agent(agent_path=agent_path, wallet_name="wallet", hotkey_name="hotkey")


def test_upload_agent_rejects_invalid_query_signature_before_wallet_or_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = _write_agent(
        tmp_path / "agent.py",
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "from harnyx_miner_sdk.query import Response\n"
        "@entrypoint('query')\n"
        "async def query(query: str) -> Response:\n"
        "    return Response(text=query)\n",
    )
    monkeypatch.setattr(
        submit_module.bt,
        "wallet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wallet should not be reached")),
    )
    monkeypatch.setattr(
        submit_module.httpx,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("http client should not be reached")),
    )

    with pytest.raises(TypeError, match="query entrypoint parameter"):
        submit_module._upload_agent(agent_path=agent_path, wallet_name="wallet", hotkey_name="hotkey")
