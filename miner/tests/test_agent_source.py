from __future__ import annotations

from pathlib import Path

import pytest

from harnyx_miner.agent_source import load_agent_query_entrypoint, validate_agent_query_entrypoint
from harnyx_miner_sdk.decorators import clear_entrypoints, entrypoint_exists, get_entrypoint


def _write_agent(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_load_agent_query_entrypoint_rejects_missing_query(tmp_path: Path) -> None:
    agent_path = _write_agent(
        tmp_path / "agent.py",
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "@entrypoint('other')\n"
        "async def other(request: dict[str, object]) -> dict[str, object]:\n"
        "    return request\n",
    )

    with pytest.raises(RuntimeError, match="agent did not register entrypoint 'query'"):
        load_agent_query_entrypoint(agent_path)
    clear_entrypoints()


def test_load_agent_query_entrypoint_keeps_registered_query_available(tmp_path: Path) -> None:
    agent_path = _write_agent(
        tmp_path / "agent.py",
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "from harnyx_miner_sdk.query import Query, Response\n"
        "@entrypoint('query')\n"
        "async def query(query: Query) -> Response:\n"
        "    return Response(text=query.text)\n",
    )

    load_agent_query_entrypoint(agent_path)

    assert entrypoint_exists("query")
    assert callable(get_entrypoint("query"))
    clear_entrypoints()


def test_validate_agent_query_entrypoint_rejects_invalid_query_signature(tmp_path: Path) -> None:
    agent_path = _write_agent(
        tmp_path / "agent.py",
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "from harnyx_miner_sdk.query import Response\n"
        "@entrypoint('query')\n"
        "async def query(query: str) -> Response:\n"
        "    return Response(text=query)\n",
    )

    with pytest.raises(TypeError, match="query entrypoint parameter"):
        validate_agent_query_entrypoint(agent_path)


def test_validate_agent_query_entrypoint_clears_registry_after_submit_preflight(tmp_path: Path) -> None:
    agent_path = _write_agent(
        tmp_path / "agent.py",
        "from harnyx_miner_sdk.decorators import entrypoint\n"
        "from harnyx_miner_sdk.query import Query, Response\n"
        "@entrypoint('query')\n"
        "async def query(query: Query) -> Response:\n"
        "    return Response(text=query.text)\n",
    )

    validate_agent_query_entrypoint(agent_path)

    assert not entrypoint_exists("query")
