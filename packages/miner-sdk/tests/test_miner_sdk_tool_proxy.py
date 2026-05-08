from __future__ import annotations

import pytest

from harnyx_miner_sdk.tools.proxy import DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS


def test_default_tool_proxy_timeout_remains_120_seconds() -> None:
    assert DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS == pytest.approx(120.0)
