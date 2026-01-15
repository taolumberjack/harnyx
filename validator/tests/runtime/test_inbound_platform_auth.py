from __future__ import annotations

import pytest

from caster_validator.runtime import bootstrap
from caster_validator.runtime.settings import Settings


def test_build_inbound_auth_uses_subnet_owner_hotkey(monkeypatch) -> None:
    expected_ss58 = "5DdemoOwnerHotkey"
    captured: dict[str, object] = {}

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            captured["network"] = network

        def get_subnet_owner_hotkey(self, *, netuid: int) -> str:
            captured["netuid"] = netuid
            return expected_ss58

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9945")
    monkeypatch.setenv("SUBTENSOR_NETUID", "2")
    monkeypatch.setattr(bootstrap.bt, "Subtensor", FakeSubtensor)

    settings = Settings()
    verifier = bootstrap._build_inbound_auth(settings)

    assert verifier.allowed_ss58 == frozenset({expected_ss58})
    assert captured["network"] == settings.subtensor.endpoint
    assert captured["netuid"] == settings.subtensor.netuid
    assert captured["closed"] is True


def test_build_inbound_auth_raises_when_owner_hotkey_missing(monkeypatch) -> None:
    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_subnet_owner_hotkey(self, *, netuid: int) -> str:
            return ""

        def close(self) -> None:
            return None

    monkeypatch.setenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9945")
    monkeypatch.setenv("SUBTENSOR_NETUID", "2")
    monkeypatch.setattr(bootstrap.bt, "Subtensor", FakeSubtensor)

    settings = Settings()
    with pytest.raises(RuntimeError, match="subnet owner hotkey"):
        bootstrap._build_inbound_auth(settings)

