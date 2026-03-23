from __future__ import annotations

import time
from threading import Event, Thread

import bittensor as bt
import pytest

import harnyx_validator.infrastructure.auth.sr25519 as sr25519
from harnyx_commons.bittensor import VerificationError, build_canonical_request
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.auth.sr25519 import BittensorSr25519InboundVerifier
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.run_progress import InMemoryRunProgress
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings


def test_build_inbound_auth_uses_subnet_owner_hotkey(monkeypatch) -> None:
    expected_owner_coldkey_ss58 = "5DdemoOwnerColdkey"
    captured: dict[str, object] = {}

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            captured["network"] = network

        class _SubnetInfo:
            def __init__(self, owner_ss58: str) -> None:
                self.owner_ss58 = owner_ss58

        def get_subnet_info(self, netuid: int) -> _SubnetInfo:
            captured["netuid"] = netuid
            return self._SubnetInfo(expected_owner_coldkey_ss58)

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9945")
    monkeypatch.setenv("SUBTENSOR_NETUID", "2")
    monkeypatch.setattr(bootstrap.bt, "Subtensor", FakeSubtensor)

    settings = Settings()
    verifier = bootstrap._build_inbound_auth(settings)

    assert verifier.owner_coldkey_ss58 == expected_owner_coldkey_ss58
    assert captured["network"] == settings.subtensor.endpoint
    assert captured["netuid"] == settings.subtensor.netuid
    assert captured["closed"] is True


def test_build_inbound_auth_raises_when_owner_hotkey_missing(monkeypatch) -> None:
    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_subnet_info(self, netuid: int):
            return None

        def close(self) -> None:
            return None

    monkeypatch.setenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9945")
    monkeypatch.setenv("SUBTENSOR_NETUID", "2")
    monkeypatch.setattr(bootstrap.bt, "Subtensor", FakeSubtensor)

    settings = Settings()
    with pytest.raises(RuntimeError, match="subnet info"):
        bootstrap._build_inbound_auth(settings)


def test_inbound_verifier_rejects_non_owner_hotkey(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    canonical = build_canonical_request("GET", "/v1/test", b"")
    signature = keypair.sign(canonical)
    header = f'Bittensor ss58="{keypair.ss58_address}",sig="{signature.hex()}"'

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_hotkey_owner(self, hotkey_ss58: str):
            return "5NotOwnerColdkey"

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    with pytest.raises(VerificationError, match="subnet owner coldkey"):
        verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)


def test_inbound_verifier_caches_hotkey_owner_lookup(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    canonical = build_canonical_request("GET", "/v1/test", b"")
    signature = keypair.sign(canonical)
    header = f'Bittensor ss58="{keypair.ss58_address}",sig="{signature.hex()}"'

    calls: dict[str, int] = {"count": 0}

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_hotkey_owner(self, hotkey_ss58: str):
            calls["count"] += 1
            return "5OwnerColdkey"

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        owner_cache_ttl_seconds=9999.0,
    )
    verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
    verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)

    assert calls["count"] == 1


class _SlowFirstReadCache(dict[str, tuple[float, str]]):
    def __init__(self, *args, first_read_started: Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._first_read_started = first_read_started
        self._delayed = False

    def get(self, key: str, default=None):
        value = super().get(key, default)
        if value is not None and not self._delayed:
            self._delayed = True
            self._first_read_started.set()
            time.sleep(0.05)
        return value


def test_inbound_verifier_concurrent_expired_cache_entry_does_not_raise(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    canonical = build_canonical_request("GET", "/v1/test", b"")
    signature = keypair.sign(canonical)
    header = f'Bittensor ss58="{keypair.ss58_address}",sig="{signature.hex()}"'

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_hotkey_owner(self, hotkey_ss58: str):
            return "5OwnerColdkey"

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    first_read_started = Event()
    verifier._owner_cache = _SlowFirstReadCache(
        {keypair.ss58_address: (time.monotonic() - 1.0, "5OwnerColdkey")},
        first_read_started=first_read_started,
    )

    results: list[str] = []
    errors: list[Exception] = []

    def run_verify() -> None:
        try:
            results.append(
                verifier.verify(
                    method="GET",
                    path_qs="/v1/test",
                    body=b"",
                    authorization_header=header,
                )
            )
        except Exception as exc:
            errors.append(exc)

    first = Thread(target=run_verify)
    second = Thread(target=run_verify)
    first.start()
    assert first_read_started.wait(timeout=1.0) is True
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert results == [keypair.ss58_address, keypair.ss58_address]


@pytest.mark.anyio
async def test_make_control_provider_offloads_verify_request(monkeypatch: pytest.MonkeyPatch) -> None:
    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def _record_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
        to_thread_calls.append((func, args, kwargs))
        return "caller"

    monkeypatch.setattr(bootstrap.asyncio, "to_thread", _record_to_thread)

    progress_tracker = InMemoryRunProgress()
    status_provider = StatusProvider()
    accept_batch = AcceptEvaluationBatch(
        inbox=InMemoryBatchInbox(),
        status=status_provider,
        progress=progress_tracker,
    )
    inbound_auth = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    deps = bootstrap._make_control_provider(
        accept_batch=accept_batch,
        status_provider=status_provider,
        inbound_auth=inbound_auth,
        progress_tracker=progress_tracker,
    )()

    caller = await deps.auth(
        "GET",
        "/validator/status?verbose=1",
        b"",
        'Bittensor ss58="5demo",sig="00"',
    )

    assert caller == "caller"
    assert len(to_thread_calls) == 1
    called_func, called_args, called_kwargs = to_thread_calls[0]
    assert called_func is bootstrap._verify_request
    assert called_args == (inbound_auth,)
    assert called_kwargs == {
        "method": "GET",
        "path_qs": "/validator/status?verbose=1",
        "body": b"",
        "authorization_header": 'Bittensor ss58="5demo",sig="00"',
    }
