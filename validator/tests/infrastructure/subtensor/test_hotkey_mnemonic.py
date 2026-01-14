from __future__ import annotations

import bittensor as bt
import pytest

from caster_validator.infrastructure.subtensor.hotkey import ensure_wallet_hotkey_from_mnemonic


def _make_wallet(tmp_path) -> bt.wallet.Wallet:
    return bt.wallet(name="validator", hotkey="default", path=str(tmp_path))


def test_ensure_wallet_hotkey_from_mnemonic_creates_hotkey_when_missing(tmp_path) -> None:
    mnemonic = bt.Keypair.generate_mnemonic()
    wallet = _make_wallet(tmp_path)

    assert wallet.hotkey_file.exists_on_device() is False

    ensure_wallet_hotkey_from_mnemonic(wallet, mnemonic)

    assert wallet.hotkey_file.exists_on_device() is True
    assert wallet.hotkey.ss58_address == bt.Keypair.create_from_mnemonic(mnemonic).ss58_address


def test_ensure_wallet_hotkey_from_mnemonic_is_idempotent_when_matches(tmp_path) -> None:
    mnemonic = bt.Keypair.generate_mnemonic()
    wallet = _make_wallet(tmp_path)

    ensure_wallet_hotkey_from_mnemonic(wallet, mnemonic)
    expected_ss58 = wallet.hotkey.ss58_address

    ensure_wallet_hotkey_from_mnemonic(wallet, mnemonic)

    assert wallet.hotkey.ss58_address == expected_ss58


def test_ensure_wallet_hotkey_from_mnemonic_raises_when_mismatched(tmp_path) -> None:
    mnemonic = bt.Keypair.generate_mnemonic()
    other_mnemonic = bt.Keypair.generate_mnemonic()
    wallet = _make_wallet(tmp_path)

    ensure_wallet_hotkey_from_mnemonic(wallet, mnemonic)

    with pytest.raises(RuntimeError, match="SUBTENSOR_HOTKEY_MNEMONIC"):
        ensure_wallet_hotkey_from_mnemonic(wallet, other_mnemonic)

