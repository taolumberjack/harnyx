"""Bittensor-backed Subtensor client used by the validator runtime."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import bittensor as bt
from bittensor.core.errors import MetadataError
from bittensor.core.extrinsics.set_weights import set_weights_extrinsic
from bittensor.core.settings import version_as_int
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids
from bittensor_drand import get_encrypted_commit, get_encrypted_commitment
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
    WeightSubmissionCadence,
    WeightSubmissionCadenceStatus,
)
from harnyx_validator.infrastructure.transient_network import classify_transient_network_failure

from .hotkey import create_wallet

logger = logging.getLogger("harnyx_validator.subtensor")

_COMMIT_REVEAL_MAX_RETRIES = 5
_PLAIN_SET_WEIGHTS_MAX_RETRIES = 5
_COMMIT_REVEAL_VERSION = 4
_DEFAULT_BLOCK_TIME_SECONDS = 12.0
_NO_WEIGHT_ATTEMPT_MESSAGE = "No attempt made. Perhaps it is too soon to commit weights!"
_LastUpdateValue: TypeAlias = int | None
_LastUpdateValues: TypeAlias = dict[int, _LastUpdateValue] | list[_LastUpdateValue]


class _SubtensorWeightTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: int
    weight: int

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, value: object) -> object:
        if isinstance(value, tuple) and len(value) == 2:
            uid, weight = value
            return {"uid": uid, "weight": weight}
        return value


class _SubtensorWeightRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uid: int
    targets: list[_SubtensorWeightTarget]

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, value: object) -> object:
        if isinstance(value, tuple) and len(value) == 2:
            source_uid, targets = value
            return {"source_uid": source_uid, "targets": targets}
        return value


class _SubtensorWeightsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[_SubtensorWeightRow]


@dataclass(slots=True)
class BittensorSubtensorClient(SubtensorClientPort):
    """Synchronous wrapper around ``bt.Subtensor``."""

    settings: SubtensorSettings

    def __post_init__(self) -> None:
        self._subtensor: bt.Subtensor | None = None
        self._wallet: bt.Wallet | None = None

    # ------------------------------------------------------------------
    # lifecycle helpers

    def connect(self) -> None:
        self._ensure_ready()

    def close(self) -> None:
        subtensor = self._subtensor
        if subtensor is None:
            return
        try:
            subtensor.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("subtensor close failed", exc_info=exc)
        finally:
            self._subtensor = None

    def _ensure_ready(self) -> None:
        if self._subtensor is None:
            self._subtensor = self._create_subtensor()
        if self._wallet is None:
            self._wallet = create_wallet(self.settings)

    def _create_subtensor(self) -> bt.Subtensor:
        endpoint = self.settings.endpoint.strip()
        network_or_endpoint = endpoint or self.settings.network
        return bt.Subtensor(network=network_or_endpoint)

    def _require_subtensor(self) -> bt.Subtensor:
        if self._subtensor is None:
            raise RuntimeError("subtensor client not initialized")
        return self._subtensor

    def _require_wallet(self) -> bt.Wallet:
        if self._wallet is None:
            raise RuntimeError("wallet not initialized")
        return self._wallet

    # ------------------------------------------------------------------
    # port implementation

    def fetch_metagraph(self) -> MetagraphSnapshot:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        metagraph = subtensor.metagraph(self.settings.netuid)
        uids = tuple(int(uid) for uid in metagraph.uids)
        hotkeys = tuple(metagraph.hotkeys)
        return MetagraphSnapshot(uids=uids, hotkeys=hotkeys)

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        commitment = subtensor.get_revealed_commitment(
            netuid=self.settings.netuid,
            uid=uid,
        )
        if not commitment:
            return None
        latest_block, data = max((int(entry[0]), entry[1]) for entry in commitment)
        return CommitmentRecord(block=latest_block, data=str(data))

    def publish_commitment(
        self,
        data: str,
        *,
        blocks_until_reveal: int = 1,
    ) -> CommitmentRecord:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        success, message = self._publish_commitment_extrinsic(
            subtensor=subtensor,
            data=data,
            blocks_until_reveal=max(1, blocks_until_reveal),
        )
        if not success:
            raise MetadataError(message)
        block_number = self._read_block_number()
        return CommitmentRecord(block=block_number, data=data)

    def current_block(self) -> int:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        return int(subtensor.get_current_block())

    def last_update_block(self, uid: int) -> int | None:
        if uid < 0:
            return None
        self._ensure_ready()
        subtensor = self._require_subtensor()
        last_update = self._query_last_update_values(
            subtensor=subtensor,
            netuid=self.settings.netuid,
        )
        if last_update is None:
            return None
        return self._last_update_for_uid(last_update, uid)

    def weight_submission_cadence(self, netuid: int) -> WeightSubmissionCadence:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        wallet = self._require_wallet()
        return self._read_weight_submission_cadence(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
        )

    def validator_info(self) -> ValidatorNodeInfo:
        snapshot = self.fetch_metagraph()
        wallet = self._require_wallet()
        hotkey = wallet.hotkey
        if hotkey is None:
            raise RuntimeError("wallet hotkey is unavailable")
        hotkey_addr = hotkey.ss58_address
        uid = -1
        if hotkey_addr and hotkey_addr in snapshot.hotkeys:
            uid = snapshot.hotkeys.index(hotkey_addr)
        version_key = self._query_version_key()
        return ValidatorNodeInfo(uid=uid, version_key=version_key)

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        if not weights:
            raise ValueError("weights mapping must not be empty")
        self._ensure_ready()
        subtensor = self._require_subtensor()
        wallet = self._require_wallet()
        uids, normalized = self._normalize_weights(weights)
        cadence = self._read_weight_submission_cadence(
            subtensor=subtensor,
            wallet=wallet,
            netuid=self.settings.netuid,
        )
        logger.debug(
            "submitting weights to subtensor",
            extra={"uids": uids, "wait_for_inclusion": self.settings.wait_for_inclusion},
        )
        if cadence.commit_reveal_enabled:
            success, message = self._submit_commit_reveal_weights(
                subtensor=subtensor,
                wallet=wallet,
                uids=uids,
                normalized=normalized,
                cadence=cadence,
            )
        else:
            success, message = self._submit_plain_weights(
                subtensor=subtensor,
                wallet=wallet,
                uids=uids,
                normalized=normalized,
                cadence=cadence,
            )
        logger.debug(
            "subtensor.set_weights returned",
            extra={"success": success, "message": message},
        )
        if not success:
            raise RuntimeError(f"set_weights failed: {message}")
        return str(message) if message is not None else ""

    def _normalize_weights(self, weights: Mapping[int, float]) -> tuple[list[int], list[float]]:
        ordered = sorted(weights.items(), key=lambda item: item[0])
        uids = [int(uid) for uid, _ in ordered]
        values = [float(score) for _, score in ordered]
        total = sum(values)
        if math.isclose(total, 0.0, abs_tol=1e-9):
            raise ValueError("weight totals must be greater than zero")
        normalized = [value / total for value in values]
        return uids, normalized

    def fetch_weight(self, uid: int) -> float:
        if uid < 0:
            return 0.0
        self._ensure_ready()
        validator_uid = self.validator_info().uid
        if validator_uid < 0:
            return 0.0
        subtensor = self._require_subtensor()
        raw_weights = subtensor.weights(netuid=self.settings.netuid)
        try:
            payload = _SubtensorWeightsPayload.model_validate(
                {"rows": raw_weights},
                strict=True,
            )
        except ValidationError as exc:
            raise RuntimeError(f"invalid subtensor weights payload: {exc}") from exc

        row = next((candidate for candidate in payload.rows if candidate.source_uid == validator_uid), None)
        if row is None:
            return 0.0

        target = next((candidate for candidate in row.targets if candidate.uid == uid), None)
        if target is None:
            return 0.0
        return float(target.weight)

    def tempo(self, netuid: int) -> int:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        tempo = subtensor.tempo(netuid=netuid)
        if tempo is None:
            raise RuntimeError("tempo is unavailable")
        return int(tempo)

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        self._ensure_ready()
        subtensor = self._require_subtensor()
        next_block = subtensor.get_next_epoch_start_block(netuid, block=reference_block)
        if next_block is None:
            raise RuntimeError("unable to determine next epoch start block")
        return int(next_block)

    # ------------------------------------------------------------------
    # helpers

    def _submit_hotkey_extrinsic(
        self,
        *,
        subtensor: bt.Subtensor,
        call: Any,
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ) -> tuple[bool, str]:
        wallet = self._require_wallet()
        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            sign_with="hotkey",
            use_nonce=True,
            nonce_key="hotkey",
            period=self.settings.transaction_period,
        )

    def _publish_commitment_extrinsic(
        self,
        *,
        subtensor: bt.Subtensor,
        data: str,
        blocks_until_reveal: int,
    ) -> tuple[bool, str]:
        encrypted, reveal_round = get_encrypted_commitment(
            data,
            blocks_until_reveal=blocks_until_reveal,
            block_time=_DEFAULT_BLOCK_TIME_SECONDS,
        )
        call = subtensor.substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={
                "netuid": self.settings.netuid,
                "info": {
                    "fields": [[{"TimelockEncrypted": {"encrypted": encrypted, "reveal_round": reveal_round}}]]
                },
            },
        )
        return self._submit_hotkey_extrinsic(
            subtensor=subtensor,
            call=call,
            wait_for_inclusion=False,
            wait_for_finalization=True,
        )

    def _submit_commit_reveal_weights(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
        cadence: WeightSubmissionCadence,
    ) -> tuple[bool, str]:
        if not cadence.can_submit:
            return False, self._cadence_refusal_message(cadence, wallet)

        success = False
        message = _NO_WEIGHT_ATTEMPT_MESSAGE
        transient_failure: Exception | None = None
        all_failures_transient = True
        for _ in range(_COMMIT_REVEAL_MAX_RETRIES):
            attempt_failed_transient = False
            try:
                call, reveal_round = self._build_commit_reveal_call(
                    subtensor=subtensor,
                    wallet=wallet,
                    uids=uids,
                    normalized=normalized,
                )
                success, message = self._submit_hotkey_extrinsic(
                    subtensor=subtensor,
                    call=call,
                    wait_for_inclusion=self.settings.wait_for_inclusion,
                    wait_for_finalization=self.settings.wait_for_finalization,
                )
            except Exception as exc:
                cause = classify_transient_network_failure(exc)
                if cause is None:
                    all_failures_transient = False
                    transient_failure = None
                else:
                    transient_failure = exc
                    attempt_failed_transient = True
                logger.warning("commit-reveal weight submission attempt failed", exc_info=exc)
                success = False
                message = str(exc)
            if success:
                return True, f"reveal_round:{reveal_round}"
            if not attempt_failed_transient:
                all_failures_transient = False
                transient_failure = None
        if all_failures_transient and transient_failure is not None:
            raise RuntimeError("commit-reveal weight submission attempts failed") from transient_failure
        return False, message

    def _submit_plain_weights(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
        cadence: WeightSubmissionCadence,
    ) -> tuple[bool, str]:
        if not cadence.can_submit:
            return False, self._cadence_refusal_message(cadence, wallet)

        success = False
        message = _NO_WEIGHT_ATTEMPT_MESSAGE
        for _ in range(_PLAIN_SET_WEIGHTS_MAX_RETRIES):
            success, message = set_weights_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                netuid=self.settings.netuid,
                uids=uids,
                weights=normalized,
                version_key=version_as_int,
                wait_for_inclusion=self.settings.wait_for_inclusion,
                wait_for_finalization=self.settings.wait_for_finalization,
                period=self.settings.transaction_period,
            )
            if success:
                return True, message
            logger.warning(
                "plain weight submission attempt failed",
                extra={"set_weights_message": message},
            )
        return False, message

    def _build_commit_reveal_call(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        uids: list[int],
        normalized: list[float],
    ) -> tuple[Any, int]:
        weight_uids, weight_vals = convert_and_normalize_weights_and_uids(uids, normalized)
        current_block = int(subtensor.get_current_block())
        hyperparameters = subtensor.get_subnet_hyperparameters(
            self.settings.netuid,
            block=current_block,
        )
        tempo = getattr(hyperparameters, "tempo", None)
        commit_reveal_period = getattr(hyperparameters, "commit_reveal_period", None)
        if tempo is None or commit_reveal_period is None:
            raise RuntimeError("subnet hyperparameters unavailable")
        hotkey = wallet.hotkey
        hotkey_public_key = hotkey.public_key if hotkey is not None else None
        if hotkey_public_key is None:
            raise RuntimeError("wallet hotkey public key is unavailable")
        commit, reveal_round = get_encrypted_commit(
            uids=weight_uids,
            weights=weight_vals,
            version_key=version_as_int,
            tempo=int(tempo),
            current_block=current_block,
            netuid=self.settings.netuid,
            subnet_reveal_period_epochs=int(commit_reveal_period),
            block_time=_DEFAULT_BLOCK_TIME_SECONDS,
            hotkey=hotkey_public_key,
        )
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="commit_timelocked_weights",
            call_params={
                "netuid": self.settings.netuid,
                "commit": commit,
                "reveal_round": reveal_round,
                "commit_reveal_version": _COMMIT_REVEAL_VERSION,
            },
        )
        return call, reveal_round

    def _read_block_number(self) -> int:
        try:
            subtensor = self._require_subtensor()
            return int(subtensor.get_current_block())
        except Exception:  # pragma: no cover - informational fallback
            return -1

    def _read_weight_submission_cadence(
        self,
        *,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        netuid: int,
    ) -> WeightSubmissionCadence:
        hotkey = wallet.hotkey
        hotkey_addr = hotkey.ss58_address if hotkey is not None else ""
        validator_uid_raw = subtensor.get_uid_for_hotkey_on_subnet(hotkey_addr, netuid)
        validator_uid = self._normalize_validator_uid(validator_uid_raw)
        if validator_uid is None:
            return self._cadence(
                status=WeightSubmissionCadenceStatus.UNREGISTERED,
                validator_uid=None,
                commit_reveal_enabled=False,
                current_block=None,
                last_update_block=None,
                blocks_since_last_update=None,
                weights_rate_limit=None,
            )

        commit_reveal_enabled = self._query_commit_reveal_enabled(
            subtensor=subtensor,
            netuid=netuid,
        )
        current_block = self._query_current_block(subtensor=subtensor)
        weights_rate_limit = self._query_weights_rate_limit(subtensor=subtensor, netuid=netuid)
        last_update_values = self._query_last_update_values(subtensor=subtensor, netuid=netuid)
        if (
            commit_reveal_enabled is None
            or current_block is None
            or weights_rate_limit is None
            or last_update_values is None
        ):
            return self._cadence(
                status=WeightSubmissionCadenceStatus.METADATA_UNAVAILABLE,
                validator_uid=validator_uid,
                commit_reveal_enabled=False if commit_reveal_enabled is None else commit_reveal_enabled,
                current_block=current_block,
                last_update_block=None,
                blocks_since_last_update=None,
                weights_rate_limit=weights_rate_limit,
            )

        last_update_block = self._last_update_for_uid(last_update_values, validator_uid)
        if last_update_block is None:
            return self._cadence(
                status=WeightSubmissionCadenceStatus.OPEN,
                validator_uid=validator_uid,
                commit_reveal_enabled=commit_reveal_enabled,
                current_block=current_block,
                last_update_block=None,
                blocks_since_last_update=None,
                weights_rate_limit=weights_rate_limit,
            )

        blocks_since_last_update = current_block - last_update_block
        if commit_reveal_enabled:
            threshold_open = blocks_since_last_update > weights_rate_limit
        else:
            threshold_open = blocks_since_last_update >= weights_rate_limit
        status = (
            WeightSubmissionCadenceStatus.OPEN
            if threshold_open
            else WeightSubmissionCadenceStatus.RATE_LIMITED
        )
        return self._cadence(
            status=status,
            validator_uid=validator_uid,
            commit_reveal_enabled=commit_reveal_enabled,
            current_block=current_block,
            last_update_block=last_update_block,
            blocks_since_last_update=blocks_since_last_update,
            weights_rate_limit=weights_rate_limit,
        )

    @staticmethod
    def _normalize_validator_uid(value: int | None) -> int | None:
        if value is None or value < 0:
            return None
        return value

    @staticmethod
    def _cadence(
        *,
        status: WeightSubmissionCadenceStatus,
        validator_uid: int | None,
        commit_reveal_enabled: bool,
        current_block: int | None,
        last_update_block: int | None,
        blocks_since_last_update: int | None,
        weights_rate_limit: int | None,
    ) -> WeightSubmissionCadence:
        return WeightSubmissionCadence(
            status=status,
            validator_uid=validator_uid,
            commit_reveal_enabled=commit_reveal_enabled,
            current_block=current_block,
            last_update_block=last_update_block,
            blocks_since_last_update=blocks_since_last_update,
            weights_rate_limit=weights_rate_limit,
        )

    def _cadence_refusal_message(self, cadence: WeightSubmissionCadence, wallet: bt.Wallet) -> str:
        if cadence.status is WeightSubmissionCadenceStatus.UNREGISTERED:
            hotkey = wallet.hotkey
            hotkey_addr = hotkey.ss58_address if hotkey is not None else ""
            return f"Hotkey {hotkey_addr} not registered in subnet {self.settings.netuid}"
        return _NO_WEIGHT_ATTEMPT_MESSAGE

    @staticmethod
    def _query_commit_reveal_enabled(*, subtensor: bt.Subtensor, netuid: int) -> bool | None:
        try:
            return bool(subtensor.commit_reveal_enabled(netuid=netuid))
        except Exception as exc:
            logger.debug("unable to read commit-reveal flag", exc_info=exc)
            return None

    @staticmethod
    def _query_current_block(*, subtensor: bt.Subtensor) -> int | None:
        try:
            return int(subtensor.get_current_block())
        except Exception as exc:
            logger.debug("unable to read current block for weight cadence", exc_info=exc)
            return None

    @staticmethod
    def _query_weights_rate_limit(*, subtensor: bt.Subtensor, netuid: int) -> int | None:
        try:
            value = subtensor.weights_rate_limit(netuid)
        except Exception as exc:
            logger.debug("unable to read weight rate limit", exc_info=exc)
            return None
        return None if value is None else int(value)

    @staticmethod
    def _query_last_update_values(*, subtensor: bt.Subtensor, netuid: int) -> _LastUpdateValues | None:
        try:
            values = subtensor.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        except Exception as exc:
            logger.debug("unable to read LastUpdate metadata", exc_info=exc)
            return None
        return BittensorSubtensorClient._normalize_last_update_values(values)

    @staticmethod
    def _normalize_last_update_values(values: object) -> _LastUpdateValues | None:
        if isinstance(values, Mapping):
            normalized: dict[int, _LastUpdateValue] = {}
            for key, value in values.items():
                if not isinstance(key, int) or (value is not None and not isinstance(value, int)):
                    return None
                normalized[key] = value
            return normalized
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            return None
        normalized_values: list[_LastUpdateValue] = []
        for value in values:
            if value is not None and not isinstance(value, int):
                return None
            normalized_values.append(value)
        return normalized_values

    @staticmethod
    def _last_update_for_uid(values: _LastUpdateValues, uid: int) -> int | None:
        if uid < 0:
            return None
        try:
            if isinstance(values, dict):
                value = values.get(uid)
            else:
                if uid >= len(values):
                    return None
                value = values[uid]
        except IndexError:
            return None
        return value

    def _query_version_key(self) -> int | None:
        try:
            subtensor = self._require_subtensor()
            subtensor_any = cast(Any, subtensor)
            return int(subtensor_any.weights_version(self.settings.netuid))
        except Exception:  # pragma: no cover - optional metadata
            return None


__all__ = ["BittensorSubtensorClient"]
