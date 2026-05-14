"""In-memory receipt log implementation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Condition, Lock
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import (
    IN_FLIGHT_LLM_UNKNOWN_EVIDENCE,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.tools.types import ToolName


@dataclass(frozen=True, slots=True)
class _PendingReceipt:
    receipt_id: str
    session_id: UUID
    session_active_attempt: int
    uid: int
    tool: ToolName
    issued_at: datetime
    details: ToolCallDetails


class InMemoryReceiptLog(ReceiptLogPort):
    """Stores tool call receipts in-memory for the lifetime of a session."""

    def __init__(self) -> None:
        self._receipts: dict[str, ToolCall] = {}
        self._session_index: defaultdict[UUID, set[str]] = defaultdict(set)
        self._pending: dict[str, _PendingReceipt] = {}
        self._pending_by_session: defaultdict[UUID, set[str]] = defaultdict(set)
        self._closed_windows: set[tuple[UUID, int, ToolName]] = set()
        self._lock = Lock()
        self._condition = Condition(self._lock)

    def record(self, receipt: ToolCall) -> None:
        with self._condition:
            self._record_locked(receipt)
            self._condition.notify_all()

    def start_pending_receipt(
        self,
        *,
        receipt_id: str,
        session_id: UUID,
        session_active_attempt: int,
        uid: int,
        tool: ToolName,
        issued_at: datetime,
        details: ToolCallDetails,
    ) -> None:
        with self._condition:
            window = (session_id, session_active_attempt, tool)
            if window in self._closed_windows:
                raise RuntimeError(
                    "cannot start pending receipt after timeout review window closed"
                )
            if receipt_id in self._receipts or receipt_id in self._pending:
                raise RuntimeError(f"receipt {receipt_id} already exists")
            pending = _PendingReceipt(
                receipt_id=receipt_id,
                session_id=session_id,
                session_active_attempt=session_active_attempt,
                uid=uid,
                tool=tool,
                issued_at=issued_at,
                details=details,
            )
            self._pending[receipt_id] = pending
            self._pending_by_session[session_id].add(receipt_id)
            self._condition.notify_all()

    def complete_pending_receipt(
        self,
        receipt: ToolCall,
        settle_usage: Callable[[], tuple[Session, bool]],
    ) -> tuple[Session, bool] | None:
        with self._condition:
            pending = self._pending.get(receipt.receipt_id)
            if pending is None:
                return None
            if receipt.session_id != pending.session_id:
                raise RuntimeError("completed receipt session does not match pending receipt")
            if receipt.tool != pending.tool:
                raise RuntimeError("completed receipt tool does not match pending receipt")
            settlement = settle_usage()
            self._record_locked(receipt)
            self._remove_pending_locked(receipt.receipt_id)
            self._condition.notify_all()
            return settlement

    def abandon_pending_receipt(self, receipt_id: str) -> None:
        with self._condition:
            self._remove_pending_locked(receipt_id)
            self._condition.notify_all()

    def wait_and_materialize_unknown_receipts(
        self,
        session_id: UUID,
        *,
        session_active_attempt: int,
        tool: ToolName,
        timeout_seconds: float,
        clock: Callable[[], datetime],
    ) -> tuple[ToolCall, ...]:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._matching_pending_ids_locked(
                    session_id,
                    session_active_attempt=session_active_attempt,
                    tool=tool,
                ),
                timeout=timeout_seconds,
            )
            window = (session_id, session_active_attempt, tool)
            self._closed_windows.add(window)
            pending_ids = self._matching_pending_ids_locked(
                session_id,
                session_active_attempt=session_active_attempt,
                tool=tool,
            )
            materialized_at = clock()
            receipts = tuple(
                self._materialize_unknown_receipt_locked(
                    self._pending[receipt_id],
                    materialized_at=materialized_at,
                )
                for receipt_id in pending_ids
            )
            for receipt in receipts:
                self._record_locked(receipt)
                self._remove_pending_locked(receipt.receipt_id)
            self._condition.notify_all()
            return receipts

    def lookup(self, receipt_id: str) -> ToolCall | None:
        with self._lock:
            return self._receipts.get(receipt_id)

    def values(self) -> tuple[ToolCall, ...]:
        with self._lock:
            receipts = tuple(self._receipts.values())
        return receipts

    def for_session(self, session_id: UUID) -> tuple[ToolCall, ...]:
        with self._lock:
            receipt_ids = tuple(self._session_index.get(session_id, ()))
            receipts = tuple(self._receipts[receipt_id] for receipt_id in receipt_ids)
        return receipts

    def clear_session(self, session_id: UUID) -> None:
        with self._condition:
            receipt_ids = self._session_index.pop(session_id, set())
            for receipt_id in receipt_ids:
                self._receipts.pop(receipt_id, None)
            pending_ids = self._pending_by_session.pop(session_id, set())
            for receipt_id in pending_ids:
                self._pending.pop(receipt_id, None)
            self._closed_windows = {
                window for window in self._closed_windows if window[0] != session_id
            }
            self._condition.notify_all()

    def _record_locked(self, receipt: ToolCall) -> None:
        self._receipts[receipt.receipt_id] = receipt
        self._session_index[receipt.session_id].add(receipt.receipt_id)

    def _remove_pending_locked(self, receipt_id: str) -> None:
        pending = self._pending.pop(receipt_id, None)
        if pending is None:
            return
        receipt_ids = self._pending_by_session.get(pending.session_id)
        if receipt_ids is None:
            return
        receipt_ids.discard(receipt_id)
        if not receipt_ids:
            self._pending_by_session.pop(pending.session_id, None)

    def _matching_pending_ids_locked(
        self,
        session_id: UUID,
        *,
        session_active_attempt: int,
        tool: ToolName,
    ) -> tuple[str, ...]:
        receipt_ids = self._pending_by_session.get(session_id, set())
        return tuple(
            receipt_id
            for receipt_id in receipt_ids
            if self._pending[receipt_id].session_active_attempt == session_active_attempt
            and self._pending[receipt_id].tool == tool
        )

    def _materialize_unknown_receipt_locked(
        self,
        pending: _PendingReceipt,
        *,
        materialized_at: datetime,
    ) -> ToolCall:
        extra = dict(pending.details.extra or {})
        extra["timeout_attribution_evidence"] = IN_FLIGHT_LLM_UNKNOWN_EVIDENCE
        extra["session_active_attempt"] = str(pending.session_active_attempt)
        execution = _materialized_execution(pending.details.execution, materialized_at)
        return ToolCall(
            receipt_id=pending.receipt_id,
            session_id=pending.session_id,
            uid=pending.uid,
            tool=pending.tool,
            issued_at=pending.issued_at,
            outcome=ToolCallOutcome.TIMEOUT,
            details=replace(
                pending.details,
                response_hash=None,
                response_payload=None,
                results=(),
                cost_usd=None,
                extra=extra,
                execution=execution,
            ),
        )


def _materialized_execution(
    execution: ToolExecutionFacts | None,
    materialized_at: datetime,
) -> ToolExecutionFacts:
    started_at = None if execution is None else execution.started_at
    elapsed_ms = None
    if started_at is not None:
        elapsed_ms = (materialized_at - started_at).total_seconds() * 1000.0
    return ToolExecutionFacts(
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        finished_at=materialized_at,
    )


__all__ = ["InMemoryReceiptLog"]
