from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import (
    IN_FLIGHT_LLM_UNKNOWN_EVIDENCE,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog


def test_receipt_log_waits_until_pending_llm_receipt_finishes() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    receipt_id = "receipt-1"
    receipt_log.start_pending_receipt(
        receipt_id=receipt_id,
        session_id=session.session_id,
        session_active_attempt=1,
        uid=session.uid,
        tool="llm_chat",
        issued_at=_issued_at(),
        details=_details(active_attempt=1),
    )

    def complete() -> None:
        receipt_log.complete_pending_receipt(
            _ok_receipt(session, receipt_id=receipt_id, active_attempt=1),
            lambda: (session, False),
        )

    thread = threading.Thread(target=complete)
    thread.start()
    thread.join(timeout=1.0)

    assert (
        receipt_log.wait_and_materialize_unknown_receipts(
            session.session_id,
            session_active_attempt=1,
            tool="llm_chat",
            timeout_seconds=0.0,
            clock=_clock,
        )
        == ()
    )
    assert receipt_log.lookup(receipt_id) == _ok_receipt(
        session,
        receipt_id=receipt_id,
        active_attempt=1,
    )


def test_receipt_log_materializes_unknown_pending_receipt_as_timeout_tool_call() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    receipt_log.start_pending_receipt(
        receipt_id="receipt-1",
        session_id=session.session_id,
        session_active_attempt=2,
        uid=session.uid,
        tool="llm_chat",
        issued_at=_issued_at(),
        details=_details(active_attempt=2),
    )

    receipts = receipt_log.wait_and_materialize_unknown_receipts(
        session.session_id,
        session_active_attempt=2,
        tool="llm_chat",
        timeout_seconds=0.0,
        clock=_clock,
    )

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.outcome is ToolCallOutcome.TIMEOUT
    assert receipt.details.response_payload is None
    assert receipt.details.extra is not None
    assert receipt.details.extra["timeout_attribution_evidence"] == IN_FLIGHT_LLM_UNKNOWN_EVIDENCE
    assert receipt.details.extra["session_active_attempt"] == "2"
    assert tuple(receipt_log.for_session(session.session_id)) == receipts


def test_receipt_log_rejects_completion_after_unknown_materialization() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    receipt_log.start_pending_receipt(
        receipt_id="receipt-1",
        session_id=session.session_id,
        session_active_attempt=1,
        uid=session.uid,
        tool="llm_chat",
        issued_at=_issued_at(),
        details=_details(active_attempt=1),
    )
    receipt_log.wait_and_materialize_unknown_receipts(
        session.session_id,
        session_active_attempt=1,
        tool="llm_chat",
        timeout_seconds=0.0,
        clock=_clock,
    )

    completion = receipt_log.complete_pending_receipt(
        _ok_receipt(session, receipt_id="receipt-1", active_attempt=1),
        lambda: (session, False),
    )

    assert completion is None


def test_receipt_log_rejects_new_pending_receipt_after_timeout_review_window_closes() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    receipt_log.wait_and_materialize_unknown_receipts(
        session.session_id,
        session_active_attempt=1,
        tool="llm_chat",
        timeout_seconds=0.0,
        clock=_clock,
    )

    with pytest.raises(RuntimeError, match="timeout review window closed"):
        receipt_log.start_pending_receipt(
            receipt_id="receipt-1",
            session_id=session.session_id,
            session_active_attempt=1,
            uid=session.uid,
            tool="llm_chat",
            issued_at=_issued_at(),
            details=_details(active_attempt=1),
        )


def _session() -> Session:
    issued_at = _issued_at()
    return Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
        budget_usd=1.0,
    )


def _issued_at() -> datetime:
    return datetime(2026, 5, 14, 12, tzinfo=UTC)


def _clock() -> datetime:
    return _issued_at() + timedelta(minutes=5)


def _details(*, active_attempt: int) -> ToolCallDetails:
    return ToolCallDetails(
        request_hash="request-hash",
        request_payload={"args": [], "kwargs": {"model": "zai-org/GLM-5-TEE"}},
        extra={"session_active_attempt": str(active_attempt)},
        execution=ToolExecutionFacts(started_at=_issued_at()),
    )


def _ok_receipt(
    session: Session,
    *,
    receipt_id: str,
    active_attempt: int,
) -> ToolCall:
    return ToolCall(
        receipt_id=receipt_id,
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=_issued_at(),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="request-hash",
            request_payload={"args": [], "kwargs": {"model": "zai-org/GLM-5-TEE"}},
            response_hash="response-hash",
            response_payload={"usage": {"total_tokens": 10}},
            extra={"session_active_attempt": str(active_attempt)},
            execution=ToolExecutionFacts(elapsed_ms=1000.0, started_at=_issued_at()),
        ),
    )
