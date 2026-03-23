from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.ports.session_registry import SessionRegistryPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.session import Session, SessionFailureCode, SessionStatus
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry


class FakeSessionRegistry(SessionRegistryPort):
    def __init__(self) -> None:
        self._sessions: dict[UUID, Session] = {}

    def create(self, session: Session) -> None:  # pragma: no cover - trivial
        self._sessions[session.session_id] = session

    def get(self, session_id: UUID) -> Session | None:
        return self._sessions.get(session_id)

    def update(self, session: Session) -> None:  # pragma: no cover - trivial
        self._sessions[session.session_id] = session

    def delete(self, session_id: UUID) -> None:  # pragma: no cover - trivial
        self._sessions.pop(session_id, None)

    def values(self) -> Iterable[Session]:  # pragma: no cover - helper
        return self._sessions.values()


def make_request(token: str | None = None, *, budget_usd: float = 0.1) -> SessionTokenRequest:
    issued_at = datetime(2025, 10, 17, 12, tzinfo=UTC)
    expires_at = issued_at + timedelta(hours=1)
    return SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=issued_at,
        expires_at=expires_at,
        budget_usd=budget_usd,
        token=token or uuid4().hex,
    )


def test_issue_persists_session_and_token() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)

    request = make_request()
    issued = manager.issue(request)

    stored = sessions.get(request.session_id)
    assert stored is not None
    assert issued.session == stored
    assert issued.session.budget_usd == pytest.approx(request.budget_usd)
    assert issued.session.effective_hard_limit_usd == pytest.approx(request.budget_usd)
    assert tokens.verify(request.session_id, request.token)

    envelope = manager.load(request.session_id)
    assert envelope is not None
    assert envelope.token_hash == issued.token_hash
    assert envelope.session == stored


def test_mark_status_updates_session() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)

    envelope = manager.mark_status(request.session_id, SessionStatus.COMPLETED)
    assert envelope.session.status is SessionStatus.COMPLETED


def test_missing_session_raises_on_mark_status() -> None:
    manager = SessionManager(FakeSessionRegistry(), InMemoryTokenRegistry())
    with pytest.raises(LookupError):
        manager.mark_status(uuid4(), SessionStatus.COMPLETED)


def test_issue_preserves_explicit_hard_limit() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=uuid4().hex,
        hard_limit_usd=1.0,
    )

    issued = manager.issue(request)

    assert issued.session.budget_usd == pytest.approx(0.5)
    assert issued.session.effective_hard_limit_usd == pytest.approx(1.0)


def test_begin_attempt_advances_counter_and_clears_failure_marker() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)
    manager.mark_failure_code(request.session_id, SessionFailureCode.TOOL_PROVIDER_FAILED)

    envelope = manager.begin_attempt(request.session_id)

    assert envelope.session.active_attempt == 1
    assert envelope.session.failure_code is None
    assert envelope.session.failure_attempt is None


def test_consume_failure_code_returns_current_attempt_marker_and_clears_it() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)
    manager.begin_attempt(request.session_id)
    manager.mark_failure_code(request.session_id, SessionFailureCode.TOOL_PROVIDER_FAILED)

    assert manager.consume_failure_code(request.session_id) is SessionFailureCode.TOOL_PROVIDER_FAILED

    stored = sessions.get(request.session_id)
    assert stored is not None
    assert stored.failure_code is None
    assert stored.failure_attempt is None


def test_consume_failure_code_discards_stale_attempt_marker() -> None:
    sessions = FakeSessionRegistry()
    tokens = InMemoryTokenRegistry()
    manager = SessionManager(sessions, tokens)
    request = make_request()
    manager.issue(request)
    manager.begin_attempt(request.session_id)
    stored = sessions.get(request.session_id)
    assert stored is not None
    sessions.update(
        replace(
            stored,
            active_attempt=2,
            failure_code=SessionFailureCode.TOOL_PROVIDER_FAILED,
            failure_attempt=1,
        )
    )

    assert manager.consume_failure_code(request.session_id) is None

    updated = sessions.get(request.session_id)
    assert updated is not None
    assert updated.failure_code is None
    assert updated.failure_attempt is None
