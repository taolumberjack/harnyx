from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import UUID, uuid4

import pytest

import harnyx_validator.application.scheduler as scheduler_module
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.invoke_entrypoint import SandboxInvocationError
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactExecutionFailedError,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def blocking_executor() -> ThreadPoolExecutor:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-validator-batch-blocking")
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class DummySandboxManager(SandboxManager):
    def __init__(self) -> None:
        self.starts: list[object | None] = []
        self.stops: list[SandboxDeployment] = []

    def start(self, options: object | None = None) -> SandboxDeployment:
        self.starts.append(options)
        return SandboxDeployment(client=object())

    def stop(self, deployment: SandboxDeployment) -> None:
        self.stops.append(deployment)


class DummyEvaluationRecordStore:
    def __init__(self) -> None:
        self.records_by_batch: list[MinerTaskRunSubmission] = []

    def record(self, result: MinerTaskRunSubmission) -> None:
        self.records_by_batch.append(result)


class DummyReceiptLog(ReceiptLogPort):
    def __init__(self) -> None:
        self._records: dict[str, object] = {}

    def record(self, receipt: object) -> None:
        self._records[str(len(self._records))] = receipt

    def lookup(self, receipt_id: str) -> object | None:
        return self._records.get(receipt_id)

    def values(self):
        return tuple(self._records.values())

    def for_session(self, session_id):
        return ()

    def clear_session(self, session_id) -> None:
        return None


class DummyProgressRecorder:
    def __init__(self, recorded: frozenset[tuple[UUID, UUID]] = frozenset()) -> None:
        self._recorded = set(recorded)

    def register(self, _batch) -> None:
        return None

    def record(self, result: MinerTaskRunSubmission) -> None:
        self._recorded.add((result.run.artifact_id, result.run.task_id))

    def recorded_pairs(self, _batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded)

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        return None

    def record_provider_call(self, *, session_id: UUID, provider: str, model: str) -> None:
        return None

    def record_provider_failure(self, *, session_id: UUID, provider: str, model: str) -> None:
        return None

    def consume_provider_failures(self, session_id: UUID) -> tuple[dict[str, object], ...]:
        return ()

    def clear_task_session(self, session_id: UUID) -> None:
        return None


def _task(text: str, *, budget_usd: float = 0.05) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
        budget_usd=budget_usd,
    )


def _sandbox_invocation_error(message: str) -> SandboxInvocationError:
    return SandboxInvocationError(
        message,
        status_code=0,
        detail_code=None,
        detail_exception="RuntimeError",
        detail_error=message,
    )


async def test_scheduler_runs_all_tasks_for_each_artifact(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    recorded_requests: list[tuple[int, MinerTask]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.task))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (
        ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),
        ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0),
    )
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(sandbox_manager.starts) == 2
    assert len(sandbox_manager.stops) == 2
    assert len(recorded_requests) == len(tasks) * 2
    assert len(result.runs) == len(recorded_requests)
    assert result.tasks == tasks
    assert len(evaluation_records.records_by_batch) == len(result.runs)


async def test_scheduler_avoids_asyncio_to_thread_for_blocking_work(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="answer one"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    async def _unexpected_to_thread(*args, **kwargs):
        raise AssertionError("scheduler should not use asyncio.to_thread")

    monkeypatch.setattr(scheduler_module.asyncio, "to_thread", _unexpected_to_thread)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 1
    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1


async def test_scheduler_cancellation_does_not_wait_for_blocking_lane_shutdown(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("shutdown")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    stop_started = Event()
    release_stop = Event()
    stop_finished = Event()

    class BlockingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            stop_started.set()
            try:
                release_stop.wait(timeout=5.0)
            finally:
                stop_finished.set()

    sandbox_manager = BlockingStopSandboxManager()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="answer shutdown"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    run_task = asyncio.create_task(
        scheduler.run(
            batch_id=uuid4(),
            requested_artifacts=(ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),),
        )
    )

    try:
        assert await asyncio.to_thread(stop_started.wait, 1.0) is True
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=0.2)
        assert release_stop.is_set() is False
    finally:
        release_stop.set()
        assert await asyncio.to_thread(stop_finished.wait, 1.0) is True


async def test_scheduler_records_zero_score_when_sandbox_invocation_errors(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("unstable")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    clock_values = iter(
        (
            datetime(2025, 10, 27, 0, 0, 0, tzinfo=UTC),
            datetime(2025, 10, 27, 0, 0, 2, tzinfo=UTC),
        ),
    )

    def orchestrator_factory(_client: object):
        class FailingOrchestrator:
            async def evaluate(self, request):
                raise _sandbox_invocation_error("upstream tool failure")

        return FailingOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: next(clock_values),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 1
    assert result.runs[0].score == 0.0
    assert result.runs[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="upstream tool failure",
    )
    assert evaluation_records.records_by_batch == list(result.runs)


async def test_scheduler_retries_only_transient_sandbox_invocation_errors(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    class FailingThenSuccessfulOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise _sandbox_invocation_error("upstream tool failure")
            details = EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    similarity_score=0.5,
                    total_score=0.75,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            )
            run = MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text=f"answer {request.task.query.text}"),
                details=details,
                completed_at=datetime(2025, 10, 27, tzinfo=UTC),
            )
            return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

    orchestrator = FailingThenSuccessfulOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 2
    assert len(evaluation_records.records_by_batch) == 2
    assert orchestrator.calls == 3
    assert result.runs[0].score == pytest.approx(0.75)
    assert result.runs[0].run.response == Response(text="answer first")
    assert result.runs[1].score == pytest.approx(0.75)
    assert result.runs[1].run.response == Response(text="answer second")


async def test_scheduler_fails_batch_for_generic_post_invoke_error(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    class GenericFailureThenSuccessOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("embedding client unavailable")
            details = EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    similarity_score=0.5,
                    total_score=0.75,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            )
            run = MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text=f"answer {request.task.query.text}"),
                details=details,
                completed_at=datetime(2025, 10, 27, tzinfo=UTC),
            )
            return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

    orchestrator = GenericFailureThenSuccessOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    with pytest.raises(ValidatorBatchFailedError, match="embedding client unavailable") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert orchestrator.calls == 1
    assert evaluation_records.records_by_batch == []


async def test_scheduler_retries_sandbox_start_once_before_running_tasks(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class FlakySandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            if len(self.starts) == 1:
                raise RuntimeError("sandbox cold start failed")
            return SandboxDeployment(client=object())

    sandbox_manager = FlakySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="answer startup"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(sandbox_manager.starts) == 2
    assert len(result.runs) == 1
    assert result.runs[0].score == pytest.approx(0.75)


async def test_scheduler_records_zero_score_for_terminal_setup_failures(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class AlwaysFailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox cold start failed")

    sandbox_manager = AlwaysFailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(sandbox_manager.starts) == 2
    assert len(result.runs) == 1
    assert result.runs[0].run.details.error == EvaluationError(
        code="sandbox_start_failed",
        message="sandbox cold start failed",
    )
    assert evaluation_records.records_by_batch == list(result.runs)


async def test_scheduler_fails_batch_after_three_sandbox_start_breakers(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class AlwaysFailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox cold start failed")

    sandbox_manager = AlwaysFailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )
    with pytest.raises(
        ValidatorBatchFailedError,
        match="validator artifact breaker tripped across 3 artifacts",
    ) as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "artifact_breaker_tripped"
    assert exc_info.value.failure_detail.error_code == "artifact_breaker_tripped"
    assert len(sandbox_manager.starts) == 6
    assert [submission.run.details.error for submission in evaluation_records.records_by_batch] == [
        EvaluationError(code="sandbox_start_failed", message="sandbox cold start failed"),
        EvaluationError(code="sandbox_start_failed", message="sandbox cold start failed"),
        EvaluationError(code="sandbox_start_failed", message="sandbox cold start failed"),
    ]


async def test_scheduler_fails_batch_after_three_artifact_fetch_breakers(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("fetch failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="artifact_fetch_failed",
                message="platform artifact fetch exhausted retries",
                exception_type="PlatformClientError",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )

    with pytest.raises(
        ValidatorBatchFailedError,
        match="validator artifact breaker tripped across 3 artifacts",
    ) as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert exc_info.value.error_code == "artifact_breaker_tripped"
    assert len(sandbox_manager.starts) == 0
    assert [submission.run.details.error for submission in evaluation_records.records_by_batch] == [
        EvaluationError(code="artifact_fetch_failed", message="platform artifact fetch exhausted retries"),
        EvaluationError(code="artifact_fetch_failed", message="platform artifact fetch exhausted retries"),
        EvaluationError(code="artifact_fetch_failed", message="platform artifact fetch exhausted retries"),
    ]


async def test_scheduler_keeps_three_hash_mismatch_failures_artifact_scoped(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("hash mismatch")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="artifact_hash_mismatch",
                message="platform agent sha256 mismatch",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)

    assert len(result.runs) == 3
    assert len(sandbox_manager.starts) == 0
    assert [submission.run.details.error for submission in result.runs] == [
        EvaluationError(code="artifact_hash_mismatch", message="platform agent sha256 mismatch"),
        EvaluationError(code="artifact_hash_mismatch", message="platform agent sha256 mismatch"),
        EvaluationError(code="artifact_hash_mismatch", message="platform agent sha256 mismatch"),
    ]


async def test_scheduler_fails_batch_after_three_artifact_breakers(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("artifact breaker")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    batch_id = uuid4()
    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0)
        for uid in (3, 5, 7)
    )

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    class ArtifactFailingRunner:
        def __init__(self) -> None:
            self.failed_artifact_ids: list[UUID] = []
            self.recorded_artifact_ids: list[UUID] = []

        async def evaluate_artifact(self, *, artifact: ScriptArtifactSpec, tasks, **_kwargs):
            self.failed_artifact_ids.append(artifact.artifact_id)
            raise ArtifactExecutionFailedError(
                error_code="sandbox_invocation_failed",
                message="shared sandbox failure",
                failure_detail=ValidatorBatchFailureDetail(
                    error_code="sandbox_invocation_failed",
                    error_message="shared sandbox failure",
                    occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                    artifact_id=artifact.artifact_id,
                    uid=artifact.uid,
                    exception_type="SandboxInvocationError",
                ),
                completed_submissions=(),
                remaining_tasks=tuple(tasks),
                artifact_breaker_tripped=True,
            )

        async def record_failure_for_artifact(
            self,
            *,
            artifact: ScriptArtifactSpec,
            tasks,
            error_code: str,
            error_message: str,
            **_kwargs,
        ) -> list[MinerTaskRunSubmission]:
            self.recorded_artifact_ids.append(artifact.artifact_id)
            assert tuple(tasks) == (task,)
            assert error_code == "sandbox_invocation_failed"
            assert error_message == "shared sandbox failure"
            return []

    failing_runner = ArtifactFailingRunner()
    scheduler._runner = failing_runner

    with pytest.raises(
        ValidatorBatchFailedError,
        match="validator artifact breaker tripped across 3 artifacts",
    ) as exc_info:
        await scheduler.run(batch_id=batch_id, requested_artifacts=artifacts)

    assert exc_info.value.error_code == "artifact_breaker_tripped"
    assert exc_info.value.failure_detail.error_code == "artifact_breaker_tripped"
    assert len(sandbox_manager.starts) == 3
    assert len(sandbox_manager.stops) == 3
    assert failing_runner.failed_artifact_ids == [artifact.artifact_id for artifact in artifacts]
    assert failing_runner.recorded_artifact_ids == [artifact.artifact_id for artifact in artifacts]


async def test_evaluation_runner_issues_session_with_task_budget(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    evaluation_records = DummyEvaluationRecordStore()
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda artifact: artifact,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
    )

    task = _task("budgeted", budget_usd=0.123)
    issued = scheduler._runner._issue_session(
        batch_id=uuid4(),
        uid=3,
        task=task,
    )

    assert issued.session.task_id == task.task_id
    assert issued.session.budget_usd == pytest.approx(0.123)


async def test_scheduler_runs_only_remaining_pairs(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    progress = DummyProgressRecorder(recorded=frozenset({(artifact.artifact_id, tasks[0].task_id)}))
    recorded_requests: list[tuple[int, MinerTask]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.task))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda current_artifact: {"uid": current_artifact.uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(artifact,))

    assert len(sandbox_manager.starts) == 1
    assert [(uid, task.task_id) for uid, task in recorded_requests] == [(artifact.uid, tasks[1].task_id)]
    assert len(result.runs) == 1
    assert result.runs[0].run.task_id == tasks[1].task_id


async def test_scheduler_skips_artifact_when_all_pairs_are_already_recorded(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    first_artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    progress = DummyProgressRecorder(
        recorded=frozenset((first_artifact.artifact_id, task.task_id) for task in tasks)
    )
    recorded_requests: list[tuple[int, UUID]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.artifact_id))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        similarity_score=0.5,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda current_artifact: {"uid": current_artifact.uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(first_artifact, second_artifact))

    assert len(sandbox_manager.starts) == 1
    assert all(uid == second_artifact.uid for uid, _artifact_id in recorded_requests)
    assert len(result.runs) == len(tasks)
