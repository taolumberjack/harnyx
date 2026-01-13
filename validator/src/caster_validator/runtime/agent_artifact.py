"""Agent artifact resolution and validation utilities (platform-provided only)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from caster_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from caster_validator.application.ports.platform import PlatformPort

logger = logging.getLogger("caster_validator.agent_artifact")

MAX_AGENT_BYTES = 256_000
_LOG_SNIPPET_LIMIT = 512


@dataclass(frozen=True, slots=True)
class AgentArtifact:
    """Represents a resolved agent script artifact."""

    content_hash: str
    host_path: Path
    container_path: str


def resolve_platform_agent_specs(
    *,
    batch_id: UUID,
    candidates: tuple[ScriptArtifactSpec, ...],
    platform_client: PlatformPort,
    state_dir: Path,
    container_root: str,
) -> dict[UUID, AgentArtifact]:
    """Resolve agent artifacts from platform-provided specs."""

    cache_root = state_dir / "platform_agents"
    cache_root.mkdir(parents=True, exist_ok=True)
    specs: dict[UUID, AgentArtifact] = {}
    artifact_ids = [candidate.artifact_id for candidate in candidates]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("batch candidates must be unique by artifact_id")

    for spec in candidates:
        try:
            data = platform_client.fetch_artifact(batch_id, spec.artifact_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch platform agent",
                extra={"batch_id": str(batch_id), "uid": spec.uid, "artifact_id": str(spec.artifact_id)},
                exc_info=exc,
            )
            continue
        if len(data) > MAX_AGENT_BYTES:
            logger.error(
                "Platform agent exceeds size limit",
                extra={
                    "batch_id": str(batch_id),
                    "uid": spec.uid,
                    "artifact_id": str(spec.artifact_id),
                    "size_bytes": len(data),
                },
            )
            continue
        content_hash = hashlib.sha256(data).hexdigest()
        if content_hash != spec.content_hash:
            logger.error(
                "Platform agent sha256 mismatch",
                extra={
                    "batch_id": str(batch_id),
                    "uid": spec.uid,
                    "artifact_id": str(spec.artifact_id),
                    "expected_sha256": spec.content_hash,
                    "actual_sha256": content_hash,
                },
            )
            continue
        artifact = _stage_platform_agent(
            cache_root=cache_root,
            content_hash=content_hash,
            data=data,
            state_dir=state_dir,
            container_root=container_root,
        )
        specs[spec.artifact_id] = artifact
        logger.info(
            "Staged platform agent",
            extra={
                "uid": spec.uid,
                "artifact_id": str(spec.artifact_id),
                "content_hash": content_hash,
                "host_path": str(artifact.host_path),
                "container_path": artifact.container_path,
            },
        )
    return specs


def _stage_platform_agent(
    *,
    cache_root: Path,
    content_hash: str,
    data: bytes,
    state_dir: Path,
    container_root: str,
) -> AgentArtifact:
    agent_dir = cache_root / content_hash
    agent_path = agent_dir / "agent.py"
    if agent_path.exists():
        container_path = _container_path_for(
            staged_path=agent_path,
            state_dir=state_dir,
            container_root=container_root,
        )
        return AgentArtifact(content_hash=content_hash, host_path=agent_path, container_path=container_path)

    agent_dir.mkdir(parents=True, exist_ok=True)
    temp_path = agent_dir / "agent.py.tmp"
    temp_path.write_bytes(data)
    try:
        _validate_agent_source(temp_path)
        temp_path.replace(agent_path)
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise
    (agent_dir / "agent.sha256").write_text(content_hash, encoding="utf-8")
    container_path = _container_path_for(
        staged_path=agent_path,
        state_dir=state_dir,
        container_root=container_root,
    )
    return AgentArtifact(content_hash=content_hash, host_path=agent_path, container_path=container_path)


def _container_path_for(*, staged_path: Path, state_dir: Path, container_root: str) -> str:
    rel = staged_path.relative_to(state_dir)
    rel_path = PurePosixPath("/".join(rel.parts))
    root = PurePosixPath(container_root or "/")
    combined = root / rel_path
    return combined.as_posix()


def _validate_agent_source(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("platform agent must be UTF-8 encoded") from exc
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        raise ValueError("platform agent failed bytecode compilation") from exc


def create_platform_agent_resolver(
    platform_client: PlatformPort,
) -> Callable[[UUID, MinerTaskBatchSpec, Path, str], dict[UUID, AgentArtifact]]:
    """Create a resolver function for platform agent artifacts."""

    def resolver(
        batch_id: UUID,
        batch: MinerTaskBatchSpec,
        state_dir: Path,
        container_root: str,
    ) -> dict[UUID, AgentArtifact]:
        return resolve_platform_agent_specs(
            batch_id=batch_id,
            candidates=batch.candidates,
            platform_client=platform_client,
            state_dir=state_dir,
            container_root=container_root,
        )

    return resolver


__all__ = [
    "AgentArtifact",
    "MAX_AGENT_BYTES",
    "create_platform_agent_resolver",
    "resolve_platform_agent_specs",
]
