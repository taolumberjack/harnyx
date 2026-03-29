from __future__ import annotations

from pathlib import Path

from harnyx_commons.sandbox.agent_staging import stage_agent_source


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_stage_agent_source_normalizes_bind_mount_permissions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    data = b"print('hello')\n"

    artifact = stage_agent_source(
        state_dir=state_dir,
        container_root="/workspace/.harnyx_state",
        namespace="local_eval_agents",
        key="artifact-1",
        data=data,
    )

    checksum_path = artifact.host_path.parent / "agent.sha256"

    state_dir.chmod(0o700)
    artifact.host_path.parent.parent.chmod(0o700)
    artifact.host_path.parent.chmod(0o700)
    artifact.host_path.chmod(0o600)
    checksum_path.chmod(0o600)

    reused = stage_agent_source(
        state_dir=state_dir,
        container_root="/workspace/.harnyx_state",
        namespace="local_eval_agents",
        key="artifact-1",
        data=data,
    )

    assert reused == artifact
    assert _mode(state_dir) == 0o755
    assert _mode(artifact.host_path.parent.parent) == 0o755
    assert _mode(artifact.host_path.parent) == 0o755
    assert _mode(artifact.host_path) == 0o644
    assert _mode(checksum_path) == 0o644
