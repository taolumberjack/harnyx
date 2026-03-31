from __future__ import annotations

from pathlib import Path

from harnyx_validator.runtime import resource_usage
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageProvider


def test_resource_usage_snapshot_reads_process_memory_cpu_and_disk(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cpu_snapshots = iter(((12.0, 120.0), (15.0, 125.0)))
    monkeypatch.setattr(resource_usage, "_read_cpu_snapshot", lambda: next(cpu_snapshots))
    monkeypatch.setattr(resource_usage, "_read_process_rss_bytes", lambda: 512 * 1024 * 1024)
    monkeypatch.setattr(resource_usage, "_read_total_memory_bytes", lambda: 2 * 1024 * 1024 * 1024)
    monkeypatch.setattr(
        resource_usage,
        "_read_disk_usage",
        lambda path: (
            100 * 1024 * 1024 * 1024,
            25 * 1024 * 1024 * 1024,
            25.0,
        ),
    )

    provider = ValidatorResourceUsageProvider(disk_usage_path=tmp_path)

    snapshot = provider.snapshot()

    assert snapshot.cpu_percent == 60.0
    assert snapshot.memory_used_bytes == 512 * 1024 * 1024
    assert snapshot.memory_total_bytes == 2 * 1024 * 1024 * 1024
    assert snapshot.memory_percent == 25.0
    assert snapshot.disk_used_bytes == 25 * 1024 * 1024 * 1024
    assert snapshot.disk_total_bytes == 100 * 1024 * 1024 * 1024
    assert snapshot.disk_percent == 25.0
    assert snapshot.captured_at.tzinfo is not None


def test_read_total_memory_bytes_prefers_cgroup_v2_limit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text(str(2 * 1024 * 1024 * 1024), encoding="utf-8")
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_MEMORY_MAX_PATH", cgroup_v2)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_MEMORY_LIMIT_PATH", cgroup_v1)
    monkeypatch.setattr(resource_usage, "_read_host_total_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    assert resource_usage._read_total_memory_bytes() == 2 * 1024 * 1024 * 1024


def test_read_total_memory_bytes_prefers_cgroup_v1_limit_when_v2_is_unbounded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text("max", encoding="utf-8")
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_MEMORY_MAX_PATH", cgroup_v2)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_MEMORY_LIMIT_PATH", cgroup_v1)
    monkeypatch.setattr(resource_usage, "_read_host_total_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    assert resource_usage._read_total_memory_bytes() == 3 * 1024 * 1024 * 1024


def test_read_total_memory_bytes_falls_back_to_host_when_cgroup_limits_are_unbounded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text("max", encoding="utf-8")
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text(str(16 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_MEMORY_MAX_PATH", cgroup_v2)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_MEMORY_LIMIT_PATH", cgroup_v1)
    monkeypatch.setattr(resource_usage, "_read_host_total_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    assert resource_usage._read_total_memory_bytes() == 16 * 1024 * 1024 * 1024
