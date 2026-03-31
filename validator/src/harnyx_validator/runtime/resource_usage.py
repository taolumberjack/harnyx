"""Runtime resource-usage sampling for validator status snapshots."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_CGROUP_V2_MEMORY_MAX_PATH = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY_LIMIT_PATH = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")


@dataclass(frozen=True, slots=True)
class ValidatorResourceUsageSnapshot:
    captured_at: datetime
    cpu_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    memory_percent: float
    disk_used_bytes: int
    disk_total_bytes: int
    disk_percent: float


@dataclass(slots=True)
class ValidatorResourceUsageProvider:
    disk_usage_path: Path = field(default_factory=Path.cwd)
    _previous_cpu_seconds: float = field(init=False, repr=False)
    _previous_wall_seconds: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.disk_usage_path = self.disk_usage_path.expanduser()
        cpu_seconds, wall_seconds = _read_cpu_snapshot()
        self._previous_cpu_seconds = cpu_seconds
        self._previous_wall_seconds = wall_seconds

    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        cpu_seconds, wall_seconds = _read_cpu_snapshot()
        cpu_percent = _cpu_percent(
            current_cpu_seconds=cpu_seconds,
            current_wall_seconds=wall_seconds,
            previous_cpu_seconds=self._previous_cpu_seconds,
            previous_wall_seconds=self._previous_wall_seconds,
        )
        self._previous_cpu_seconds = cpu_seconds
        self._previous_wall_seconds = wall_seconds

        memory_used_bytes = _read_process_rss_bytes()
        memory_total_bytes = _read_total_memory_bytes()
        disk_total_bytes, disk_used_bytes, disk_percent = _read_disk_usage(self.disk_usage_path)
        return ValidatorResourceUsageSnapshot(
            captured_at=datetime.now(UTC),
            cpu_percent=cpu_percent,
            memory_used_bytes=memory_used_bytes,
            memory_total_bytes=memory_total_bytes,
            memory_percent=_usage_percent(memory_used_bytes, memory_total_bytes),
            disk_used_bytes=disk_used_bytes,
            disk_total_bytes=disk_total_bytes,
            disk_percent=disk_percent,
        )


def _read_cpu_snapshot() -> tuple[float, float]:
    return time.process_time(), time.monotonic()


def _cpu_percent(
    *,
    current_cpu_seconds: float,
    current_wall_seconds: float,
    previous_cpu_seconds: float,
    previous_wall_seconds: float,
) -> float:
    elapsed_wall_seconds = current_wall_seconds - previous_wall_seconds
    if elapsed_wall_seconds <= 0:
        return 0.0
    elapsed_cpu_seconds = current_cpu_seconds - previous_cpu_seconds
    if elapsed_cpu_seconds <= 0:
        return 0.0
    return (elapsed_cpu_seconds / elapsed_wall_seconds) * 100.0


def _read_process_rss_bytes() -> int:
    statm = Path("/proc/self/statm").read_text(encoding="utf-8").strip().split()
    if len(statm) < 2:
        raise RuntimeError("unexpected /proc/self/statm format")
    resident_pages = int(statm[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _read_total_memory_bytes() -> int:
    host_total_memory_bytes = _read_host_total_memory_bytes()
    for path in (_CGROUP_V2_MEMORY_MAX_PATH, _CGROUP_V1_MEMORY_LIMIT_PATH):
        limit = _read_finite_memory_limit(path, host_total_memory_bytes)
        if limit is not None:
            return limit
    return host_total_memory_bytes


def _read_finite_memory_limit(path: Path, host_total_memory_bytes: int) -> int | None:
    if not path.exists():
        return None
    raw_value = path.read_text(encoding="utf-8").strip()
    if raw_value == "max":
        return None
    limit = int(raw_value)
    if limit <= 0:
        return None
    if limit >= host_total_memory_bytes:
        return None
    return limit


def _read_host_total_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _read_disk_usage(path: Path) -> tuple[int, int, float]:
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, _usage_percent(usage.used, usage.total)


def _usage_percent(used_bytes: int, total_bytes: int) -> float:
    if total_bytes <= 0:
        return 0.0
    return (used_bytes / total_bytes) * 100.0


__all__ = ["ValidatorResourceUsageProvider", "ValidatorResourceUsageSnapshot"]
