from __future__ import annotations

import hashlib
from pathlib import Path

MAX_AGENT_BYTES = 256_000


def require_existing_agent_path(raw_path: str, *, label: str = "agent path") -> Path:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def load_agent_bytes(agent_path: Path) -> bytes:
    content = agent_path.read_bytes()
    return validate_agent_bytes(content, label=str(agent_path))


def validate_agent_bytes(agent_bytes: bytes, *, label: str = "agent script") -> bytes:
    if len(agent_bytes) > MAX_AGENT_BYTES:
        raise ValueError(f"{label} exceeds {MAX_AGENT_BYTES} bytes ({len(agent_bytes)} bytes)")
    try:
        source = agent_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 encoded") from exc
    try:
        compile(source, label, "exec")
    except SyntaxError as exc:
        raise ValueError(f"{label} failed bytecode compilation") from exc
    return agent_bytes


def agent_sha256(agent_bytes: bytes) -> str:
    return hashlib.sha256(agent_bytes).hexdigest()


__all__ = [
    "MAX_AGENT_BYTES",
    "agent_sha256",
    "load_agent_bytes",
    "require_existing_agent_path",
    "validate_agent_bytes",
]
