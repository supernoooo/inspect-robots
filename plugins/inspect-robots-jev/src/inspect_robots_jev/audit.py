"""Bounded, detached audit rows for the two Jev policies."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

MAX_RECORD_BYTES = 32_768
MAX_RECORDS = 48  # below the core transcript's two MiB limit
_DECISION_ID = re.compile(r"([0-9a-f]{32})-([0-9]{6,9})\Z")
_EPISODE_ID = re.compile(r"[0-9a-f]{32}\Z")


def _clean(value: Any, secret: str | None, depth: int = 0) -> Any:
    if depth > 8:
        return "[TRUNCATED]"
    if isinstance(value, str):
        value = value.replace(secret, "[REDACTED]") if secret else value
        return value[:256] + ("[TRUNCATED]" if len(value) > 256 else "")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_clean(item, secret, depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {str(key)[:128]: _clean(item, secret, depth + 1)
                for key, item in list(value.items())[:64]}
    return "[UNSERIALIZABLE]"


def bounded_record(row: dict[str, object], secret: str | None) -> dict[str, object]:
    """Keep all candidate decisions, trimming only bulky vision details first."""
    clean = _clean(row, secret)
    assert isinstance(clean, dict)
    vision = clean.get("vision")
    if isinstance(vision, dict):
        for entry in vision.values():
            if isinstance(entry, dict) and isinstance(entry.get("detections"), list):
                detections = entry["detections"]
                total = entry.get("detection_count")
                entry["detection_count"] = total if isinstance(total, int) else len(detections)
                entry["detections"] = detections[:8]
                entry["detections_omitted"] = entry["detection_count"] - len(entry["detections"])
    while len(json.dumps(clean, allow_nan=False).encode("utf-8")) > MAX_RECORD_BYTES:
        # Every remaining field is small and controlled except detection detail.
        if not isinstance(vision, dict) or not any(
            isinstance(v, dict) and v.get("detections") for v in vision.values()
        ):
            clean = {key: clean.get(key) for key in (
                "decision_id", "branch", "call", "reason", "selected_id", "trajectory",
                "candidates", "filtered", "probabilities", "observation", "execution",
            )}
            clean["audit_truncated"] = True
            if len(json.dumps(clean, allow_nan=False).encode("utf-8")) > MAX_RECORD_BYTES:
                # Candidate IDs and reasons are bounded by the planner; this is
                # only possible for a pathological future producer.
                clean["candidates"] = []
                clean["filtered"] = {"audit": "oversized"}
            break
        for entry in vision.values():
            if isinstance(entry, dict) and entry.get("detections"):
                entry["detections"].pop()
                entry["detections_omitted"] += 1
    return clean


def append_record(rows: list[dict[str, object]], row: dict[str, object], secret: str | None) -> int:
    rows.append(bounded_record(row, secret))
    if len(rows) > MAX_RECORDS:
        rows.pop(0)
        return 1
    return 0


def decision_file(decision_id: str) -> str:
    """Return a relative file name derived only from a validated decision ID."""
    match = _DECISION_ID.fullmatch(decision_id)
    if match is None:
        raise ValueError("invalid decision ID")
    return f"jev-audit/{match.group(1)}/{decision_id}.json"


def save_decision(log_dir: str | Path, row: dict[str, object]) -> str:
    """Atomically persist one bounded JSON decision under the trial run directory."""
    decision_id = row.get("decision_id")
    if not isinstance(decision_id, str):
        raise ValueError("missing decision ID")
    relative = decision_file(decision_id)
    payload = json.dumps(row, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("audit decision exceeds record limit")
    path = Path(log_dir) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return relative


def load_decision(log_dir: str | Path, decision_id: str) -> dict[str, object]:
    """Read and verify a decision by ID without accepting arbitrary paths."""
    path = Path(log_dir) / decision_file(decision_id)
    with path.open("rb") as stream:
        payload = stream.read(MAX_RECORD_BYTES + 1)
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("audit decision exceeds record limit")
    row = json.loads(payload)
    if not isinstance(row, dict) or row.get("decision_id") != decision_id:
        raise ValueError("audit decision ID mismatch")
    return row


def load_episode(log_dir: str | Path, episode_id: str, count: int) -> list[dict[str, object]]:
    """Retrieve every recorded decision using the bounded trial index."""
    if _EPISODE_ID.fullmatch(episode_id) is None or type(count) is not int or not 0 <= count <= 1_000_000:
        raise ValueError("invalid audit index")
    return [load_decision(log_dir, f"{episode_id}-{index:06d}")
            for index in range(1, count + 1)]
