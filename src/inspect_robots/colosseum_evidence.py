"""Update local scoring metadata without changing recorded robot evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


def refresh_outcome(path: Path, outcome: dict[str, Any]) -> None:
    """Rescore a verified local export, preserving its videos and action/state arrays.

    Review saves only the manifest. Submission calls this before the normal exporter,
    whose signature and file checks still run. Never call after publication begins.
    """
    from colosseum_client.evaluation import save

    dataset = path / "lerobot"
    marker_path = dataset / "meta/colosseum.json"
    if not dataset.exists():
        return
    marker = json.loads(marker_path.read_text())
    if marker.get("outcome") == outcome:
        return
    context = {key: value for key, value in marker.items() if key not in {"files", "source_sha256"}}
    raw = (path / "frames.jsonl").read_bytes()
    signature = hashlib.sha256(raw + json.dumps(context, sort_keys=True).encode()).hexdigest()
    if signature != marker["source_sha256"]:
        raise ValueError("Export provenance differs from recorded evidence; cannot rescore")
    for relative, digest in marker["files"].items():
        file = dataset / relative
        if file.is_symlink() or not file.resolve().is_relative_to(dataset.resolve()):
            raise ValueError("Export checksum path escapes the dataset")
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError("Export file is modified; cannot rescore")
    score_csv = dataset / "eval_trials.csv"
    if score_csv.exists():
        with score_csv.open(newline="") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            rows = list(reader)
        if len(rows) != 1 or rows[0]["run"] != path.name or fields is None:
            raise ValueError("Import score CSV does not match this run")
        rows[0]["score"] = str(outcome["score_0_4"])
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = score_csv.with_suffix(".tmp")
        temporary.write_text(output.getvalue())
        temporary.replace(score_csv)
        if "eval_trials.csv" in marker["files"]:
            marker["files"]["eval_trials.csv"] = hashlib.sha256(score_csv.read_bytes()).hexdigest()
    context["outcome"] = outcome
    marker.update(
        outcome=outcome,
        source_sha256=hashlib.sha256(
            raw + json.dumps(context, sort_keys=True).encode()
        ).hexdigest(),
    )
    save(marker_path, marker)
