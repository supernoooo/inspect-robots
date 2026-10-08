"""Real tiny RRD fixtures for evaluation-only last reward extraction."""

from __future__ import annotations

import importlib.util
import json
import zlib
from pathlib import Path

import pytest

rr = pytest.importorskip("rerun")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _log(tmp_path: Path, policy: str, *, scene_count: int = 2) -> Path:
    run_dir = tmp_path / policy
    run_dir.mkdir()
    recording = run_dir / "tiny.rrd"
    with rr.RecordingStream("batch08-test") as stream:
        stream.save(recording)
        for step, value in ((0, 0.5), (1, 0.0)):
            stream.set_time("step", sequence=step)
            stream.log("trial/isaacsim-2yam-put-everything-in-box-000/e0/reward",
                       rr.Scalars(value))

    samples = []
    for index in range(scene_count):
        scene_id = f"isaacsim-2yam-put-everything-in-box-{index:03d}"
        metadata = {}
        if policy != "molmoact2":
            episode_id = f"{index + 1:032x}"
            decision_id = f"{episode_id}-000001"
            audit_dir = run_dir / "jev-audit" / episode_id
            audit_dir.mkdir(parents=True)
            (audit_dir / f"{decision_id}.json").write_text(json.dumps({
                "decision_id": decision_id, "reason": "vision_error" if index else None,
                "selected_id": "hold" if index else "move", "latency_s": 0.2,
                "jev_latency_s": 0.1,
            }), encoding="utf-8")
            metadata["jev_audit"] = {"episode_id": episode_id, "decision_count": 1,
                                     "missing_files": 0, "write_failures": 0}
        samples.append({"scene_id": scene_id, "scene_metadata": {"episode_index": index},
                        "epochs": [{"success_at_end": 1.0}], "trial_metadata": [metadata]})
    path = run_dir / "eval.json"
    path.write_text(json.dumps({"eval": {"task": "isaacsim-2yam-put-everything-in-box",
                                              "policy": policy, "seed": 9, "max_steps": 400},
                                "samples": samples}), encoding="utf-8")
    return path


def test_last_reward_and_scene_seed_alignment(tmp_path: Path) -> None:
    logs = {policy: _log(tmp_path, policy) for policy in summary.POLICIES}
    rows = summary.align(logs, task_seed=7, eval_seed=9)
    assert len(rows) == 6
    assert [row["policy"] for row in rows[:3]] == list(summary.POLICIES)
    assert {row["trial_seed"] for row in rows[:3]} == {zlib.crc32(b"9:7:0") & 0xFFFFFFFF}
    assert {row["trial_seed"] for row in rows[3:]} == {zlib.crc32(b"9:8:0") & 0xFFFFFFFF}
    assert all(row["scene_seed"] == 7 for row in rows[:3])
    assert all(row["scene_seed"] == 8 for row in rows[3:])
    assert all(row["last_reward"] == 0.0 and row["last_reward_step"] == 1
               for row in rows[:3])
    assert all(row["last_reward"] is None and row["reward_status"] == "missing"
               and row["success"] == 1.0 for row in rows[3:])
    assert rows[4]["decisions"] == 1 and rows[4]["holds"] == 1


def test_rejects_unaligned_scene_sets(tmp_path: Path) -> None:
    logs = {policy: _log(tmp_path, policy, scene_count=1 if policy == "jev-direct" else 2)
            for policy in summary.POLICIES}
    with pytest.raises(ValueError, match="different scene/epoch/seed sets"):
        summary.align(logs, task_seed=7, eval_seed=9)
