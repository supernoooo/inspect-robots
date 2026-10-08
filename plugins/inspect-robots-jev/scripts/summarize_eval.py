"""Offline, evaluator-only alignment of YAM JSON scores and Rerun last rewards.

Requires rerun-sdk with experimental.RrdReader (tested with 0.37.2). It never
imports a policy or feeds evaluation truth into a policy decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import zlib
from pathlib import Path
from typing import Any


POLICIES = ("molmoact2", "jev-hybrid", "jev-direct")
FIELDS = ("policy", "scene_id", "epoch", "scene_seed", "trial_seed", "success",
          "last_reward", "reward_status", "last_reward_step", "decisions", "holds",
          "mean_policy_latency_s", "mean_molmo_latency_s", "mean_jev_latency_s",
          "mean_post_policy_interval_s", "run_mean_inference_latency_s")


def _scalar(value: Any) -> float:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict) and set(value) == {"value"}:
        value = value["value"]
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"unexpected reward scalar shape: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite reward")
    return result


def rewards_from_rrd(path: Path) -> dict[tuple[str, int], tuple[int, float]]:
    """Return the highest logged step for each trial, never a maximum reward."""
    import rerun as rr

    result: dict[tuple[str, int], tuple[int, float]] = {}
    for chunk in rr.experimental.RrdReader(path).stream():
        parts = chunk.entity_path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "trial" or parts[3] != "reward":
            continue
        if not parts[2].startswith("e") or not parts[2][1:].isdigit():
            raise ValueError(f"invalid trial reward path: {chunk.entity_path}")
        key = (parts[1], int(parts[2][1:]))
        batch = chunk.to_record_batch()
        step_col = batch.schema.get_field_index("step")
        reward_cols = [i for i, name in enumerate(batch.schema.names)
                       if name.endswith("Scalars:scalars")]
        if step_col < 0 or len(reward_cols) != 1:
            raise ValueError(f"missing step or scalar reward in {chunk.entity_path}")
        steps = batch.column(step_col).to_pylist()
        values = batch.column(reward_cols[0]).to_pylist()
        for step, value in zip(steps, values):
            if step is None or value is None:
                continue
            if type(step) is not int or step < 0:
                raise ValueError(f"invalid reward step: {step!r}")
            previous = result.get(key)
            if previous is not None and step == previous[0]:
                raise ValueError(f"duplicate last reward step for {key}")
            if previous is None or step > previous[0]:
                result[key] = step, _scalar(value)
    return result


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows
              if isinstance(row.get(field), (int, float)) and not isinstance(row[field], bool)
              and math.isfinite(float(row[field]))]
    return statistics.mean(values) if values else None


def _post_policy_interval(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        observation, execution = row.get("observation"), row.get("execution")
        latency = row.get("latency_s")
        if not isinstance(observation, dict) or not isinstance(execution, dict):
            continue
        start, end = observation.get("state_time"), execution.get("observed_state_time")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value)) for value in (start, end, latency)):
            interval = float(end) - float(start) - float(latency)
            if interval >= 0:
                values.append(interval)
    return statistics.mean(values) if values else None


def _audit(run_dir: Path, index: Any) -> list[dict[str, Any]]:
    if not isinstance(index, dict):
        return []
    episode_id, count = index.get("episode_id"), index.get("decision_count")
    if not isinstance(episode_id, str) or len(episode_id) != 32 or not all(
        char in "0123456789abcdef" for char in episode_id
    ) or type(count) is not int or not 0 <= count <= 1_000_000:
        raise ValueError("invalid Jev audit index")
    if index.get("write_failures") or index.get("missing_files"):
        raise ValueError("incomplete Jev audit; inspect trial_metadata.jev_audit")
    rows = []
    for number in range(1, count + 1):
        decision_id = f"{episode_id}-{number:06d}"
        path = run_dir / "jev-audit" / episode_id / f"{decision_id}.json"
        with path.open(encoding="utf-8") as stream:
            row = json.load(stream)
        if row.get("decision_id") != decision_id:
            raise ValueError(f"audit decision ID mismatch: {path}")
        rows.append(row)
    return rows


def rows_for_log(path: Path, expected_policy: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        log = json.load(stream)
    spec = log["eval"]
    if spec["policy"] != expected_policy or spec["task"] != "isaacsim-2yam-put-everything-in-box":
        raise ValueError(f"wrong policy or task in {path}")
    if spec.get("max_steps") != 400:
        raise ValueError(f"expected task's 400-step horizon in {path}")
    recordings = list(path.parent.glob("*.rrd"))
    if len(recordings) != 1:
        raise ValueError(f"expected exactly one .rrd beside {path}, found {len(recordings)}")
    rewards = rewards_from_rrd(recordings[0])
    rows: list[dict[str, Any]] = []
    for sample in log["samples"]:
        scene_id = sample["scene_id"]
        scene_index = sample.get("scene_metadata", {}).get("episode_index")
        if type(scene_index) is not int:
            raise ValueError(f"missing episode_index for {scene_id}")
        # The task constructor seed is supplied separately because EvalSpec
        # records only the outer eval seed, not -T seed.
        for epoch, score in enumerate(sample["epochs"]):
            success = score.get("success_at_end")
            if success is not None and success not in (0, 1, 0.0, 1.0):
                raise ValueError(f"unexpected success score for {scene_id}/e{epoch}")
            reward = rewards.get((scene_id, epoch))
            metadata = sample.get("trial_metadata", [])
            audit_rows = _audit(path.parent, metadata[epoch].get("jev_audit")) if (
                expected_policy != "molmoact2" and epoch < len(metadata)
            ) else []
            if expected_policy != "molmoact2" and (epoch >= len(metadata) or
                    "jev_audit" not in metadata[epoch]):
                raise ValueError(f"missing Jev audit index for {scene_id}/e{epoch}")
            rows.append({"policy": expected_policy, "scene_id": scene_id, "epoch": epoch,
                         "scene_index": scene_index, "success": success,
                         "last_reward": reward[1] if reward else None,
                         "reward_status": "present" if reward else "missing",
                         "last_reward_step": reward[0] if reward else None,
                         "decisions": len(audit_rows) if audit_rows else (0 if expected_policy != "molmoact2" else None),
                         "holds": sum(row.get("reason") is not None or row.get("selected_id") in
                                      ("hold", "reobserve") for row in audit_rows) if expected_policy != "molmoact2" else None,
                         "mean_policy_latency_s": _mean(audit_rows, "latency_s") if audit_rows else None,
                         "mean_molmo_latency_s": _mean(audit_rows, "molmo_latency_s") if audit_rows else None,
                         "mean_jev_latency_s": _mean(audit_rows, "jev_latency_s") if audit_rows else None,
                         "mean_post_policy_interval_s": _post_policy_interval(audit_rows),
                         "run_mean_inference_latency_s": log.get("stats", {}).get(
                             "mean_inference_latency_s")})
    unexpected = set(rewards) - {(row["scene_id"], row["epoch"]) for row in rows}
    if unexpected:
        raise ValueError(f"RRD has unmatched reward trials: {sorted(unexpected)}")
    return rows


def align(logs: dict[str, Path], *, task_seed: int, eval_seed: int) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    key_sets = []
    for policy in POLICIES:
        path = logs[policy]
        with path.open(encoding="utf-8") as stream:
            spec = json.load(stream)["eval"]
        if spec.get("seed") != eval_seed:
            raise ValueError(f"eval seed mismatch in {path}")
        rows = rows_for_log(path, policy)
        keys = set()
        for row in rows:
            index = row.pop("scene_index")
            if row["scene_id"] != f"isaacsim-2yam-put-everything-in-box-{index:03d}":
                raise ValueError(f"scene/index mismatch in {path}")
            row["scene_seed"] = task_seed + index
            payload = f"{eval_seed}:{row['scene_seed']}:{row['epoch']}".encode()
            row["trial_seed"] = zlib.crc32(payload) & 0xFFFFFFFF
            key = row["scene_id"], row["epoch"], row["trial_seed"]
            if key in keys:
                raise ValueError(f"duplicate trial in {path}: {key}")
            keys.add(key)
        key_sets.append(keys)
        all_rows.extend(rows)
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("the three policy runs have different scene/epoch/seed sets")
    return sorted(all_rows, key=lambda row: (row["scene_id"], row["epoch"], POLICIES.index(row["policy"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for policy in POLICIES:
        parser.add_argument("--" + policy + "-log", required=True, type=Path)
    parser.add_argument("--task-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logs = {policy: getattr(args, policy.replace("-", "_") + "_log") for policy in POLICIES}
    rows = align(logs, task_seed=args.task_seed, eval_seed=args.eval_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
