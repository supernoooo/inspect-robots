"""Batch 07 offline evidence, boundary, and fault-matrix regression."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import json_numpy
import numpy as np
import pytest

from inspect_robots import Scene
from inspect_robots.frames import FrameStore
from inspect_robots_isaacsim_2yam.contract import CAMERA_NAMES
from inspect_robots_jev.audit import (
    MAX_RECORD_BYTES, MAX_RECORDS, append_record, bounded_record, load_decision,
    load_episode,
)
from inspect_robots_jev.diagnostics import diagnose_transcript

from test_direct import MODEL, obs, services as direct_services
from test_hybrid import services as hybrid_services


@pytest.mark.parametrize("branch", ["direct_services", "hybrid_services"])
def test_audit_links_frames_masks_decision_and_observed_joint_result(
    request, branch, q0, tmp_path: Path
) -> None:
    policy, calls, mode = request.getfixturevalue(branch)
    if branch == "hybrid_services":
        mode["jev"] = "eef"
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    first_obs = obs(q0)
    first = policy.act(first_obs)
    first_row = policy.audit_records[0]
    assert first_row["decision_id"] and first_row["branch"] == policy.info.name
    assert first_row["trajectory"] == [action.data.tolist() for action in first.actions]
    assert first_row["observation"]["state_time"] == first_obs.state_time
    assert first_row["observation"]["image_times"] == first_obs.image_times
    assert first_row["vision"]["left_cam"]["detections"][0]["mask_ref"]["mask_area_px"] > 0
    mask_name = first_row["vision"]["left_cam"]["detections"][0]["mask_ref"]["mask_file"]
    assert mask_name.startswith("jev-masks/") and (tmp_path / mask_name).is_file()
    with np.load(tmp_path / mask_name) as saved:
        assert saved["mask"].dtype == np.bool_

    frame_store = FrameStore(str(tmp_path / "frames"))
    refs = {name: frame_store.put("scene-e0", 0, name, first_obs.images[name])
            for name in CAMERA_NAMES}
    record = SimpleNamespace(policy_transcript=policy.transcript(),
                             steps=[SimpleNamespace(observation=first_obs, image_refs=refs)])
    policy.on_trial_end(record, str(tmp_path), "run")
    logged = record.policy_transcript[0]
    assert logged["decision_id"] == first_row["decision_id"]
    assert all((tmp_path / path).is_file()
               for path in logged["observation"]["frame_refs"].values())

    # A fresh allowed observation supplies execution evidence. A stopped arm
    # must not be diagnosed as a successful stage advance.
    second = policy.act(obs(q0, image_value=1))
    assert second is not None
    after = policy.audit_records[0]
    assert after["execution"]["status"] == "not_reached_joint_target"
    assert after["execution"]["max_joint_error"] > 0.05
    assert after["execution"]["observed_joint_pos"] == q0.tolist()
    assert policy.candidates.stage.value == "approach"
    diagnosed = diagnose_transcript(list(policy.audit_records))[0]
    assert diagnosed["execution_not_reached"]["status"] == "suspected"
    assert diagnosed["decision_id"] == first_row["decision_id"]
    assert len(calls["jev"]) == 2


@pytest.mark.parametrize("branch", ["direct_services", "hybrid_services"])
def test_extra_truth_baits_never_enter_decision_or_log(request, branch, q0, monkeypatch) -> None:
    policy, calls, mode = request.getfixturevalue(branch)
    if branch == "hybrid_services":
        mode["jev"] = "eef"
    source = obs(q0)
    source.state.update(truth_object_pose="BAIT_POSE_999", reward="BAIT_REWARD_999",
                        success="BAIT_SUCCESS_999")
    policy.act(source)
    policy.act(obs(q0, image_value=1))
    transcript = json.dumps(policy.transcript())
    assert "BAIT_" not in transcript and "secret-test-key" not in transcript
    assert "rgb_base64" not in transcript and "Authorization" not in transcript
    assert len(transcript.encode()) < 2 * 1024 * 1024
    for kind in ("vision", "jev", "molmo_post"):
        for sent in calls.get(kind, []):
            if sent is not None:
                body = json_numpy.dumps(sent) if kind == "molmo_post" else json.dumps(sent)
                assert "BAIT_" not in body and "secret-test-key" not in body
    first_id = policy.audit_records[0]["decision_id"]
    policy.reset(Scene(id="second", instruction="place objects in box"))
    assert policy.transcript() == [] and policy.episode_calls == 0
    policy.act(obs(q0))
    assert policy.audit_records[0]["decision_id"] != first_id
    assert policy.audit_records[0]["call"] == 1


@pytest.mark.parametrize("branch", ["direct_services", "hybrid_services"])
def test_both_branches_advance_only_after_fresh_reached_observation(request, branch, q0) -> None:
    policy, calls, mode = request.getfixturevalue(branch)
    if branch == "hybrid_services":
        mode["jev"] = "eef"
    first = policy.act(obs(q0))
    assert first.meta["candidate_id"].endswith(":approach")
    assert policy.candidates.stage.value == "approach"
    second = policy.act(obs(first.actions[-1].data, image_value=1))
    assert second.meta["candidate_id"].endswith(":align")
    assert policy.candidates.stage.value == "align"
    assert policy.audit_records[0]["execution"]["status"] == "reached_joint_target"
    assert policy.audit_records[0]["execution"]["max_joint_error"] == pytest.approx(0)
    assert len(calls["jev"]) == 2
    if branch == "hybrid_services":
        assert len(calls["molmo_post"]) == 2


@pytest.mark.parametrize(("branch", "fault", "expected"), [
    ("direct_services", {"vision": "timeout"}, "vision_timeout"),
    ("direct_services", {"vision": "empty"}, "vision_empty_detection"),
    ("direct_services", {"vision": "box_only"}, "no_safe_candidate"),
    ("direct_services", {"jev": "timeout"}, "jev_timeout"),
    ("direct_services", {"jev": "unknown"}, "jev_unknown_candidate_id"),
    ("hybrid_services", {"vision": "timeout"}, "vision_timeout"),
    ("hybrid_services", {"molmo": "nan"}, "molmo_nonfinite_actions"),
    ("hybrid_services", {"molmo": "collision"}, "molmo_table_collision"),
    ("hybrid_services", {"molmo": "timeout"}, "molmo_timeout"),
    ("hybrid_services", {"jev": "timeout"}, "jev_timeout"),
    ("hybrid_services", {"jev": "unknown"}, "jev_unknown_candidate_id"),
])
def test_fault_matrix_has_specific_hold_reason_and_no_cross_branch(
    request, branch, fault, expected, q0
) -> None:
    policy, calls, mode = request.getfixturevalue(branch)
    mode.update(fault)
    result = policy.act(obs(q0))
    assert result.meta == {"kind": "hold", "reason": expected}
    np.testing.assert_array_equal(result.actions[0].data, q0)
    assert policy.audit_records[-1]["reason"] == expected
    assert policy.audit_records[-1]["decision_id"]
    if branch == "direct_services":
        assert "molmo_post" not in calls
    elif expected.startswith("molmo_") or expected.startswith("vision_"):
        assert calls["jev"] == []
    if expected.startswith("molmo_"):
        assert policy.audit_records[-1]["filtered"]["molmo:prefix"]
        assert policy.audit_records[-1]["candidates"]


def test_diagnostics_separate_vision_geometry_choice_and_execution(direct_services, q0) -> None:
    policy, _, mode = direct_services
    mode["vision"] = "timeout"
    policy.act(obs(q0))
    assert diagnose_transcript(list(policy.audit_records))[-1]["vision_error"]["status"] == "suspected"
    mode["vision"] = "box_only"
    policy.act(obs(q0))
    assert diagnose_transcript(list(policy.audit_records))[-1]["unreachable_or_filtered"]["status"] == "suspected"
    mode.update(vision="normal", jev="reobserve")
    policy.act(obs(q0))
    choice = diagnose_transcript(list(policy.audit_records))[-1]["jev_choice"]
    assert choice["status"] == "review_required" and choice["evidence"]["selected_id"] == "reobserve"


def test_audit_is_bounded_redacted_and_reset_clears_ring(direct_services, q0) -> None:
    row = {"decision_id": "id", "branch": "jev-direct", "reason": None,
           "candidates": [{"id": "safe", "source": "eef"}], "filtered": {"unsafe": "collision"},
           "vision": {"left_cam": {"detections": [{"model_version": "secret-test-key" * 100,
                                                      "category": "red_block"}] * 100}}}
    clean = bounded_record(row, "secret-test-key")
    assert len(json.dumps(clean).encode()) <= MAX_RECORD_BYTES
    assert clean["candidates"][0]["id"] == "safe"
    assert clean["filtered"]["unsafe"] == "collision"
    assert "secret-test-key" not in json.dumps(clean)
    assert clean["vision"]["left_cam"]["detections_omitted"] > 0
    rows = []
    for index in range(MAX_RECORDS + 3):
        append_record(rows, {"decision_id": str(index)}, None)
    assert len(rows) == MAX_RECORDS and rows[0]["decision_id"] == "3"
    policy, _, _ = direct_services
    for _ in range(MAX_RECORDS + 3):
        policy.act(obs(q0, at=time.monotonic() - 2))
    assert len(policy.transcript()) == MAX_RECORDS
    assert policy.audit_records[-1]["audit_omitted_prior"] >= 2
    policy.reset(Scene(id="next", instruction="place objects in box"))
    assert policy.transcript() == []


def test_diagnostic_command_reads_eval_log_without_truth(tmp_path: Path, direct_services, q0) -> None:
    policy, _, mode = direct_services
    mode["vision"] = "box_only"
    policy.act(obs(q0))
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"samples": [{"scene_id": "synthetic", "policy_transcripts":
                                          [policy.transcript()], "reward": "BAIT_REWARD"}]}),
                    encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "diagnose_trajectory.py"
    result = subprocess.run([sys.executable, str(script), str(path)], check=True,
                            text=True, capture_output=True)
    output = json.loads(result.stdout)
    assert output[0]["scene_id"] == "synthetic"
    assert output[0]["diagnoses"][0]["unreachable_or_filtered"]["status"] == "suspected"
    assert "BAIT_REWARD" not in result.stdout


@pytest.mark.parametrize("branch", ["direct_services", "hybrid_services"])
def test_long_trial_every_decision_is_retrievable_after_inline_limit(
    request, branch, q0, tmp_path: Path, monkeypatch
) -> None:
    policy, _, mode = request.getfixturevalue(branch)
    monkeypatch.setitem(MODEL, "detector_revision", "secret-test-key")
    if branch == "hybrid_services":
        mode["jev"] = "eef"
    policy.on_trial_start("long", 0, str(tmp_path), "run")
    policy.reset(Scene(id="long", instruction="place objects in box"))
    frame_store = FrameStore(str(tmp_path / "frames"))
    steps = []
    for index in range(MAX_RECORDS + 12):
        source = obs(q0)
        source.state.update(truth_object_pose="BAIT_POSE", reward="BAIT_REWARD",
                            success="BAIT_SUCCESS")
        result = policy.act(source)
        assert result.meta["candidate_id"].endswith(":approach")
        refs = {name: frame_store.put("long-e0", index, name, source.images[name])
                for name in CAMERA_NAMES}
        steps.append(SimpleNamespace(observation=source, image_refs=refs))
    assert len(policy.transcript()) == MAX_RECORDS
    record = SimpleNamespace(policy_transcript=policy.transcript(), steps=steps, metadata={})
    policy.on_trial_end(record, str(tmp_path), "run")
    index = record.metadata["jev_audit"]
    assert index["decision_count"] == MAX_RECORDS + 12
    assert index["missing_files"] == index["write_failures"] == 0
    rows = load_episode(tmp_path, index["episode_id"], index["decision_count"])
    assert len(rows) == MAX_RECORDS + 12
    assert len({row["decision_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["candidates"] and isinstance(row["filtered"], dict)
        assert all(isinstance(reason, str) and reason for reason in row["filtered"].values())
        assert row["selected_id"] and row["trajectory"]
        assert row["execution"] and row["observation"]["frame_refs"]
        assert len((tmp_path / row["audit_file"]).read_bytes()) <= MAX_RECORD_BYTES
        text = json.dumps(row)
        assert "BAIT_" not in text and "secret-test-key" not in text
        assert "rgb_base64" not in text and "Authorization" not in text
    assert rows[0]["execution"]["status"] == "not_reached_joint_target"
    assert rows[-1]["execution"]["status"] == "awaiting_observation"
    assert load_decision(tmp_path, rows[0]["decision_id"]) == rows[0]
    with pytest.raises(ValueError, match="decision ID"):
        load_decision(tmp_path, "../outside")
    assert record.policy_transcript[0]["decision_id"] == rows[12]["decision_id"]

    log_path = tmp_path / "eval.json"
    log_path.write_text(json.dumps({"samples": [{"scene_id": "long",
                                                "policy_transcripts": [record.policy_transcript],
                                                "trial_metadata": [record.metadata]}]}), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "diagnose_trajectory.py"
    result = subprocess.run([sys.executable, str(script), str(log_path)], check=True,
                            text=True, capture_output=True)
    assert len(json.loads(result.stdout)[0]["diagnoses"]) == len(rows)
    one = subprocess.run([sys.executable, str(script), str(log_path), "--decision-id",
                          rows[0]["decision_id"]], check=True, text=True, capture_output=True)
    assert json.loads(one.stdout)[0]["decision_id"] == rows[0]["decision_id"]
    old_id = rows[0]["decision_id"]
    policy.reset(Scene(id="another", instruction="place objects in box"))
    assert policy.transcript() == [] and policy.episode_calls == 0
    assert load_decision(tmp_path, old_id)["decision_id"] == old_id
