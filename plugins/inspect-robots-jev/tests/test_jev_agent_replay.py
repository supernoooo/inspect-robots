"""Batch 06: saved RGB replay is proposal-only, including failure paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from inspect_robots_agent.proposals import ProposalBatch, ProposalCandidate, ProposalFailure, ProposalTermination
from inspect_robots_jev.agent_candidates import AgentCandidateValidator
from inspect_robots_jev.audit import load_decision, load_episode
from inspect_robots_jev.jev_choice import ChoiceError, ChoiceResult
from inspect_robots_jev.replay import replay
from inspect_robots_yam.config import YamConfig

from test_agent_candidates import FakeCollisionChecker, FakeRig


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class Agent:
    def __init__(self, *results, clock: Clock | None = None, cost: float = 0.0) -> None:
        self.results = list(results)
        self.clock = clock
        self.cost = cost
        self.feedback = []

    def propose(self, instruction, observation, feedback=None):
        assert instruction == "place objects in box"
        assert set(observation.images) == {"top_cam", "left_cam", "right_cam"}
        self.feedback.append(feedback)
        if self.clock:
            self.clock.now += self.cost
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class Choice:
    def __init__(self, *results, clock: Clock | None = None, cost: float = 0.0) -> None:
        self.results = list(results)
        self.clock = clock
        self.cost = cost
        self.calls = []

    def choose_generic(self, *, instruction, observation_context, candidates):
        assert instruction == "place objects in box"
        assert "preferred" not in observation_context
        self.calls.append([item.id for item in candidates])
        if self.clock:
            self.clock.now += self.cost
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return ChoiceResult(result, None, "jev-1.13.0", self.cost,
                            {"input_tokens": 5, "output_tokens": 1})


def batch(*items, preferred=None):
    return ProposalBatch("saved", tuple(items), preferred or items[0].id,
                         "fixture-agent", 0.02, {"input_tokens": 3}, {})


def candidate(id, targets, note="fixture"):
    return ProposalCandidate(id, targets, note, "fixture effect")


@pytest.fixture
def replay_validator(motion):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_left_base_pos=(0.0, 0.2, 0.2),
                    collision_right_base_pos=(0.0, -0.2, 0.2),
                    collision_left_base_yaw=0.0, collision_right_base_yaw=0.0,
                    collision_table_height=0.0)
    rig = FakeRig(cfg)
    checker = FakeCollisionChecker(cfg)
    validator = AgentCandidateValidator(
        rig, motion, collision_checker=checker,
        max_image_age_s=0.5, max_skew_s=0.2)
    # Any accidental hardware-like operation on the injected objects fails.
    for name in ("eval", "reset", "health", "holdcheck", "step", "send_action"):
        setattr(validator, name, lambda *args, **kwargs: pytest.fail("hardware path called"))
    return validator, checker


def saved_manifest(tmp_path: Path, q, *, rounds=2):
    entries = []
    for index in range(rounds):
        rgb = {}
        for name in ("top_cam", "left_cam", "right_cam"):
            path = tmp_path / f"{index + 1:04d}-{name}.npy"
            np.save(path, np.full((101, 101, 3), 73 + index, dtype=np.uint8))
            rgb[name] = path.name
        stamp = 9.9 + index * 0.2
        entries.append({"replay_at": stamp + 0.1, "state_time": stamp,
                        "image_times": {name: stamp for name in rgb},
                        "rgb": rgb.copy(), "joint_pos": q.tolist(),
                        "instruction": "place objects in box",
                        "approvals": []})
    return {"episode_id": "a" * 32, "scene_id": "saved-scene",
            "max_image_age_s": 0.5, "max_skew_s": 0.2,
            "max_dispatch_age_s": 0.5, "rounds": entries}


def test_two_rounds_choice_comparison_feedback_and_sidecars(tmp_path, replay_validator, q0):
    validator, _ = replay_validator
    manifest = saved_manifest(tmp_path, q0)
    agent = Agent(batch(candidate("left", {"left_j0": 0.2}),
                        candidate("right", {"right_j0": 0.2}),
                        candidate("bad", {"left_bad": 0.1}), preferred="left"),
                  batch(candidate("next", {"left_j0": 0.1})))
    choice = Choice("right", "hold")
    clock = Clock()
    first = replay({**manifest, "rounds": manifest["rounds"][:1]},
                   base=tmp_path, validator=validator, clock=clock,
                   proposer=Agent(agent.results[0]), choice=Choice("right"))[0]
    manifest["rounds"][1]["joint_pos"] = first["selected_prefix"][-1]
    manifest["rounds"][1]["approvals"] = [{"decision_id": first["decision_id"],
                                            "detail": "clamped"}]
    clock.now = 10.0
    rows = replay(manifest, base=tmp_path, validator=validator, clock=clock,
                  proposer=agent, choice=choice, out_dir=tmp_path / "log")
    assert len(rows) == 2
    assert rows[0]["agent_selection"] == "left"
    assert rows[0]["jev_selection"] == "right"
    assert rows[0]["selected_for_dispatch"] is None
    assert rows[0]["dispatch_status"] == "proposed_only"
    assert [item["id"] for item in rows[0]["candidates"]] == ["left", "right"]
    assert rows[0]["filtered"] == [{"id": "bad", "code": "unknown_joint",
                                    "proposed_only": True}]
    assert rows[0]["execution"] == {"status": "comparison_only", "proposed_only": True,
                                     "matches_proposed_target": True}
    assert rows[1]["reason"] == "jev_hold"
    assert rows[1]["approval"]["events"] == [{"detail": "clamped"}]
    assert rows[0]["agent_usage"] == {"input_tokens": 3}
    assert rows[0]["jev_usage"] == {"input_tokens": 5, "output_tokens": 1}
    assert agent.feedback[1] is not None
    assert '"proposed_only":true' in agent.feedback[1].outcome
    assert choice.calls == [["left", "right", "hold"], ["next", "hold"]]
    assert load_decision(tmp_path / "log", rows[0]["decision_id"]) == rows[0]
    assert all(row["proposed_only"] is True for row in rows)
    assert all(row["execution"]["status"] not in ("observed_after_dispatch", "reached_joint_target")
               for row in rows)


@pytest.mark.parametrize(("fault", "reason"), [
    ("missing_rgb", "missing_rgb"),
    ("stale_time", "stale_time"),
    ("future_time", "stale_time"),
    ("invalid_time", "invalid_time"),
    ("time_skew", "time_skew"),
    ("duplicate", "observation_not_new"),
    ("duplicate_content", "duplicate_frame"),
    ("replay_order", "replay_time_not_new"),
    ("agent_failure", "agent_request_error"),
    ("jev_failure", "jev_timeout"),
    ("unknown_id", "jev_unknown_candidate_id"),
    ("collision", "no_safe_candidate"),
    ("next_state_mismatch", "next_state_mismatch"),
    ("stop", "done"),
])
def test_fault_matrix(tmp_path, replay_validator, q0, fault, reason):
    validator, checker = replay_validator
    manifest = saved_manifest(tmp_path, q0)
    good = batch(candidate("move", {"left_j0": 0.2}))
    agent = Agent(good, good)
    choice = Choice("move", "move")
    clock = Clock()
    if fault == "missing_rgb":
        manifest["rounds"][0]["rgb"]["top_cam"] = "missing.npy"
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "stale_time":
        manifest["rounds"][0]["image_times"]["top_cam"] = 9.0
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "future_time":
        manifest["rounds"][0]["state_time"] = 10.2
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "invalid_time":
        manifest["rounds"][0]["state_time"] = "invalid"
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "time_skew":
        manifest["rounds"][0]["image_times"]["top_cam"] = 9.6
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "duplicate":
        manifest["rounds"][1]["state_time"] = manifest["rounds"][0]["state_time"]
        manifest["rounds"][1]["image_times"] = manifest["rounds"][0]["image_times"].copy()
    elif fault == "duplicate_content":
        manifest["rounds"][1]["rgb"] = manifest["rounds"][0]["rgb"].copy()
    elif fault == "replay_order":
        manifest["rounds"][1]["replay_at"] = 9.99
        manifest["rounds"][1]["state_time"] = 9.95
        manifest["rounds"][1]["image_times"] = {name: 9.95 for name in
                                                 manifest["rounds"][1]["image_times"]}
    elif fault == "agent_failure":
        agent = Agent(ProposalFailure("request_error", "secret detail", "fixture", 0, None, None))
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "jev_failure":
        choice = Choice(ChoiceError("timeout"))
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "unknown_id":
        choice = Choice("unknown")
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "collision":
        checker.trigger = lambda q: True
        manifest["rounds"] = manifest["rounds"][:1]
    elif fault == "next_state_mismatch":
        pass
    elif fault == "stop":
        agent = Agent(batch(candidate("first", {"left_j0": 0.1})),
                      ProposalTermination("done", "stop", "", "fixture", 0, None, {}))
        choice = Choice("first")
        next_q = q0.copy()
        next_q[0] = 0.1
        manifest["rounds"][1]["joint_pos"] = next_q.tolist()
    if len(manifest["rounds"]) == 2 and fault != "next_state_mismatch":
        choice = Choice("move", "move") if fault == "duplicate" else choice
    rows = replay(manifest, base=tmp_path, validator=validator, clock=clock,
                  proposer=agent, choice=choice)
    target = rows[-1] if fault in ("duplicate", "duplicate_content", "replay_order",
                                  "next_state_mismatch", "stop") else rows[0]
    assert target["reason"] == reason
    assert target["dispatch_status"] == "proposed_hold"
    assert target["proposed_only"] is True
    assert target["selected_for_dispatch"] is None
    if fault == "collision":
        assert target["filtered"][0]["code"] == "rig_collision"
    if fault == "next_state_mismatch":
        assert rows[0]["observed_state"]["max_joint_error_to_proposed"] > 0.05
        assert rows[0]["execution"]["matches_proposed_target"] is False
    if fault == "stop":
        assert len(rows) == 2 and rows[0]["selected_id"] == "first"
        assert target["termination"] == {"status": "done", "proposed_only": True}


def test_injected_monotonic_clock_preserves_age_and_late_service_holds(tmp_path,
                                                                       replay_validator, q0):
    validator, _ = replay_validator
    manifest = saved_manifest(tmp_path, q0, rounds=1)
    manifest["max_dispatch_age_s"] = 0.3
    clock = Clock()
    agent = Agent(batch(candidate("move", {"left_j0": 0.1})), clock=clock, cost=0.1)
    choice = Choice("move", clock=clock, cost=0.15)
    row = replay(manifest, base=tmp_path, validator=validator, clock=clock,
                 proposer=agent, choice=choice)[0]
    assert row["observation"]["state_age_s"] == pytest.approx(0.1)
    assert row["agent_latency_s"] == pytest.approx(0.1)
    assert row["jev_latency_s"] == pytest.approx(0.15)
    assert row["latency_s"] == pytest.approx(0.25)
    assert row["reason"] == "dispatch_stale"
    assert row["proposed_only"] is True


def test_filtered_agent_preference_and_jev_choice_share_checked_set(tmp_path,
                                                                     replay_validator, q0):
    validator, _ = replay_validator
    manifest = saved_manifest(tmp_path, q0, rounds=1)
    agent = Agent(batch(candidate("unsafe", {"left_bad": 0.1}),
                        candidate("safe", {"left_j0": 0.1}), preferred="unsafe"))
    choice = Choice("safe")
    row = replay(manifest, base=tmp_path, validator=validator, clock=Clock(),
                 proposer=agent, choice=choice)[0]
    assert row["agent_preferred_id"] == "unsafe"
    assert row["agent_selection"] == "hold"
    assert row["agent_selection_reason"] == "preferred_filtered"
    assert row["jev_selection"] == "safe"
    assert choice.calls == [["safe", "hold"]]
    assert row["selected_prefix"] and row["proposed_only"] is True


def test_saved_png_rgb_files_are_decoded_without_entering_audit(tmp_path,
                                                                 replay_validator, q0):
    image_module = pytest.importorskip("PIL.Image")
    validator, _ = replay_validator
    manifest = saved_manifest(tmp_path, q0, rounds=1)
    for name in manifest["rounds"][0]["rgb"]:
        path = tmp_path / f"{name}.png"
        image_module.fromarray(np.full((101, 101, 3), 173, dtype=np.uint8), "RGB").save(path)
        manifest["rounds"][0]["rgb"][name] = path.name
    manifest["rounds"][0]["agent"] = {"candidates": [
        {"id": "move", "targets": {"left_j0": 0.1}}]}
    manifest["rounds"][0]["jev_id"] = "move"
    row = replay(manifest, base=tmp_path, validator=validator, clock=Clock())[0]
    assert row["selected_id"] == "move"
    assert "173, 173, 173" not in json.dumps(row)


def test_fixture_mode_redacts_secrets_and_never_emits_rgb(tmp_path, replay_validator, q0, monkeypatch):
    validator, _ = replay_validator
    monkeypatch.setenv("OPENAI_API_KEY", "secret-canary-123")
    manifest = saved_manifest(tmp_path, q0, rounds=1)
    manifest["rounds"][0]["agent"] = {"preferred_id": "one", "candidates": [
        {"id": "one", "targets": {"left_j0": 0.1}, "note": "secret-canary-123"}]}
    manifest["rounds"][0]["jev_id"] = "one"
    rows = replay(manifest, base=tmp_path, validator=validator,
                  clock=Clock(), out_dir=tmp_path / "log")
    wire = json.dumps(rows)
    assert "secret-canary-123" not in wire
    assert "[REDACTED]" not in wire  # Free-form service notes are absent.
    assert "73, 73, 73" not in wire
    assert rows[0]["execution"]["status"] == "unverified_no_next_observation"
    assert load_decision(tmp_path / "log", rows[0]["decision_id"])["proposed_only"] is True


def test_import_does_not_load_real_yam_embodiment():
    code = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name == 'yam_arms' or name.startswith('inspect_robots_yam.embodiment'):
        raise AssertionError('real hardware import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import runpy
runpy.run_path('plugins/inspect-robots-jev/scripts/replay_jev_agent.py', run_name='replay_import_only')
assert 'inspect_robots_yam.embodiment' not in __import__('sys').modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=Path(__file__).parents[3], env=os.environ.copy())
    assert result.returncode == 0, result.stderr


def test_cli_reads_saved_images_and_writes_batch05_sidecar(tmp_path, q0):
    manifest = saved_manifest(tmp_path, q0, rounds=2)
    second_q = q0.copy()
    second_q[0] = 0.1
    manifest["rounds"][1]["joint_pos"] = second_q.tolist()
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_left_base_pos=(0.0, 0.2, 0.2),
                    collision_right_base_pos=(0.0, -0.2, 0.2),
                    collision_left_base_yaw=0.0, collision_right_base_yaw=0.0,
                    collision_table_height=0.0)
    manifest["rig"] = asdict(cfg)
    fixtures = Path(__file__).parent / "fixtures"
    manifest["calibration"] = str(fixtures / "synthetic_calibration_v1.json")
    manifest["mjcf"] = str(fixtures / "tiny_yam.xml")
    manifest["rounds"][0]["agent"] = {"preferred_id": "move", "candidates": [
        {"id": "move", "targets": {"left_j0": 0.1}}]}
    manifest["rounds"][0]["jev_id"] = "move"
    manifest["rounds"][1]["agent"] = {"preferred_id": "next", "candidates": [
        {"id": "next", "targets": {"right_j0": 0.1}}]}
    manifest["rounds"][1]["jev_id"] = "next"
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "replay_jev_agent.py"
    out = tmp_path / "log"
    guarded_cli = """
import builtins, runpy, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name == 'yam_arms' or name.startswith('inspect_robots_yam.embodiment'):
        raise AssertionError('hardware import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name='__main__')
assert 'inspect_robots_yam.embodiment' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", guarded_cli, str(script),
                             str(source), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["proposed_only"] is True
    assert all(item["proposed_only"] is True for item in summary["decisions"])
    first, second = (load_decision(out, item["decision_id"]) for item in summary["decisions"])
    assert first["proposed_only"] is second["proposed_only"] is True
    assert first["selected_for_dispatch"] is second["selected_for_dispatch"] is None
    assert first["selected_id"] == "move" and second["selected_id"] == "next"
    assert first["execution"]["matches_proposed_target"] is True
    assert second["execution"]["status"] == "unverified_no_next_observation"
    assert first["observation"]["frame_refs"] == {name: f"0001-{name}.npy" for name in
                                                        ("top_cam", "left_cam", "right_cam")}
    index = json.loads((out / "replay-index.json").read_text(encoding="utf-8"))
    assert index["proposed_only"] is index["jev_audit"]["proposed_only"] is True
    assert index["jev_audit"]["decision_count"] == 2
    assert load_episode(out, index["jev_audit"]["episode_id"], 2) == [first, second]
