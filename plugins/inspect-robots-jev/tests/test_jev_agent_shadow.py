"""Batch 07 offline checks. No test opens a device or sends an action."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import Action
from inspect_robots_agent.proposals import ProposalBatch, ProposalCandidate
from inspect_robots_jev.agent_candidates import AgentCandidateValidator
from inspect_robots_jev.audit import load_decision
from inspect_robots_jev.jev_choice import ChoiceResult
from inspect_robots_jev.shadow import CameraRead, JointRead, ShadowRunner
from inspect_robots_yam.collision import build_yam_guardrails
from inspect_robots_yam.config import YamConfig, action_box
from inspect_robots_yam.embodiment import YAMEmbodiment

from test_agent_candidates import FakeCollisionChecker, FakeRig, batch, candidate, observation


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class Source:
    def __init__(self, q: np.ndarray, clock: Clock) -> None:
        self.q, self.clock = q, clock
        self.calls: list[str] = []
        self.stamp = 9.9

    def read_cameras(self) -> CameraRead:
        self.calls.append("cameras")
        return CameraRead(
            {name: np.zeros((101, 101, 3), np.uint8) for name in ("top_cam", "left_cam", "right_cam")},
            {name: self.stamp for name in ("top_cam", "left_cam", "right_cam")},
            {name: f"frame-{name}" for name in ("top_cam", "left_cam", "right_cam")})

    def read_joints(self) -> JointRead:
        self.calls.append("joints")
        return JointRead(self.q.copy(), self.stamp)

    def __getattr__(self, name: str):
        if name in {"connect", "close", "reset", "step", "send_action", "eval", "health", "holdcheck"}:
            pytest.fail(f"shadow touched forbidden hardware path {name}")
        raise AttributeError(name)


class Proposer:
    def __init__(self, clock: Clock, cost: float = 0.0) -> None:
        self.clock, self.cost = clock, cost
        self.calls = 0

    def propose(self, instruction, obs, feedback=None):
        self.calls += 1
        assert instruction == "place objects in box"
        self.clock.now += self.cost
        return batch(candidate("move", {"left_j0": 0.1}),
                     candidate("limit", {"left_j0": 9.0}))


class Choice:
    def __init__(self, clock: Clock, cost: float = 0.0) -> None:
        self.clock, self.cost = clock, cost
        self.calls = []

    def choose_generic(self, *, instruction, observation_context, candidates):
        self.calls.append([c.id for c in candidates])
        self.clock.now += self.cost
        return ChoiceResult("move", None, "fixture-jev", self.cost, None)


@pytest.fixture
def setup(motion, q0):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_left_base_pos=(0.0, 0.2, 0.2),
                    collision_right_base_pos=(0.0, -0.2, 0.2),
                    collision_left_base_yaw=0.0, collision_right_base_yaw=0.0,
                    collision_table_height=0.0)
    checker = FakeCollisionChecker(cfg)
    rig = FakeRig(cfg)
    validator = AgentCandidateValidator(rig, motion, collision_checker=checker,
                                         max_image_age_s=0.5, max_skew_s=0.2)
    return cfg, rig, validator, checker, q0


def runner(tmp_path, setup, *, agent_cost=0.0, jev_cost=0.0, max_dispatch_age_s=0.5):
    _, _, validator, _, q0 = setup
    clock = Clock()
    source = Source(q0, clock)
    agent, choice = Proposer(clock, agent_cost), Choice(clock, jev_cost)
    shadow = ShadowRunner(source, validator, agent, choice, out_dir=tmp_path,
                          max_dispatch_age_s=max_dispatch_age_s,
                          max_image_age_s=0.5, max_skew_s=0.2, clock=clock,
                          episode_id="b" * 32)
    return shadow, source, agent, choice, clock


def test_live_readbacks_proposal_only_and_frame_evidence(tmp_path, setup):
    shadow, source, agent, choice, clock = runner(tmp_path, setup, agent_cost=0.1, jev_cost=0.1)
    row = shadow.run_round("place objects in box", scene_change="unchanged")
    assert source.calls == ["cameras", "joints"]
    assert agent.calls == 1 and choice.calls == [["move", "hold"]]
    assert row["dispatch_status"] == "proposed_only" and row["proposed_only"] is True
    assert row["selected_for_dispatch"] is None and row["selected_id"] == "move"
    assert row["filtered"] == [{"id": "limit", "code": "joint_limit", "proposed_only": True}]
    assert row["agent_latency_s"] == pytest.approx(0.1)
    assert row["jev_latency_s"] == pytest.approx(0.1)
    assert row["dispatch_age_s"] == pytest.approx(0.3)
    refs = row["observation"]["frame_refs"]
    assert set(refs) == {"top_cam", "left_cam", "right_cam"}
    assert all((tmp_path / item["path"]).is_file() for item in refs.values())
    assert all(hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]
               for item in refs.values())
    assert load_decision(tmp_path, row["decision_id"]) == row
    stats = shadow.summary()
    assert stats["latency"]["agent_latency_s"]["p95_s"] == pytest.approx(0.1)
    assert stats["over_age_hold_rate"] == 0.0
    assert stats["proposed_only"] is True


@pytest.mark.parametrize("fault,reason", [
    ("stale", "stale_time"),
    ("late", "dispatch_stale"),
    ("changed", "scene_changed"),
    ("unknown", "scene_unknown"),
    ("bad_joint", "invalid_joint_pos"),
    ("bad_rgb", "invalid_rgb"),
])
def test_faults_only_record_hold(tmp_path, setup, fault, reason):
    shadow, source, agent, choice, clock = runner(tmp_path, setup, agent_cost=0.2,
                                                   jev_cost=0.2, max_dispatch_age_s=0.25)
    scene_change = fault if fault in {"changed", "unknown"} else "unchanged"
    if fault == "stale":
        source.stamp = 9.0
    elif fault == "bad_joint":
        source.q[0] = np.nan
    elif fault == "bad_rgb":
        original = source.read_cameras
        def bad_camera():
            read = original()
            return CameraRead({**read.images, "top_cam": np.zeros((1, 1, 3), np.uint8)},
                              read.image_times, read.frame_ids)
        source.read_cameras = bad_camera
    row = shadow.run_round("place objects in box", scene_change=scene_change)
    assert row["reason"] == reason
    assert row["selected_for_dispatch"] is None
    assert row["dispatch_status"] == "proposed_hold"
    assert row["proposed_only"] is True
    if fault == "late":
        assert agent.calls == 1 and choice.calls
        assert shadow.summary()["over_age_hold_rate"] == 1.0
    else:
        assert agent.calls == 0 and not choice.calls


def test_read_failure_and_repeated_timestamp_fail_closed(tmp_path, setup):
    shadow, source, agent, choice, clock = runner(tmp_path, setup)
    first = shadow.run_round("place objects in box", scene_change="unchanged")
    second = shadow.run_round("place objects in box", scene_change="unchanged")
    assert first["reason"] is None
    assert second["reason"] == "observation_not_new"
    assert agent.calls == 1


def test_configuration_gates(tmp_path, setup, motion):
    cfg, _, _, _, q0 = setup
    for variant in (replace(cfg, collision_guardrail=False),
                    replace(cfg, auto_start=True), replace(cfg, unattended=True)):
        if not variant.collision_guardrail:
            with pytest.raises(ValueError, match="collision"):
                AgentCandidateValidator(FakeRig(variant), motion,
                                        collision_checker=FakeCollisionChecker(variant))
            continue
        checker = FakeCollisionChecker(variant)
        validator = AgentCandidateValidator(FakeRig(variant), motion, collision_checker=checker)
        clock = Clock()
        with pytest.raises(ValueError, match="safety configuration"):
            ShadowRunner(Source(q0, clock), validator, Proposer(clock), Choice(clock),
                         out_dir=tmp_path, max_dispatch_age_s=0.5,
                         max_image_age_s=0.5, max_skew_s=0.2)


def test_local_filter_and_yam_contributed_guardrail_offline(setup):
    cfg, _, validator, checker, q0 = setup
    checker.trigger = lambda q: q[0] >= 0.09
    local = validator.validate(batch(candidate("collision", {"left_j0": 0.1}),
                                     candidate("limit", {"left_j0": 9.0})),
                               observation(q0), now=10.0)
    assert not local.available
    assert {item.code for item in local.filtered} == {"rig_collision", "joint_limit"}

    # This is the installed YAM guardrail contribution and real MuJoCo model,
    # exercised entirely in memory. It is not proof of the field rig geometry.
    yam = YAMEmbodiment(cfg, driver_factory=lambda _: pytest.fail("driver opened"))
    contribution = yam.contribute_guardrails(action_box(cfg.low, cfg.high))
    assert not contribution.warnings
    assert [name for name, _ in contribution.approvers] == ["yam-collision"]
    target = q0.copy()
    target[1] = 2.0  # Legal absolute joint target, known self-collision in model.
    reviewed = contribution.approvers[0][1].review(Action(data=target), {})
    assert reviewed.meta["collision_blocked"] is True
    np.testing.assert_array_equal(reviewed.data, q0)
    assert target[1] <= cfg.high[1]

    # The YAM execution chain also clamps a gripper target outside [0, 1].
    over_limit = q0.copy()
    over_limit[6] = 2.0
    approved = build_yam_guardrails(action_box(cfg.low, cfg.high), cfg).review(
        Action(data=over_limit), {})
    assert approved.data[6] == 1.0
    assert approved.data[6] != over_limit[6]


def test_shadow_cli_import_has_no_real_yam_or_driver() -> None:
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name == 'yam_arms' or name.startswith('inspect_robots_yam.embodiment'):
        raise AssertionError('hardware import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
runpy.run_path('plugins/inspect-robots-jev/scripts/shadow_jev_agent.py', run_name='import_only')
assert 'inspect_robots_yam.embodiment' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=Path(__file__).parents[3], env=os.environ.copy())
    assert result.returncode == 0, result.stderr
