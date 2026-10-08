"""Batch 03 local candidate expansion against fake rig and deterministic geometry."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots import Action, ActionChunk, Observation
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots_agent._tools import ToolResult
from inspect_robots_agent.proposals import ProposalBatch, ProposalCandidate, ProposalFailure
from inspect_robots_jev.agent_candidates import AgentCandidateValidator
from inspect_robots_jev.contract import InputError
from inspect_robots_yam.collision import _config_from_yam
from inspect_robots_yam.config import YamConfig, action_box, observation_space


class FakeRig:
    def __init__(self, cfg: YamConfig) -> None:
        self._cfg = cfg
        self.info = EmbodimentInfo(
            "yam_arms", action_box(cfg.low, cfg.high),
            observation_space(cfg.cam_height, cfg.cam_width,
                              ("top_cam", "left_cam", "right_cam")),
            is_simulated=False)

    def reset(self, *_: object) -> None:
        pytest.fail("reset must not be called")

    def step(self, *_: object) -> None:
        pytest.fail("motor drive must not be called")

    def eval(self, *_: object) -> None:
        pytest.fail("ordinary real eval must not be called")


class FakeCollisionChecker:
    def __init__(self, cfg: YamConfig) -> None:
        self.config = _config_from_yam(cfg)
        self.trigger = lambda q: False
        self.samples: list[np.ndarray] = []

    def check(self, q: np.ndarray) -> SimpleNamespace:
        self.samples.append(q.copy())
        hit = bool(self.trigger(q))
        return SimpleNamespace(collided=hit, geom1="left_link", geom2="table")


def candidate(id: str, targets: dict[str, float]) -> ProposalCandidate:
    return ProposalCandidate(id, targets, "model note", "model intended effect")


def batch(*candidates: ProposalCandidate) -> ProposalBatch:
    return ProposalBatch("scene", tuple(candidates), candidates[0].id,
                         "fake-model", 0.01, None, {})


def observation(q: np.ndarray, *, stamp: float = 9.9) -> Observation:
    names = ("top_cam", "left_cam", "right_cam")
    return Observation(
        images={name: np.zeros((101, 101, 3), dtype=np.uint8) for name in names},
        state={"joint_pos": q.copy()}, instruction="place objects in box",
        image_times={name: stamp for name in names}, state_time=stamp)


@pytest.fixture
def harness(motion):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_left_base_pos=(0.0, 0.2, 0.2),
                    collision_right_base_pos=(0.0, -0.2, 0.2),
                    collision_left_base_yaw=0.0,
                    collision_right_base_yaw=0.0,
                    collision_table_height=0.0)
    rig = FakeRig(cfg)
    checker = FakeCollisionChecker(cfg)
    validator = AgentCandidateValidator(rig, motion, collision_checker=checker,
                                         max_image_age_s=0.5, max_skew_s=0.2)
    return validator, checker, rig


def test_both_arms_single_arm_prefix_summary_and_same_measured_origin(harness, q0) -> None:
    validator, checker, rig = harness
    proposal = batch(candidate("both", {"left_j0": 0.2, "right_j0": 0.2}),
                     candidate("right", {"right_j1": 0.2}))
    source = observation(q0)
    result = validator.validate(proposal, source, now=10.0)
    assert [x.id for x in result.available] == ["both", "right"]
    assert result.filtered == () and result.observation_reason is None
    assert len(result.hold) == 1
    np.testing.assert_array_equal(result.hold.actions[0].data, q0)
    assert result.hold.actions[0].data is not q0
    assert result.hold.meta == {"kind": "hold", "reason": "independent_hold"}
    both, right = result.available
    assert len(both.chunk) == 3
    assert both.summary["source_steps_checked"] > 3
    assert len(right.chunk) == 3
    np.testing.assert_array_equal(right.chunk.actions[0].data[:7], q0[:7])
    assert right.chunk.actions[0].data[8] > q0[8]
    assert both.chunk.actions[-1].data[0] < 0.2  # The verified suffix is absent.
    assert both.chunk.meta == {"candidate_id": "both", "reobserve_after_chunk": True}
    assert all("chunk_final" not in action.meta for action in both.chunk.actions)
    summary = both.jev_summary()
    assert summary["model_intent"] == {"note": "model note",
                                        "intended_effect": "model intended effect",
                                        "expected_visual_change": "model intended effect"}
    assert summary["computed"]["joint_delta"]["left_j0"] == pytest.approx(
        both.chunk.actions[-1].data[0])
    assert summary["computed"]["eef_delta"]["left"]["position_delta_m"][0] == pytest.approx(
        both.chunk.actions[-1].data[0])
    assert summary["computed"]["gripper"]["left"] == {
        "from": 1.0, "to": 1.0, "delta": 0.0}
    assert summary["local_checks"]["unknown_object_collision"] == "unchecked"
    provenance = summary["provenance"]
    assert provenance["mjcf_sha256"] == "synthetic-model"
    assert provenance["calibration_version"] == 1
    assert len(provenance["rig_config_sha256"]) == 64
    assert len(provenance["collision_config_sha256"]) == 64
    assert len(provenance["table_config_sha256"]) == 64
    json.dumps(summary)
    assert checker.samples  # Local model was queried.
    np.testing.assert_array_equal(source.state["joint_pos"], q0)
    assert rig.info.name == "yam_arms"


def test_full_path_and_interpolated_substep_collisions(harness, q0, motion) -> None:
    validator, checker, _ = harness
    checker.trigger = lambda q: q[0] >= 0.09
    result = validator.validate(batch(candidate("full", {"left_j0": 0.1})),
                                observation(q0), now=10.0)
    assert result.available == ()
    assert result.filtered[0].code == "rig_collision"
    assert any(q[0] >= 0.09 for q in checker.samples)

    checker.samples.clear()
    checker.trigger = lambda q: 0.024 < q[0] < 0.026
    result = validator.validate(batch(candidate("substep", {"left_j0": 0.15})),
                                observation(q0), now=10.0)
    assert result.available == ()
    assert result.filtered[0].code == "rig_collision"
    assert any(0.024 < q[0] < 0.026 for q in checker.samples)

    checker.trigger = lambda q: False
    original = motion.kinematics.link_positions

    def lowered(side: str, q: np.ndarray) -> np.ndarray:
        links = original(side, q)
        if side == "left" and 0.048 < q[0] < 0.052:
            links[:, 2] -= 0.3
        return links

    motion.kinematics.link_positions = lowered
    result = validator.validate(batch(candidate("local", {"left_j0": 0.1})),
                                observation(q0), now=10.0)
    assert result.available == ()
    assert result.filtered[0].code == "table_collision"


def test_suffix_collision_rejects_entire_candidate(harness, q0) -> None:
    validator, checker, _ = harness
    checker.trigger = lambda q: q[0] > 0.175
    result = validator.validate(batch(candidate("late", {"left_j0": 0.2})),
                                observation(q0), now=10.0)
    assert result.available == ()
    assert result.filtered[0].code == "rig_collision"
    assert result.filtered[0].detail.startswith("step 4:")


def test_rejections_dedup_and_single_survivor(harness, q0) -> None:
    validator, _, _ = harness
    proposal = batch(
        candidate("good", {"left_j0": 0.3}),
        candidate("same_prefix", {"left_j0": 0.36}),
        candidate("unknown", {"left_bad": 0.1}),
        candidate("limit", {"left_j0": 9.0}),
        candidate("nan", {"left_j0": float("nan")}),
        candidate("zero", {"left_j0": 0.0}),
    )
    result = validator.validate(proposal, observation(q0), now=10.0)
    assert [item.id for item in result.available] == ["good"]
    assert {item.id: item.code for item in result.filtered} == {
        "same_prefix": "duplicate_prefix", "unknown": "unknown_joint",
        "limit": "joint_limit", "nan": "non_finite_target",
        "zero": "zero_displacement"}
    assert len(result.available[0].chunk) == 3


@pytest.mark.parametrize(("change", "code"), [
    (lambda obs: obs.images.pop("top_cam"), "missing_camera"),
    (lambda obs: obs.image_times.update(left_cam=8.0), "stale_time"),
    (lambda obs: obs.image_times.update(left_cam=9.6), "time_skew"),
])
def test_bad_images_and_time_hold_all(harness, q0, change, code: str) -> None:
    validator, checker, _ = harness
    source = observation(q0)
    change(source)
    result = validator.validate(batch(candidate("one", {"left_j0": 0.1})),
                                source, now=10.0)
    assert result.available == () and result.observation_reason == code
    assert result.filtered[0].code == code
    assert result.hold.meta["reason"] == code
    assert checker.samples == []


def test_all_rejected_and_invalid_state_cannot_hold(harness, q0) -> None:
    validator, _, _ = harness
    result = validator.validate(batch(candidate("bad", {"left_j0": 8.0})),
                                observation(q0), now=10.0)
    assert result.available == () and len(result.filtered) == 1
    np.testing.assert_array_equal(result.hold.actions[0].data, q0)
    failed = ProposalFailure("model_failure", "bad response", "fake", 0.0, None, None)
    result = validator.validate(failed, observation(q0), now=10.0)
    assert result.available == () and result.observation_reason == "model_failure"
    bad = observation(q0)
    bad.state["joint_pos"][0] = np.nan
    with pytest.raises(InputError, match="finite"):
        validator.validate(batch(candidate("one", {"left_j0": 0.1})), bad, now=10.0)


def test_gripper_summary_and_bad_expanded_chunk(harness, q0, monkeypatch) -> None:
    validator, _, _ = harness
    result = validator.validate(batch(candidate("close", {"left_gripper": 0.8})),
                                observation(q0), now=10.0)
    assert [item.id for item in result.available] == ["close"]
    short = result.available[0]
    assert len(short.chunk) == 3
    assert short.summary["computed"]["gripper"]["left"]["to"] == pytest.approx(
        short.chunk.actions[-1].data[6])
    assert short.chunk.actions[-1].data[6] > 0.8
    assert short.summary["local_checks"]["finger_contact_collision"] == "unchecked"

    malformed = ActionChunk(actions=[Action(data=np.full(14, np.nan))])
    monkeypatch.setattr(validator._toolset, "execute", lambda *_: ToolResult(chunk=malformed))
    result = validator.validate(batch(candidate("broken", {"left_j0": 0.1})),
                                observation(q0), now=10.0)
    assert result.available == ()
    assert result.filtered[0].code == "invalid_chunk"


def test_speed_fraction_matches_agent_expansion(harness, motion, q0) -> None:
    default, _, rig = harness
    faster = AgentCandidateValidator(
        rig, motion, collision_checker=FakeCollisionChecker(rig._cfg),
        max_image_age_s=0.5, max_skew_s=0.2, max_speed_frac=0.25)
    proposal = batch(candidate("move", {"left_j0": 0.2}))
    slow_result = default.validate(proposal, observation(q0), now=10.0)
    fast_result = faster.validate(proposal, observation(q0), now=10.0)
    assert len(slow_result.available) == len(fast_result.available) == 1
    assert fast_result.available[0].chunk.actions[0].data[0] > slow_result.available[0].chunk.actions[0].data[0]
    assert fast_result.available[0].summary["provenance"]["agent_max_speed_frac"] == 0.25


def test_expansion_uses_actual_rig_gripper_step_declaration(harness, motion, q0) -> None:
    _, _, rig = harness
    cfg = rig._cfg
    rig.info = EmbodimentInfo(
        "yam_arms", action_box(cfg.low, cfg.high, gripper_max_step=cfg.gripper_max_step),
        observation_space(cfg.cam_height, cfg.cam_width,
                          ("top_cam", "left_cam", "right_cam")),
        is_simulated=False)
    validator = AgentCandidateValidator(
        rig, motion, collision_checker=FakeCollisionChecker(cfg),
        max_image_age_s=0.5, max_skew_s=0.2, max_speed_frac=0.25)
    result = validator.validate(batch(candidate("close", {"left_gripper": 0.0})),
                                observation(q0), now=10.0)
    assert [item.id for item in result.available] == ["close"]
    assert 0.9 <= result.available[0].chunk.actions[0].data[6] < 0.92


def test_nonfinite_eef_summary_is_filtered(harness, motion, q0, monkeypatch) -> None:
    validator, _, _ = harness
    original = motion.kinematics.forward

    def bad_forward(side: str, q: np.ndarray):
        pose = original(side, q)
        pose.position_base_m[0] = np.nan
        return pose

    monkeypatch.setattr(motion.kinematics, "forward", bad_forward)
    result = validator.validate(batch(candidate("bad_fk", {"left_j0": 0.1})),
                                observation(q0), now=10.0)
    assert result.available == ()
    assert result.filtered[0].code == "invalid_kinematics"


def test_actual_rig_step_limits_and_model_binding(harness, motion, q0) -> None:
    _, checker, rig = harness
    strict_steps = (0.001,) * 6 + (1.0,) + (0.001,) * 6 + (1.0,)
    cfg = replace(rig._cfg, step_limits=strict_steps)
    validator = AgentCandidateValidator(FakeRig(cfg), motion, collision_checker=checker)
    result = validator.validate(batch(candidate("too_fast", {"left_j0": 0.1})),
                                observation(q0), now=10.0)
    assert result.available == () and result.filtered[0].code == "rig_step_limit"

    from inspect_robots_jev.kinematics import YamKinematics
    from inspect_robots_jev.motion import MotionPlanner
    from pathlib import Path

    model = YamKinematics(Path(__file__).parent / "fixtures" / "tiny_yam.xml")
    planner = MotionPlanner(model, motion.calibration)
    AgentCandidateValidator(rig, planner, collision_checker=checker)
    wrong = FakeRig(replace(rig._cfg, collision_left_base_pos=(0.0, 0.3, 0.2)))
    with pytest.raises(ValueError, match="MJCF base"):
        AgentCandidateValidator(wrong, planner, collision_checker=FakeCollisionChecker(wrong._cfg))


def test_out_of_rig_bound_state_cannot_hold_even_on_missing_image(harness, motion, q0) -> None:
    _, checker, rig = harness
    narrow = replace(rig._cfg, joint_high=(0.1,) + rig._cfg.joint_high[1:])
    validator = AgentCandidateValidator(FakeRig(narrow), motion, collision_checker=checker)
    q = q0.copy()
    q[0] = 0.2
    source = observation(q)
    source.images.pop("top_cam")
    with pytest.raises(InputError) as caught:
        validator.validate(batch(candidate("one", {"left_j0": 0.05})), source, now=10.0)
    assert caught.value.code == "invalid_joint_pos"
    assert caught.value.joint_pos is None


def test_pairing_and_collision_config_mismatch_fail_before_work(harness, motion) -> None:
    _, checker, rig = harness
    bad_cfg = replace(rig._cfg, joints_are_delta=True)
    with pytest.raises(ValueError, match="pairing"):
        AgentCandidateValidator(FakeRig(bad_cfg), motion, collision_checker=checker)
    other = FakeCollisionChecker(replace(rig._cfg, collision_table_height=0.1))
    with pytest.raises(ValueError, match="collision checker"):
        AgentCandidateValidator(rig, motion, collision_checker=other)


def test_yam_mode_uses_rig_limits_without_measured_geometry(q0) -> None:
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_guardrail=False, collision_table=False)
    validator = AgentCandidateValidator(FakeRig(cfg), max_image_age_s=0.5,
                                        max_skew_s=0.2, max_speed_frac=0.25)
    result = validator.validate(batch(candidate("move", {"left_j0": 0.2})),
                                observation(q0), now=10.0)
    assert [item.id for item in result.available] == ["move"]
    summary = result.available[0].summary
    assert summary["computed"]["eef_delta"] is None
    assert summary["local_checks"]["full_trajectory"] == "not_checked"
    assert summary["local_checks"]["interpolated_collision"] == "not_checked"
    assert summary["provenance"]["validation_mode"] == "yam"
    assert summary["provenance"]["mjcf_sha256"] is None
