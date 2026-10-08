"""Batch 04 trajectory safety gates with deterministic kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_jev.kinematics import YamKinematics

from conftest import location


def test_left_right_chunks_hold_inactive_arm(motion, q0) -> None:
    for side in ("left", "right"):
        base_y = 0.2 if side == "left" else -0.2
        result = motion.plan_pose(q0, side, [0.75, base_y, 0.2], np.eye(3))
        assert result.safe and 1 <= len(result.trajectory) <= motion.limits.max_steps
        inactive = slice(7, 14) if side == "left" else slice(0, 7)
        for step in result.trajectory:
            assert step.shape == (14,) and np.isfinite(step).all()
            np.testing.assert_array_equal(step[inactive], q0[inactive])
        chunk = result.chunk(f"test:{side}")
        assert len(chunk) == len(result.trajectory)
        assert chunk.meta["reobserve_after_chunk"] is True
        np.testing.assert_array_equal(chunk.actions[-1].data, result.trajectory[-1])


@pytest.mark.parametrize(("target", "reason"), [
    (lambda q: np.r_[np.pi + 0.01, q[1:]], "joint_limit"),
    (lambda q: np.r_[0.3, q[1:]], "step_too_large"),
    (lambda q: np.r_[q[:2], -0.2, q[3:]], "step_too_large"),
    (lambda q: np.r_[q[:6], -0.01, q[7:]], "invalid_gripper_encoding"),
    (lambda q: np.r_[q[:7], 0.01, q[8:]], "inactive_arm_changed"),
    (lambda q: np.zeros(13), "invalid_shape_or_value"),
])
def test_direct_trajectory_rejections(motion, q0, target, reason) -> None:
    result = motion.check_trajectory(q0, [target(q0)], "left")
    assert result.reason == reason and not result.trajectory


def test_table_collision_on_interpolated_state(motion, q0) -> None:
    # Endpoint is safe: a joint-space arc dips below the table in the middle.
    # This fixture's straight FK makes the endpoint itself unsafe, so the
    # sampled-path assertion is made with a synthetic nonlinear link trajectory.
    original = motion.kinematics.link_positions

    def bent(side, q):
        links = original(side, q)
        if side == "left":
            links[:, 2] -= 0.18 * np.sin(np.pi * q[0] / 0.1)
        return links

    motion.kinematics.link_positions = bent
    endpoint = q0.copy()
    endpoint[0] = 0.1
    assert motion._collision(endpoint, None) is None
    result = motion.check_trajectory(q0, [endpoint], "left")
    assert result.reason == "table_collision"


def test_arm_and_box_wall_collisions(motion, q0) -> None:
    original = motion.kinematics.link_positions

    def crossing(side, q):
        links = original(side, q)
        if side == "left":
            links[:, 1] -= 0.4 * np.sin(np.pi * q[0] / 0.1)
        return links

    motion.kinematics.link_positions = crossing
    endpoint = q0.copy()
    endpoint[0] = 0.1
    assert motion._collision(endpoint, None) is None
    assert motion.check_trajectory(q0, [endpoint], "left").reason == "arm_collision"
    motion.kinematics.link_positions = original

    # Place a synthetic box rim across the EEF path at safe table height.
    box = location("box", (0.5, 0.2, 0.25))
    q = q0.copy()
    q[2] = -0.06
    assert motion.check_trajectory(q, [q.copy()], "left", box=box).reason == "box_wall_collision"


def test_ik_failure_and_gripper_encoding(motion, synthetic_kinematics, q0) -> None:
    synthetic_kinematics.fail_ik = True
    result = motion.plan_pose(q0, "left", [0.7, 0.2, 0.2], np.eye(3))
    assert result.reason == "ik_non_converged"
    assert motion.plan_gripper(q0, "left", -0.0475).reason == "invalid_gripper_encoding"
    with pytest.raises(ValueError, match="filtered"):
        result.chunk("bad")


def test_real_batch03_kinematics_is_accepted(calibration, q0) -> None:
    from pathlib import Path
    from inspect_robots_jev.motion import MotionPlanner

    model = YamKinematics(Path(__file__).parent / "fixtures" / "tiny_yam.xml")
    planner = MotionPlanner(model, calibration)
    current = model.forward("left", q0)
    goal_q = q0.copy()
    goal_q[0] = 0.05
    goal = model.forward("left", goal_q)
    result = planner.plan_pose(q0, "left", goal.position_base_m, goal.rotation_base_eef)
    assert result.safe, result.reason
    assert model.link_positions("left", q0).shape == (6, 3)
