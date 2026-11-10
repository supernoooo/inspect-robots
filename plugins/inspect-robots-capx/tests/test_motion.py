"""Motion helpers preserve cursor semantics and the core delta backstop."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from inspect_robots.approver import DeltaLimitApprover
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Action
from inspect_robots_capx._motion import ArmLayout, MotionQueue


def _space() -> Box:
    return Box(
        shape=(3,),
        low=np.array([-1.0, -2.0, 0.0]),
        high=np.array([1.0, 2.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=("shoulder", "elbow", "gripper"),
        ),
    )


def _declaring_space() -> Box:
    return Box(
        shape=(3,),
        low=np.array([-1.0, -2.0, 0.0]),
        high=np.array([1.0, 2.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=("shoulder", "elbow", "gripper"),
            max_step=(0.02, None, 0.1),
        ),
    )


def _motion(
    *, control_hz: float = 10.0, max_speed_frac: float = 0.1, open_high: bool = True
) -> MotionQueue:
    return MotionQueue(
        _space(),
        control_hz=control_hz,
        max_speed_frac=max_speed_frac,
        gripper_index=2,
        gripper_open_is_high=open_high,
    )


def test_speed_fraction_and_control_rate_set_per_step_interpolation() -> None:
    motion = _motion(control_hz=10.0, max_speed_frac=0.1)
    start = np.array([0.0, 0.0, 1.0])
    motion.begin_turn(start)

    motion.move_to_joints(np.array([0.2, -0.4]))
    chunk = motion.take_chunk()

    points = [start, *(action.data for action in chunk.actions)]
    deltas = np.abs(np.diff(np.stack(points), axis=0))
    assert len(chunk.actions) == 11
    assert np.all(deltas <= np.array([0.02, 0.04, 0.01]))
    assert np.array_equal(chunk.actions[-1].data, np.array([0.2, -0.4, 1.0]))
    target = np.array([0.2, -0.4, 1.0])
    expected = np.stack(
        [start + (target - start) * fraction for fraction in np.linspace(1.0 / 11, 1.0, 11)]
    )
    expected[-1] = target
    assert np.array_equal(np.stack([action.data for action in chunk.actions]), expected)
    assert chunk.control_hz == 10.0


def test_low_control_rate_never_exceeds_delta_limit_approver_backstop() -> None:
    space = _space()
    motion = MotionQueue(
        space,
        control_hz=1.0,
        max_speed_frac=0.5,
        gripper_index=2,
    )
    start = np.array([-1.0, -2.0, 1.0])
    motion.begin_turn(start)
    motion.move_to_joints(np.array([1.0, 2.0]))
    chunk = motion.take_chunk()

    approver = DeltaLimitApprover(space)
    store: dict[str, object] = {}
    approver.review(Action(data=start), store)
    for action in chunk.actions:
        assert approver.review(action, store) is action

    points = [start, *(action.data for action in chunk.actions)]
    per_step = np.abs(np.diff(np.stack(points), axis=0))
    native_backstop = 0.05 * (space.high - space.low)  # type: ignore[operator]
    assert np.all(per_step <= native_backstop)


def test_declared_step_chunks_traverse_at_pace_and_pass_default_approver() -> None:
    space = _declaring_space()
    motion = MotionQueue(
        space,
        control_hz=10.0,
        max_speed_frac=0.1,
        gripper_index=2,
    )
    start = np.array([0.0, 0.0, 1.0])
    motion.begin_turn(start)
    motion.close_gripper()
    chunk = motion.take_chunk()

    assert len(chunk.actions) == 11
    points = np.stack([start, *(action.data for action in chunk.actions)])
    assert np.all(np.abs(np.diff(points[:, 2])) < 0.1)
    assert chunk.actions[-1].data[2] == 0.0

    approver = DeltaLimitApprover(space)
    store: dict[str, object] = {}
    approver.review(Action(data=start), store)
    for action in chunk.actions:
        assert approver.review(action, store) is action


@pytest.mark.parametrize(
    ("max_speed_frac", "expected_steps"),
    [(0.05, 21), (0.2, 11)],
)
def test_declared_step_pacing_scales_down_but_not_up(
    max_speed_frac: float,
    expected_steps: int,
) -> None:
    motion = MotionQueue(
        _declaring_space(),
        control_hz=10.0,
        max_speed_frac=max_speed_frac,
        gripper_index=2,
    )
    motion.begin_turn(np.array([0.0, 0.0, 1.0]))
    motion.close_gripper()

    assert len(motion.take_chunk().actions) == expected_steps


def test_cursor_chains_across_arm_and_gripper_calls() -> None:
    motion = _motion()
    motion.begin_turn(np.array([0.0, 0.0, 1.0]))

    motion.move_to_joints(np.array([0.1, 0.2]))
    arm_endpoint = motion.cursor
    motion.close_gripper()
    closed_endpoint = motion.cursor
    motion.open_gripper()
    open_endpoint = motion.cursor
    chunk = motion.take_chunk(code="move_to_joints(...)")

    assert arm_endpoint is not None and np.array_equal(arm_endpoint, [0.1, 0.2, 1.0])
    assert closed_endpoint is not None and np.array_equal(closed_endpoint, [0.1, 0.2, 0.0])
    assert open_endpoint is not None and np.array_equal(open_endpoint, [0.1, 0.2, 1.0])
    assert np.array_equal(chunk.actions[-1].data, open_endpoint)
    assert chunk.actions[0].meta["code"] == "move_to_joints(...)"
    assert all(
        np.max(np.abs(right.data - left.data)) <= 0.04 for left, right in pairwise(chunk.actions)
    )


def test_hold_chunk_has_one_full_state_action_and_stop_metadata() -> None:
    motion = _motion()
    state = np.array([0.3, -0.2, 0.8])

    chunk = motion.hold_chunk(state, stop_reason="FINISH", inference_latency_s=0.25)

    assert len(chunk.actions) == 1
    assert np.array_equal(chunk.actions[0].data, state)
    assert chunk.actions[0].meta == {"request_stop": True, "stop_reason": "FINISH"}
    assert chunk.inference_latency_s == 0.25


def test_gripper_values_come_from_box_bounds_with_configurable_polarity() -> None:
    normal = _motion(open_high=True)
    inverted = _motion(open_high=False)

    assert normal.gripper_open_value == 1.0
    assert normal.gripper_closed_value == 0.0
    assert inverted.gripper_open_value == 0.0
    assert inverted.gripper_closed_value == 1.0


def test_offset_bounds_with_coarse_float_grid_are_rejected() -> None:
    space = Box(
        shape=(2,),
        low=np.array([1e16, 0.0]),
        high=np.array([1e16 + 2.0, 1.0]),
        semantics=ActionSemantics("joint_pos", gripper="continuous", dim_labels=("j0", "gripper")),
    )

    with pytest.raises(ValueError, match="too coarse"):
        MotionQueue(space, control_hz=10.0, max_speed_frac=0.1, gripper_index=1)


def test_declared_step_is_the_float_spacing_guard_budget() -> None:
    space = Box(
        shape=(2,),
        low=np.array([1e9, 0.0]),
        high=np.array([1e9 + 100.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=("j0", "gripper"),
            max_step=(0.1, None),
        ),
    )

    with pytest.raises(ValueError, match="too coarse"):
        MotionQueue(space, control_hz=10.0, max_speed_frac=0.1, gripper_index=1)


def test_declared_step_pacing_precedes_movable_zero_limit_guard() -> None:
    space = Box(
        shape=(2,),
        low=np.array([0.0, 0.0]),
        high=np.array([1e-300, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=("j0", "gripper"),
            max_step=(1e-300, None),
        ),
    )

    with pytest.raises(ValueError, match="underflows the per-step limit"):
        MotionQueue(
            space,
            control_hz=10.0,
            max_speed_frac=1e-300,
            gripper_index=1,
        )


def test_arm_cursor_tracks_queued_endpoint_and_requires_a_seed() -> None:
    space = Box(
        shape=(3,),
        low=np.array([-1.0, -1.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        semantics=ActionSemantics("joint_pos", gripper="continuous"),
    )
    motion = MotionQueue(space, control_hz=10.0, max_speed_frac=0.1, gripper_index=1)
    with pytest.raises(RuntimeError):
        _ = motion.arm_cursor
    motion.begin_turn(np.array([0.1, 0.5, 0.2]))
    motion.move_to_joints(np.array([0.3, 0.4]))

    assert motion.arm_cursor.tolist() == [0.3, 0.4]


def test_gripper_speed_frac_ramps_the_gripper_faster_within_the_backstop() -> None:
    space = Box(
        shape=(2,),
        low=np.array([-100.0, 0.0]),
        high=np.array([100.0, 100.0]),
        semantics=ActionSemantics("joint_pos", gripper="continuous"),
    )
    slow = MotionQueue(space, control_hz=10.0, max_speed_frac=0.05, gripper_index=1)
    fast = MotionQueue(
        space, control_hz=10.0, max_speed_frac=0.05, gripper_index=1, gripper_speed_frac=0.5
    )
    lengths = []
    for motion in (slow, fast):
        motion.begin_turn(np.array([0.0, 0.0]))
        motion.open_gripper()
        motion.move_to_joints(np.array([10.0]))
        chunk = motion.take_chunk()
        steps = np.diff(np.array([action.data for action in chunk.actions]), axis=0)
        assert np.abs(steps[:, 1]).max() <= 0.05 * 100 + 1e-9  # DeltaLimit backstop
        assert np.abs(steps[:, 0]).max() <= 0.05 / 10 * 200 + 1e-9  # arm speed unchanged
        grip = np.array([action.data[1] for action in chunk.actions])
        lengths.append(int(np.argmax(grip >= 100.0)) + 1)
    assert lengths == [201, 21]  # one extra step from the per-step headroom


def test_gripper_speed_frac_scales_declared_max_step_and_rejects_bad_values() -> None:
    space = Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=ActionSemantics("joint_pos", gripper="continuous", max_step=(0.01, 0.02)),
    )
    motion = MotionQueue(
        space, control_hz=10.0, max_speed_frac=0.05, gripper_index=1, gripper_speed_frac=0.2
    )
    assert motion._step_limits.tolist() == pytest.approx([0.005, 0.02])
    with pytest.raises(ValueError, match="gripper_speed_frac"):
        MotionQueue(
            space, control_hz=10.0, max_speed_frac=0.05, gripper_index=1, gripper_speed_frac=0.0
        )


_TWO_ARMS = (
    ArmLayout(name="left", joint_indices=(0, 1), gripper_index=2),
    ArmLayout(name="right", joint_indices=(3, 4), gripper_index=5),
)


def _two_arm_motion(*, gripper_speed_frac: float | None = None) -> MotionQueue:
    space = Box(
        shape=(6,),
        low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=(
                "left_j0",
                "left_j1",
                "left_gripper",
                "right_j0",
                "right_j1",
                "right_gripper",
            ),
            max_step=(0.02, 0.02, 0.1, 0.02, 0.02, 0.1),
        ),
    )
    return MotionQueue(
        space,
        control_hz=10.0,
        max_speed_frac=0.1,
        arms=_TWO_ARMS,
        gripper_speed_frac=gripper_speed_frac,
    )


def test_multi_arm_moves_one_arm_and_holds_the_other_and_both_grippers() -> None:
    motion = _two_arm_motion()
    start = np.array([0.0, 0.0, 1.0, 0.5, -0.5, 0.4])
    motion.begin_turn(start)

    motion.move_to_joints(np.array([0.1, -0.1]), arm="left")
    chunk = motion.take_chunk()

    data = np.stack([action.data for action in chunk.actions])
    assert len(chunk.actions) == 6
    assert np.array_equal(data[-1], [0.1, -0.1, 1.0, 0.5, -0.5, 0.4])
    assert np.all(data[:, 2:] == start[2:])
    assert motion.arms == ("left", "right")
    assert motion.joint_count("right") == 2


def test_move_arms_is_one_synchronized_segment_and_unnamed_arms_hold() -> None:
    motion = _two_arm_motion()
    motion.begin_turn(np.zeros(6))

    motion.move_arms({"left": np.array([0.1, 0.0]), "right": np.array([0.0, -0.2])})
    chunk = motion.take_chunk()

    data = np.stack([action.data for action in chunk.actions])
    # The slower right arm (0.2 rad at 0.02/step) paces both arms: they finish together.
    assert len(chunk.actions) == 11
    assert np.array_equal(data[-1], [0.1, 0.0, 0.0, 0.0, -0.2, 0.0])
    assert np.all(np.diff(data[:, 0]) > 0) and np.all(np.diff(data[:, 4]) < 0)
    motion.begin_turn(np.zeros(6))
    motion.move_arms({"right": np.array([0.04, 0.0])})
    assert np.all(np.stack([a.data for a in motion.take_chunk().actions])[:, :3] == 0)


def test_multi_arm_grippers_and_ik_seeds_are_per_arm() -> None:
    motion = _two_arm_motion()
    motion.begin_turn(np.array([0.1, 0.2, 1.0, 0.3, 0.4, 1.0]))

    motion.close_gripper(arm="right")
    motion.move_to_joints(np.array([0.32, 0.4]), arm="right")

    assert motion.cursor is not None
    assert motion.cursor.tolist() == pytest.approx([0.1, 0.2, 1.0, 0.32, 0.4, 0.0])
    assert motion.arm_joints("left").tolist() == pytest.approx([0.1, 0.2])
    assert motion.arm_joints("right").tolist() == pytest.approx([0.32, 0.4])
    motion.open_gripper(arm="right")
    assert motion.cursor[5] == 1.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda m: m.move_to_joints(np.zeros(2)), "name the arm"),
        (lambda m: m.move_to_joints(np.zeros(2), arm="middle"), "unknown arm 'middle'"),
        (lambda m: m.move_to_joints(np.zeros(3), arm="left"), "expects 2 'left' arm joints"),
        (lambda m: m.open_gripper(), "name the arm"),
        (lambda m: m.move_arms({}), "at least one arm target"),
        (lambda m: m.move_arms({"left": np.array([np.nan, 0.0])}), "non-finite"),
        (lambda m: m.arm_dof, "name the arm"),
    ],
    ids=["no_arm", "unknown_arm", "wrong_size", "gripper_no_arm", "empty", "nan", "arm_dof"],
)
def test_multi_arm_helpers_require_a_known_arm(call: object, message: str) -> None:
    motion = _two_arm_motion()
    motion.begin_turn(np.zeros(6))

    with pytest.raises(ValueError, match=message):
        call(motion)  # type: ignore[operator]
    assert not motion.has_actions()


def test_single_arm_helpers_reject_an_arm_name_and_move_arms() -> None:
    motion = _motion()
    motion.begin_turn(np.array([0.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="single arm"):
        motion.move_to_joints(np.zeros(2), arm="left")
    with pytest.raises(ValueError, match="multi-arm embodiment"):
        motion.move_arms({"left": np.zeros(2)})
    assert motion.arms == ()


def test_multi_arm_gripper_speed_frac_applies_to_every_gripper() -> None:
    motion = _two_arm_motion(gripper_speed_frac=0.05)

    assert motion._step_limits.tolist() == pytest.approx([0.02, 0.02, 0.05, 0.02, 0.02, 0.05])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gripper_index": 2, "arms": _TWO_ARMS}, "either gripper_index"),
        ({}, "either gripper_index"),
        ({"arms": (_TWO_ARMS[0],)}, "exactly once"),
        ({"arms": (_TWO_ARMS[0], _TWO_ARMS[0])}, "declared twice"),
        ({"arms": (_TWO_ARMS[0], ArmLayout("right", (), 5))}, "no joints"),
    ],
    ids=["both", "neither", "uncovered", "duplicate", "jointless"],
)
def test_arm_layouts_are_validated(kwargs: dict[str, object], message: str) -> None:
    space = Box(
        shape=(6,), low=-np.ones(6), high=np.ones(6), semantics=ActionSemantics("joint_pos")
    )

    with pytest.raises(ValueError, match=message):
        MotionQueue(space, control_hz=10.0, max_speed_frac=0.1, **kwargs)  # type: ignore[arg-type]
