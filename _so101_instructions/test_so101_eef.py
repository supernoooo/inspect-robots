"""Offline SO-101 EEF adapter tests; every motor command goes to a fake driver."""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _run_so101 import configured_so101_eef
from _so101_eef import SO101EEFEmbodiment, load_kinematics
from _so101_motion import CLIArm
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import build_toolset
from inspect_robots_so101 import SOArmConfig, packing

from inspect_robots import Action, Scene
from inspect_robots.errors import ConfigError

MODEL = Path(__file__).resolve().parent.parent / ".so101/models/SO101/so101_new_calib.urdf"
START = np.array([0.0, -50.0, 60.0, 20.0, 0.0, 50.0])
LOGGED_START = np.array([-21.1429, -102.1538, 96.2637, 68.8791, -5.7582, 2.3537])
RUN004_START = np.array([-14.8132, -102.0659, 96.2637, 68.7033, -5.7582, 0.8742])
RUN005_START = np.array([-6.8132, -102.1538, 96.0879, 58.6813, -5.5824, 1.0087])
RUN014_START = np.array([8.7912, -102.2418, 96.8791, 55.2967, 8.4835, 1.345])


class FakeDriver:
    def __init__(self, pose: np.ndarray):
        self.pose = pose.copy()
        self.commands: list[np.ndarray] = []

    def get_observation(self):
        return packing.to_action_dict(self.pose)

    def send_action(self, action):
        self.pose = packing.from_obs_dict(action)
        self.commands.append(self.pose.copy())
        return packing.to_action_dict(self.pose)

    def disconnect(self):
        pass


class StalledDriver(FakeDriver):
    def send_action(self, action):
        self.commands.append(packing.from_obs_dict(action).copy())
        return action


class PartialArmDriver(FakeDriver):
    """Model a follower that reaches only a fraction of each arm command."""

    def send_action(self, action):
        command = packing.from_obs_dict(action)
        self.pose[:5] += (command[:5] - self.pose[:5]) * 0.05
        self.pose[5] = command[5]
        self.commands.append(command.copy())
        return action


class DelayedGripperDriver(FakeDriver):
    def __init__(self, pose: np.ndarray):
        super().__init__(pose)
        self.gripper_target: float | None = None

    def send_action(self, action):
        command = packing.from_obs_dict(action)
        self.pose[:5] = command[:5]
        self.gripper_target = float(command[5])
        self.commands.append(command.copy())
        return action

    def get_observation(self):
        if self.gripper_target is not None:
            difference = self.gripper_target - self.pose[5]
            self.pose[5] += np.clip(difference, -1.0, 1.0)
        return super().get_observation()


class LaggingGripperDriver(FakeDriver):
    """Follow each open command with a small repeatable servo offset."""

    def send_action(self, action):
        command = packing.from_obs_dict(action)
        previous = self.pose[5]
        self.pose = command.copy()
        if command[5] > previous:
            self.pose[5] = max(previous, command[5] - 0.7)
        self.commands.append(command.copy())
        return action


class DriftingHoldDriver(FakeDriver):
    """Model one encoder tick of shoulder offset during a gripper-only step."""

    def send_action(self, action):
        accepted = super().send_action(action)
        self.pose[0] += 0.2637362637362637
        return accepted


@pytest.fixture
def rig():
    if not MODEL.is_file():
        pytest.skip("run _so101_instructions/_fetch_so101_model.py for the offline model test")
    driver = FakeDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, return_home=True,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    yield wrapped, driver
    wrapped.close()


def test_model_joint_order_and_fk_ik_round_trip():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    joints = {joint.get("name"): joint for joint in ET.parse(MODEL).getroot().findall("joint")}
    assert set(packing.MOTORS) <= joints.keys()
    link = "gripper_frame_link"
    chain = []
    while link != "base_link":
        joint = next(j for j in joints.values() if j.find("child").get("link") == link)
        if joint.get("type") == "revolute":
            chain.append(joint.get("name"))
            assert joint.find("axis").get("xyz") == "0 0 1"
            bounds = joint.find("limit")
            assert -3.2 < float(bounds.get("lower")) < float(bounds.get("upper")) < 3.2
        link = joint.find("parent").get("link")
    assert chain[::-1] == list(packing.MOTORS[:5])
    assert packing.from_obs_dict(packing.to_action_dict(START)) == pytest.approx(START)
    model = load_kinematics(MODEL)
    for q in (np.zeros(5), START[:5], np.array([15.0, -40.0, 50.0, 10.0, 0.0])):
        actual = model.forward_kinematics(q)
        target = actual.copy()
        target[2, 3] += 0.002
        solved = model.inverse_kinematics(q, target, orientation_weight=0.01)
        achieved = model.forward_kinematics(solved)
        assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 0.0008
        assert np.linalg.norm(achieved[:3, :3] - target[:3, :3]) < 0.004


def test_ik_uses_calibrated_limits_instead_of_generic_urdf_limits():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    low = (-108.0, -103.38, -97.32, -101.32, -180.0, 0.0)
    high = (108.0, 103.38, 97.32, 101.32, 180.0, 100.0)
    model = load_kinematics(MODEL, joint_low=low, joint_high=high)
    for index, name in enumerate(packing.MOTORS[:5]):
        assert np.rad2deg(model.robot.get_joint_limits(name)) == pytest.approx(
            [low[index], high[index]]
        )


def test_configured_eef_tracking_thresholds_reach_adapter(tmp_path):
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    wrapped = configured_so101_eef(
        camera_configs={}, robot_id="missing", calibration_dir=str(tmp_path),
        urdf_path=MODEL, tracking_position_tolerance_m=0.003,
        tracking_orientation_tolerance_deg=1.0,
        tracking_joint_tolerance_deg=0.75,
    )
    try:
        assert wrapped._tracking_position_tolerance_m == pytest.approx(0.003)
        assert np.rad2deg(wrapped._tracking_orientation_tolerance_rad) == pytest.approx(1.0)
        assert wrapped._tracking_joint_tolerance_deg == pytest.approx(0.75)
    finally:
        wrapped.close()
    with pytest.raises(ConfigError, match="tracking_position_tolerance_m"):
        configured_so101_eef(
            camera_configs={}, robot_id="missing", calibration_dir=str(tmp_path),
            urdf_path=MODEL, tracking_position_tolerance_m=0,
        )
    with pytest.raises(ConfigError, match="tracking_joint_tolerance_deg"):
        configured_so101_eef(
            camera_configs={}, robot_id="missing", calibration_dir=str(tmp_path),
            urdf_path=MODEL, tracking_joint_tolerance_deg=0,
        )


def test_logged_eef_step_sends_exact_ik_joint_solution():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    start = np.array([-2.989, -102.5934, 96.4396, 54.3297, -6.3736, 1.5467])
    driver = FakeDriver(start)
    config = SOArmConfig(
        cameras=(), camera_configs={}, control_hz=10.0, settle_tolerance=None,
        joint_low=(-118.77, -104.0, -97.67, -101.36, -180.0, 0.0),
        joint_high=(118.77, 104.0, 97.67, 101.36, 180.0, 100.0),
    )
    arm = CLIArm(
        config, capture_home=False, return_home=False,
        driver_factory=lambda _: driver,
        operator=SimpleNamespace(wait_ready=lambda *args: None),
        poll_end=lambda: False, sleep_fn=lambda _: None,
    )
    kinematics = load_kinematics(
        MODEL, joint_low=config.joint_low, joint_high=config.joint_high
    )
    wrapped = SO101EEFEmbodiment(arm, kinematics)
    try:
        initial = wrapped.reset(Scene("run002", "lift"))
        delta = np.array([0.002, 0, 0, 0, 0, 0, 0])
        desired = kinematics.forward_kinematics(start[:5]).copy()
        desired[0, 3] += delta[0]
        solved = kinematics.inverse_kinematics(
            start[:5], desired, position_weight=1.0, orientation_weight=0.01
        )
        result = wrapped.step(Action(data=delta, meta={"chunk_final": True}))
        assert len(driver.commands) == 1
        np.testing.assert_allclose(driver.commands[0][:5], solved, atol=1e-10)
        np.testing.assert_allclose(result.info["eef_commanded_joint_pos"][:5], solved)
        assert np.max(np.abs(solved - start[:5])) > 3.0
        assert result.observation.state["eef_state"][0] > initial.state["eef_state"][0]
    finally:
        wrapped.close()


def _eef_target_call(observation, *, changes: dict[str, float], steps: int):
    state = observation.state["eef_state"]
    quat = observation.state["eef_quat_wxyz"]
    targets = dict(zip(("x", "y", "z"), state[:3], strict=True))
    targets.update(dict(zip(("qw", "qx", "qy", "qz"), quat, strict=True)))
    targets.update(changes)
    return json.dumps({"targets": targets, "steps": steps, "note": "test motion"})


def test_agent_absolute_eef_target_reaches_adapter(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "move a little"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    assert toolset.schemas()[0]["function"]["name"] == "move_eef"
    assert toolset.state_labels() == (
        "eef_state", ("x", "y", "z", "rx", "ry", "rz", "gripper")
    )
    result = toolset.execute(
        ToolCall(id="test", name="move_eef", arguments=_eef_target_call(
            observation, changes={"z": float(observation.state["eef_state"][2] + 0.002)},
            steps=2,
        )),
        observation,
    )
    assert result.chunk is not None
    for action in result.chunk.actions:
        output = wrapped.step(action)
    assert len(driver.commands) == len(result.chunk.actions)
    assert output.observation.state["eef_state"][2] > observation.state["eef_state"][2]
    assert np.max(np.abs(driver.commands[-1][:5] - START[:5])) <= 1.0


def test_agent_bounds_large_diagonal_step(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "diagonal"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    result = toolset.execute(
        ToolCall(id="diagonal", name="move_eef", arguments=_eef_target_call(
            observation,
            changes={
                "x": float(observation.state["eef_state"][0] + 0.002),
                "z": float(observation.state["eef_state"][2] + 0.002),
            },
            steps=1,
        )),
        observation,
    )
    assert result.chunk is not None
    assert len(result.chunk.actions) == 2
    assert np.linalg.norm(result.chunk.actions[0].data[:3]) <= 0.002
    np.testing.assert_allclose(sum(action.data[:3] for action in result.chunk.actions),
                               [0.002, 0, 0.002])
    assert "target shortened" not in result.note
    assert driver.commands == []


def test_logged_diagonal_targets_advance_within_total_translation_limit(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "diagonal replay"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    state = observation.state["eef_state"]
    description = toolset.schemas()[0]["function"]["description"]
    assert "TOTAL 3D translation length" in description
    for dx, dz in ((0.0100, 0.0100), (0.0078, 0.0080)):
        call = json.dumps({"targets": {"x": float(state[0] + dx),
                                       "z": float(state[2] + dz)},
                           "steps": 5, "note": "Replay a long diagonal target"})
        result = toolset.execute(ToolCall(id="logged", name="move_eef", arguments=call),
                                 observation)
        assert result.chunk is not None
        assert len(result.chunk.actions) > 5
        assert np.linalg.norm(result.chunk.actions[0].data[:3]) <= 0.002
        np.testing.assert_allclose(
            sum(action.data[:3] for action in result.chunk.actions), [dx, 0, dz]
        )
        assert "target shortened" not in result.note
    assert driver.commands == []


def test_large_gripper_target_reaches_full_target_and_keeps_arm_still(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "open gripper"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    call = json.dumps({"targets": {"gripper": 99.0}, "steps": 5,
                       "note": "Open toward the requested gap"})
    result = toolset.execute(ToolCall(id="grip", name="move_eef", arguments=call),
                             observation)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 10
    for action in result.chunk.actions:
        assert action.data[:6] == pytest.approx(np.zeros(6))
        assert 0 < action.data[6] <= 5.0
        wrapped.step(action)
    assert driver.pose[:5] == pytest.approx(START[:5])
    assert driver.pose[5] == pytest.approx(99.0)


def test_agent_eef_omitted_pose_holds_exact_measurement(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "open gripper"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    target = float(observation.state["eef_state"][6] + 1.0)
    call = json.dumps({"targets": {"gripper": target}, "steps": 2,
                       "note": "Open the gripper only"})
    result = toolset.execute(ToolCall(id="grip", name="move_eef", arguments=call),
                             observation)
    assert result.chunk is not None
    for action in result.chunk.actions:
        assert action.data[:6] == pytest.approx(np.zeros(6))
        wrapped.step(action)
    assert driver.pose[:5] == pytest.approx(START[:5])
    assert driver.pose[5] == pytest.approx(target)


def test_agent_eef_requires_complete_quaternion_when_provided(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "partial orientation"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    call = json.dumps({"targets": {"z": 0.01, "qw": 1.0}, "steps": 1,
                       "note": "Try partial orientation"})
    result = toolset.execute(ToolCall(id="partial", name="move_eef", arguments=call),
                             observation)
    assert result.error is not None and "all of qw,qx,qy,qz" in result.error
    assert driver.commands == []


def test_agent_rejects_full_target_that_exceeds_horizon_without_motion(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "open gripper"))
    space = replace(wrapped.info.action_space, semantics=replace(
        wrapped.info.action_space.semantics, max_chunk_steps=6,
    ))
    toolset = build_toolset(space, wrapped.info.observation_space, wrapped.info.control_hz)
    assert "At most 6 control steps per call" in toolset.schemas()[0]["function"]["description"]
    too_long = toolset.execute(
        ToolCall(id="long", name="move_eef", arguments=_eef_target_call(
            observation, changes={"gripper": 99.0}, steps=1,
        )),
        observation,
    )
    assert too_long.error is not None and "full EEF target exceeds the 6-step" in too_long.error
    assert driver.commands == []
    short = toolset.execute(
        ToolCall(id="short", name="move_eef", arguments=_eef_target_call(
            observation,
            changes={"gripper": float(observation.state["eef_state"][6] + 2.0)},
            steps=5,
        )),
        observation,
    )
    assert short.chunk is not None
    assert len(short.chunk.actions) <= 5
    for action in short.chunk.actions:
        wrapped.step(action)
    assert driver.pose[:5] == pytest.approx(START[:5])
    assert driver.pose[5] == pytest.approx(START[5] + 2.0)


def test_agent_eef_target_quaternion_is_base_frame_rotation(rig):
    wrapped, _ = rig
    observation = wrapped.reset(Scene("eef", "turn"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    current = observation.state["eef_quat_wxyz"]
    half = 0.0025
    target_quat = np.array([
        np.cos(half) * current[0] - np.sin(half) * current[3],
        np.cos(half) * current[1] - np.sin(half) * current[2],
        np.cos(half) * current[2] + np.sin(half) * current[1],
        np.cos(half) * current[3] + np.sin(half) * current[0],
    ])
    call = _eef_target_call(observation, changes=dict(zip(
        ("qw", "qx", "qy", "qz"), target_quat, strict=True,
    )), steps=1)
    result = toolset.execute(ToolCall(id="rotate", name="move_eef", arguments=call), observation)
    assert result.chunk is not None
    assert result.chunk.actions[0].data[3:6] == pytest.approx([0.0, 0.0, 0.005], abs=1e-8)


def test_agent_eef_target_rejects_non_unit_quaternion(rig):
    wrapped, driver = rig
    observation = wrapped.reset(Scene("eef", "invalid orientation"))
    toolset = build_toolset(wrapped.info.action_space, wrapped.info.observation_space,
                            wrapped.info.control_hz)
    call = _eef_target_call(observation, changes={"qw": 2.0}, steps=1)
    result = toolset.execute(ToolCall(id="bad", name="move_eef", arguments=call), observation)
    assert result.error is not None and "unit quaternion" in result.error
    assert driver.commands == []


def test_gripper_and_rotation_deltas(rig):
    wrapped, driver = rig
    first = wrapped.reset(Scene("eef", "turn and open"))
    gripper = np.zeros(7)
    gripper[6] = 0.5
    result = wrapped.step(Action(data=gripper))
    assert driver.pose[5] == START[5] + 0.5
    assert result.observation.state["eef_state"][6] == START[5] + 0.5
    rotation = np.zeros(7)
    rotation[5] = 0.005
    later = wrapped.step(Action(data=rotation))
    assert not np.allclose(later.observation.state["eef_state"][3:6],
                           first.state["eef_state"][3:6])


def test_logged_gripper_only_move_never_calls_ik_or_moves_arm():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = FakeDriver(LOGGED_START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "open gripper"))
        wrapped._kinematics = SimpleNamespace(
            forward_kinematics=wrapped._kinematics.forward_kinematics,
            inverse_kinematics=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("gripper-only movement must not invoke IK")
            ),
        )
        delta = np.zeros(7)
        delta[6] = 40.0 / 81
        for _ in range(81):
            wrapped.step(Action(data=delta))
        assert len(driver.commands) == 81
        assert all(np.array_equal(command[:5], LOGGED_START[:5])
                   for command in driver.commands)
        assert driver.pose[5] == pytest.approx(LOGGED_START[5] + 40.0)
    finally:
        wrapped.close()


def test_stalled_gripper_records_feedback_and_continues():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = StalledDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, return_home=True,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "open gripper"))
        delta = np.zeros(7)
        delta[6] = 0.5
        for _ in range(3):
            result = wrapped.step(Action(data=delta))
        assert len(driver.commands) == 3
        assert "gripper_no_motion" in result.info["eef_tracking_warnings"]
        assert result.info["eef_target_gripper_error"] > 1
        assert not wrapped._arm._motion_failed
    finally:
        wrapped.close()


def test_gripper_waits_for_feedback_after_five_unit_command():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = DelayedGripperDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "open gripper"))
        result = wrapped.step(Action(data=np.array([0, 0, 0, 0, 0, 0, 5.0])))
        assert driver.commands[-1][5] == pytest.approx(55.0)
        assert result.observation.state["joint_pos"][5] >= 54.0
        assert not result.terminated
    finally:
        wrapped.close()


def test_gripper_tracking_uses_sent_command_when_agent_target_is_ahead():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = LaggingGripperDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "open gripper"))
        delta = np.array([0, 0, 0, 0, 0, 0, 5.0])
        first = wrapped.step(Action(data=delta))
        for index in range(1, 5):
            result = wrapped.step(Action(data=delta, meta={"chunk_final": index == 4}))
        assert len(driver.commands) == 5
        assert result.observation.state["joint_pos"][5] == pytest.approx(START[5] + 21.5)
        assert first.info["eef_target_gripper_error"] == pytest.approx(0.7)
        assert result.info["eef_target_gripper_error"] == pytest.approx(3.5)
        assert not result.terminated
    finally:
        wrapped.close()


def test_lagging_gripper_still_receives_full_interpolated_target():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = LaggingGripperDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "open gripper"))
        delta = np.array([0, 0, 0, 0, 0, 0, 1.0])
        for index in range(5):
            result = wrapped.step(Action(data=delta, meta={"chunk_final": index == 4}))
        assert [command[5] for command in driver.commands] == pytest.approx(
            [51.0, 52.0, 53.0, 54.0, 55.0]
        )
        assert result.observation.state["joint_pos"][5] == pytest.approx(54.3)
        assert result.info["eef_target_gripper_error"] == pytest.approx(0.7)
    finally:
        wrapped.close()


def test_stalled_gripper_commands_stay_within_measured_step_limit():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = StalledDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None, settle_timeout_s=0.01)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "open gripper"))
        delta = np.array([0, 0, 0, 0, 0, 0, 1.0])
        for index in range(10):
            result = wrapped.step(Action(data=delta, meta={"chunk_final": index == 9}))
        assert max(command[5] for command in driver.commands) == pytest.approx(55.0)
        assert result.info["eef_target_gripper_error"] == pytest.approx(10.0)
    finally:
        wrapped.close()


def test_gripper_only_step_accepts_logged_one_tick_shoulder_drift():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    pose = np.array([3.2527, -103.033, 96.8791, 54.4176, -1.3626, 1.2105])
    driver = DriftingHoldDriver(pose)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=0.2)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("run009", "open gripper"))
        result = wrapped.step(Action(data=np.array([0, 0, 0, 0, 0, 0, 5.0])))
        assert result.info["settled"] is False
        assert result.info["settle_residual"] == pytest.approx(0.2637362637362637)
        assert result.observation.state["joint_pos"][0] == pytest.approx(
            pose[0] + 0.2637362637362637
        )
        assert not result.terminated
    finally:
        wrapped.close()


def test_stalled_cartesian_motion_records_timeout_and_continues():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = StalledDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=0.2)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    try:
        wrapped.reset(Scene("eef", "move forward"))
        delta = np.zeros(7)
        delta[0] = 0.002
        result = wrapped.step(Action(data=delta))
        assert len(driver.commands) == 1
        assert result.info["settled"] is False
        assert "settle_timeout" in result.info["eef_tracking_warnings"]
        assert "arm_no_motion" in result.info["eef_tracking_warnings"]
        assert result.info["eef_tracking_position_error_m"] > 0
        assert not wrapped._arm._motion_failed
        wrapped.step(Action(data=np.zeros(7), meta={"chunk_final": True}))
        assert len(driver.commands) == 2
    finally:
        wrapped.close()


def test_cartesian_tracking_error_is_feedback_not_a_step_failure(
    caplog: pytest.LogCaptureFixture,
):
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = PartialArmDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=1.0)
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(
        arm, load_kinematics(MODEL), tracking_position_tolerance_m=0.0005
    )
    try:
        wrapped.reset(Scene("eef", "lift"))
        result = wrapped.step(Action(data=np.array([0, 0, 0.002, 0, 0, 0, 0])))
        assert not result.terminated
        assert result.info["eef_tracking_position_error_m"] > 0.0005
        assert "eef_position_tracking_error" in result.info["eef_tracking_warnings"]
        assert result.info["eef_tracking_worst_joint"] in packing.MOTORS[:5]
        assert not wrapped._arm._motion_failed
        assert not any(
            "SO-101 EEF tracking feedback" in record.message for record in caplog.records
        )
    finally:
        wrapped.close()


def test_run014_small_ik_orientation_residual_is_feedback():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = FakeDriver(RUN014_START)
    config = SOArmConfig(
        cameras=(), camera_configs={}, control_hz=10.0, settle_tolerance=1.0,
        joint_low=(-118.77, -104.0, -97.67, -101.36, -180.0, 0.0),
        joint_high=(118.77, 104.0, 97.67, 101.36, 180.0, 100.0),
    )
    arm = CLIArm(config, capture_home=False, return_home=False,
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    kinematics = load_kinematics(
        MODEL, joint_low=config.joint_low, joint_high=config.joint_high
    )
    wrapped = SO101EEFEmbodiment(arm, kinematics)
    try:
        initial = wrapped.reset(Scene("run014", "approach"))
        position = initial.state["eef_state"][:3]
        delta = np.r_[([0.1877, -0.0173, 0.016] - position) / 5,
                      [0.0, 0.0, 0.0, 5.0]]
        result = wrapped.step(Action(data=delta))
        assert not result.terminated
        assert result.info["eef_ik_position_error_m"] < 0.0008
        assert np.rad2deg(result.info["eef_ik_orientation_error_rad"]) == pytest.approx(
            0.28, abs=0.02
        )
        assert "ik_orientation_residual" in result.info["eef_tracking_warnings"]
        for index in range(1, 5):
            result = wrapped.step(Action(data=delta, meta={"chunk_final": index == 4}))
        assert len(driver.commands) == 5
        assert result.observation.state["eef_state"][0] > position[0]
        assert result.info["eef_ik_position_error_m"] > 0.0008
        assert not wrapped._arm._motion_failed
    finally:
        wrapped.close()


def test_tracking_arm_and_gripper_complete_chunk(rig):
    wrapped, driver = rig
    initial = wrapped.reset(Scene("eef", "move and open"))
    delta = np.zeros(7)
    delta[0] = 0.00025
    delta[6] = 0.49
    for step in range(10):
        meta = {"chunk_final": True} if step == 9 else {}
        result = wrapped.step(Action(data=delta, meta=meta))
    assert len(driver.commands) == 10
    assert result.observation.state["eef_state"][0] == pytest.approx(
        initial.state["eef_state"][0] + 0.0025, abs=0.0015
    )
    assert driver.pose[5] == pytest.approx(START[5] + 4.9)
    assert wrapped._expected_pose is None


def test_run004_small_eef_step_can_use_more_than_one_degree_of_joint_motion():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = FakeDriver(RUN004_START)
    config = SOArmConfig(
        cameras=(), camera_configs={}, control_hz=10.0, settle_tolerance=None,
        joint_low=(-108.0, -103.38, -97.32, -101.32, -180.0, 0.0),
        joint_high=(108.0, 103.38, 97.32, 101.32, 180.0, 100.0),
    )
    arm = CLIArm(
        config, capture_home=False, return_home=False,
        driver_factory=lambda _: driver,
        operator=SimpleNamespace(wait_ready=lambda *args: None),
        poll_end=lambda: False, sleep_fn=lambda _: None,
    )
    wrapped = SO101EEFEmbodiment(
        arm, load_kinematics(MODEL, joint_low=config.joint_low, joint_high=config.joint_high)
    )
    try:
        wrapped.reset(Scene("eef", "approach"))
        delta = np.array([0.0012, 0.0008, 0.0004, 0.0, 0.0, 0.0, 0.4])
        result = wrapped.step(Action(data=delta))
        assert len(driver.commands) == 1
        assert np.max(np.abs(driver.commands[0][:5] - RUN004_START[:5])) > 1.0
        assert not result.terminated
    finally:
        wrapped.close()


def test_run005_ik_tracks_motor_command_and_reports_target_gap():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = FakeDriver(RUN005_START)
    config = SOArmConfig(
        cameras=(), camera_configs={}, control_hz=10.0, settle_tolerance=0.2,
        joint_low=(-108.0, -103.385, -97.319, -101.32, -180.0, 0.0),
        joint_high=(108.0, 103.385, 97.319, 101.32, 180.0, 100.0),
    )
    arm = CLIArm(
        config, capture_home=False, return_home=False,
        driver_factory=lambda _: driver,
        operator=SimpleNamespace(wait_ready=lambda *args: None),
        poll_end=lambda: False, sleep_fn=lambda _: None,
    )
    wrapped = SO101EEFEmbodiment(
        arm, load_kinematics(MODEL, joint_low=config.joint_low,
                             joint_high=config.joint_high)
    )
    try:
        initial = wrapped.reset(Scene("run005", "approach"))
        eef = initial.state["eef_state"]
        delta = np.array([
            (0.1843 - eef[0]) / 5, 0, (0.0124 - eef[2]) / 5,
            0, 0, 0, (3.5 - eef[6]) / 5,
        ])
        first = wrapped.step(Action(data=delta))
        assert np.max(np.abs(driver.commands[0][:5] - RUN005_START[:5])) > 1.0
        assert first.info["settled"]
        assert 0 < first.info["eef_target_position_error_m"] < 0.0015
        for index in range(1, 5):
            result = wrapped.step(Action(data=delta, meta={"chunk_final": index == 4}))
        assert len(driver.commands) == 5
        assert result.info["eef_target_position_error_m"] < 0.0015
        assert wrapped._expected_pose is None
    finally:
        wrapped.close()


@pytest.mark.parametrize("invalid", [
    np.array([0.003, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0.02, 0, 0, 0]),
    np.array([np.nan, 0, 0, 0, 0, 0, 0]),
    np.array([0.002, 0.002, 0, 0, 0, 0, 0]),
])
def test_invalid_action_sends_no_command(rig, invalid):
    wrapped, driver = rig
    wrapped.reset(Scene("eef", "invalid"))
    with pytest.raises(ValueError, match="rejected"):
        wrapped.step(Action(data=invalid))
    assert driver.commands == []
    assert wrapped._arm._motion_failed


@pytest.mark.parametrize("solution", [
    np.full(5, np.nan),
    np.array([200.0, 0, 0, 0, 0]),
])
def test_invalid_ik_solution_sends_no_command(rig, solution):
    wrapped, driver = rig
    wrapped.reset(Scene("eef", "invalid IK"))
    wrapped._kinematics = SimpleNamespace(
        forward_kinematics=wrapped._kinematics.forward_kinematics,
        inverse_kinematics=lambda *args, **kwargs: solution,
    )
    delta = np.zeros(7)
    delta[2] = 0.002
    with pytest.raises(ValueError, match="rejected"):
        wrapped.step(Action(data=delta))
    assert driver.commands == []


def test_inaccurate_ik_solution_records_feedback_without_changing_solution(rig):
    wrapped, driver = rig
    wrapped.reset(Scene("eef", "approximate IK"))
    inaccurate = START[:5].copy()
    inaccurate[0] += 5.0
    wrapped._kinematics = SimpleNamespace(
        forward_kinematics=wrapped._kinematics.forward_kinematics,
        inverse_kinematics=lambda *args, **kwargs: inaccurate.copy(),
    )
    delta = np.array([0, 0, 0.002, 0, 0, 0, 0])
    first = wrapped.step(Action(data=delta))
    second = wrapped.step(Action(data=delta))
    assert "ik_position_residual" in first.info["eef_tracking_warnings"]
    assert first.info["eef_ik_position_error_m"] > 0.002
    np.testing.assert_array_equal(driver.commands[0][:5], inaccurate)
    np.testing.assert_array_equal(driver.commands[1][:5], inaccurate)
    assert second.info["eef_commanded_joint_pos"][:5] == pytest.approx(inaccurate)
    assert not wrapped._arm._motion_failed


def test_unreachable_ik_exception_sends_no_command(rig):
    wrapped, driver = rig
    wrapped.reset(Scene("eef", "unreachable target"))
    wrapped._kinematics = SimpleNamespace(
        forward_kinematics=wrapped._kinematics.forward_kinematics,
        inverse_kinematics=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("target cannot be reached")
        ),
    )
    delta = np.zeros(7)
    delta[2] = 0.002
    with pytest.raises(ValueError, match="target cannot be reached"):
        wrapped.step(Action(data=delta))
    assert driver.commands == []
    assert wrapped._arm._motion_failed


def test_gripper_limit_rejects_without_motor_command(rig):
    wrapped, driver = rig
    driver.pose[5] = 100.0
    wrapped.reset(Scene("eef", "gripper bound"))
    delta = np.zeros(7)
    delta[6] = 0.5
    with pytest.raises(ValueError, match="gripper delta exceeds"):
        wrapped.step(Action(data=delta))
    assert driver.commands == []


def test_original_home_lifecycle_still_runs():
    if not MODEL.is_file():
        pytest.skip("SO-101 model cache missing")
    driver = FakeDriver(START)
    config = SOArmConfig(cameras=(), camera_configs={}, control_hz=10.0,
                         settle_tolerance=None)
    arm = CLIArm(config, capture_home=False, home_pose=tuple(START),
                 driver_factory=lambda _: driver,
                 operator=SimpleNamespace(wait_ready=lambda *args: None),
                 poll_end=lambda: False, sleep_fn=lambda _: None)
    wrapped = SO101EEFEmbodiment(arm, load_kinematics(MODEL))
    wrapped.reset(Scene("eef", "home"))
    delta = np.zeros(7)
    delta[6] = 0.5
    wrapped.step(Action(data=delta))
    assert driver.commands[-1][5] == 50.5
    wrapped.close()
    assert driver.commands[-1][5] == START[5]
    assert wrapped._arm._driver is None


def test_kinematics_exception_blocks_home_and_motor_command(rig):
    wrapped, driver = rig
    wrapped.reset(Scene("eef", "FK exception"))
    wrapped._kinematics = SimpleNamespace(
        forward_kinematics=lambda *_: (_ for _ in ()).throw(RuntimeError("FK failed")),
    )
    with pytest.raises(RuntimeError, match="FK failed"):
        wrapped.step(Action(data=np.zeros(7)))
    assert driver.commands == []
    assert wrapped._arm._motion_failed
