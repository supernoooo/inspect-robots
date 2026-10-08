"""The named EEF plugin passes Cartesian actions through YAM's IK path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from inspect_robots.scene import Scene
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import build_toolset
from inspect_robots_yam import YamConfig
from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, DEFAULT_JOINT_HOME_POSE
from inspect_robots_yam_eef import YamEEFEmbodiment


class _Driver:
    def __init__(self) -> None:
        self.joints = np.zeros(14)
        self.commands: list[np.ndarray] = []

    def get_joint_pos(self) -> np.ndarray:
        return self.joints.copy()

    def get_joint_eff(self) -> np.ndarray:
        return np.zeros(14)

    def get_motor_temps(self) -> np.ndarray:
        return np.full(14, 30.0)

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.joints = target.copy()
        self.commands.append(target.copy())

    def close(self) -> None:
        pass


class _RawKinematics:
    def __init__(self, joint_offset: float) -> None:
        self.ranges = np.array([[-2.0, 2.0]] * 6 + [[0.0, 0.04], [0.0, 0.04]])
        self.joint_offset = joint_offset
        self.targets: list[np.ndarray] = []

    def get_joint_ranges(self) -> np.ndarray:
        return self.ranges.copy()

    def set_joint_ranges(self, ranges: np.ndarray) -> None:
        self.ranges = ranges.copy()

    def fk(self, _q: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = [0.3, 0.0, 0.2]
        return pose

    def ik(self, target: np.ndarray, init_q: np.ndarray, _max_iters: int) -> tuple[bool, np.ndarray]:
        self.targets.append(target.copy())
        result = init_q.copy()
        result[:6] += self.joint_offset
        return True, result


def test_named_eef_mode_runs_ik_then_sends_joint_targets() -> None:
    driver = _Driver()
    left, right = _RawKinematics(0.05), _RawKinematics(-0.05)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    embodiment = YamEEFEmbodiment(
        cam_height=4,
        cam_width=4,
        unattended=True,
        rest_secs=0.1,
        driver_factory=lambda _config: driver,
        kinematics_factory=lambda _config: (left, right),
        camera_reader=lambda _config: {name: image for name in ("top_cam", "left_cam", "right_cam")},
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )
    assert embodiment.info.name == "yam_eef"
    assert embodiment.info.action_space.semantics.control_mode == "eef_abs_pose"
    assert embodiment.info.observation_space.state_keys == {"joint_pos", "eef_state"}
    assert embodiment._cfg.home_pose == DEFAULT_JOINT_HOME_POSE
    assert embodiment._cfg.rest_pose == DEFAULT_JOINT_HOME_POSE

    try:
        observed = embodiment.reset(Scene(id="eef", instruction="move"))
        np.testing.assert_allclose(observed.state["joint_pos"], DEFAULT_JOINT_HOME_POSE)
        toolset = build_toolset(
            embodiment.info.action_space,
            embodiment.info.observation_space,
            control_hz=embodiment.info.control_hz,
        )
        assert toolset.schemas()[0]["function"]["name"] == "move_to"
        proposed = toolset.execute(
            ToolCall(
                id="move-1",
                name="move_to",
                arguments=json.dumps(
                    {
                        "targets": {
                            "left_x": 0.35,
                            "right_y": 0.1,
                            "left_gripper": 0.2,
                            "right_gripper": 0.8,
                        },
                        "note": "Both arms are in the home pose; move them to the target.",
                    }
                ),
            ),
            observed,
        )
        assert proposed.error is None
        assert proposed.chunk is not None
        result = embodiment.step(proposed.chunk.actions[-1])
        assert left.targets[-1][0, 3] == pytest.approx(0.35)
        assert right.targets[-1][1, 3] == pytest.approx(0.1)
        np.testing.assert_allclose(
            driver.commands[-1][:6], np.array(DEFAULT_JOINT_HOME_POSE[:6]) + 0.05
        )
        np.testing.assert_allclose(
            driver.commands[-1][7:13], np.array(DEFAULT_JOINT_HOME_POSE[7:13]) - 0.05
        )
        np.testing.assert_allclose(driver.commands[-1][[6, 13]], [0.2, 0.8])
        assert "eef_state" in result.observation.state
    finally:
        embodiment.close()


def test_joint_mode_cannot_be_selected_under_eef_name() -> None:
    with pytest.raises(ValueError, match="control_interface='eef_pos'"):
        YamEEFEmbodiment(control_interface="joints")
    with pytest.raises(ValueError, match="config.control_interface='eef_pos'"):
        YamEEFEmbodiment(YamConfig())


def test_config_object_uses_joint_home_unless_overridden() -> None:
    default = YamEEFEmbodiment(YamConfig(control_interface="eef_pos"))
    assert default._cfg.home_pose == DEFAULT_JOINT_HOME_POSE
    assert default._cfg.rest_pose == DEFAULT_JOINT_HOME_POSE
    assert "No object pose, demonstration, or scene geometry is supplied" in default.info.docs
    assert "Negative pitch tilts that tool axis toward base -z" in default.info.docs
    overridden = YamEEFEmbodiment(
        YamConfig(control_interface="eef_pos", home_pose=DEFAULT_EEF_HOME_POSE)
    )
    assert overridden._cfg.home_pose == DEFAULT_EEF_HOME_POSE
    assert "At the default zero-joint home" not in overridden.info.docs
