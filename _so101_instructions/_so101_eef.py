"""SO-101 Cartesian delta adapter over the existing joint-position driver.

The adapter never connects to hardware itself.  Its wrapped CLIArm retains the
calibration, home, camera, pacing, settling, and disconnect lifecycle.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import numpy as np
from inspect_robots_so101 import MOTORS
from lerobot.utils.rotation import Rotation

from inspect_robots.spaces import ActionSemantics, Box, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

EEF_STATE_LABELS = ("x", "y", "z", "rx", "ry", "rz", "gripper")
EEF_LABELS = EEF_STATE_LABELS  # move_by makes these named values deltas.
EEF_STEP_LIMIT = np.array((0.002, 0.002, 0.002, 0.01, 0.01, 0.01, 5.0))
IK_POSITION_WARNING_M = 0.0008
IK_ORIENTATION_WARNING_RAD = 0.004
MAX_TRACKING_POSITION_ERROR_M = 0.0015
MAX_TRACKING_ORIENTATION_ERROR_RAD = 0.01
MAX_TRACKING_GRIPPER_ERROR = 1.0
DEFAULT_JOINT_TRACKING_TOLERANCE_DEG = 1.0
_EPS = 1e-8
_Result = TypeVar("_Result")


def _mark_motion_failure(method: Callable[..., _Result]) -> Callable[..., _Result]:
    """Prevent a shutdown home ramp after a failed EEF calculation."""
    @wraps(method)
    def guarded(self: SO101EEFEmbodiment, *args: Any, **kwargs: Any) -> _Result:
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self._arm._motion_failed = True
            raise

    return guarded


def load_kinematics(
    urdf_path: str | Path,
    *,
    joint_low: tuple[float, ...] | None = None,
    joint_high: tuple[float, ...] | None = None,
) -> Any:
    """Load a local URDF before connecting the robot; do not fetch at runtime."""
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"SO-101 URDF is missing: {path}. Run _so101_instructions/_fetch_so101_model.py first."
        )
    root = ET.parse(path).getroot()
    joints = {joint.get("name") for joint in root.findall("joint")}
    links = {link.get("name") for link in root.findall("link")}
    missing = set(MOTORS[:5]) - joints
    if missing or "gripper_frame_link" not in links:
        raise ValueError(
            f"SO-101 URDF joint/frame mismatch: missing {sorted(missing)} or gripper frame"
        )
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not (path.parent / filename).is_file():
            raise FileNotFoundError(f"SO-101 URDF mesh is missing: {path.parent / filename}")
    from lerobot.model.kinematics import RobotKinematics

    model = RobotKinematics(
        str(path), target_frame_name="gripper_frame_link", joint_names=list(MOTORS[:5])
    )
    if (joint_low is None) != (joint_high is None):
        raise ValueError("SO-101 IK requires both calibrated joint limit vectors")
    if joint_low is not None and joint_high is not None:
        low = np.asarray(joint_low[:5], dtype=np.float64)
        high = np.asarray(joint_high[:5], dtype=np.float64)
        if (low.shape != (5,) or high.shape != (5,)
                or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
                or np.any(low >= high)):
            raise ValueError("SO-101 calibrated IK joint limits are invalid")
        # LeRobot's placo solver enforces URDF limits, which can be narrower
        # than this follower's calibrated motor range. Use the same limits as
        # the driver and keep the independent pre-command range check.
        for name, lower, upper in zip(MOTORS[:5], low, high, strict=True):
            model.robot.set_joint_limits(name, math.radians(lower), math.radians(upper))
    return model


class SO101EEFEmbodiment:
    """Expose base-frame EEF deltas while sending checked joint targets to CLIArm."""

    def __init__(
        self,
        arm: Any,
        kinematics: Any,
        *,
        tracking_position_tolerance_m: float = MAX_TRACKING_POSITION_ERROR_M,
        tracking_orientation_tolerance_deg: float = math.degrees(
            MAX_TRACKING_ORIENTATION_ERROR_RAD
        ),
        tracking_joint_tolerance_deg: float = DEFAULT_JOINT_TRACKING_TOLERANCE_DEG,
    ):
        if not arm._cfg.use_degrees or arm._cfg.joints_are_delta:
            raise ValueError("SO-101 EEF control requires absolute joint targets in degrees")
        for name, tolerance in (
            ("tracking_position_tolerance_m", tracking_position_tolerance_m),
            ("tracking_orientation_tolerance_deg", tracking_orientation_tolerance_deg),
            ("tracking_joint_tolerance_deg", tracking_joint_tolerance_deg),
        ):
            if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                    or not math.isfinite(tolerance) or tolerance <= 0):
                raise ValueError(f"{name} must be a positive finite number")
        self._arm = arm
        self._kinematics = kinematics
        self._tracking_position_tolerance_m = float(tracking_position_tolerance_m)
        self._tracking_orientation_tolerance_rad = math.radians(
            tracking_orientation_tolerance_deg
        )
        self._tracking_joint_tolerance_deg = float(tracking_joint_tolerance_deg)
        self._last_joint_observation: Observation | None = None
        self._expected_pose: np.ndarray | None = None
        self._expected_gripper: float | None = None
        joint_space = arm.info.observation_space
        fields = joint_space.state.fields if joint_space.state is not None else ()
        observation_space = replace(
            joint_space,
            state=StateSpec(fields=(
                *fields,
                StateField(key="eef_state", shape=(7,), unit="m+axis_angle_rad+gripper_0_100"),
                StateField(key="eef_quat_wxyz", shape=(4,), unit="unit_quaternion"),
            )),
            state_keys=frozenset({*joint_space.state_keys, "eef_state", "eef_quat_wxyz"}),
        )
        action_space = Box(
            shape=(7,),
            low=-EEF_STEP_LIMIT.copy(),
            high=EEF_STEP_LIMIT.copy(),
            semantics=ActionSemantics(
                control_mode="eef_delta_pose",
                rotation_repr="axis_angle",
                gripper="continuous",
                frame="base",
                dim_labels=EEF_LABELS,
                agent_eef_target=True,
            ),
        )
        self.info = replace(
            arm.info,
            name="so101_eef",
            action_space=action_space,
            observation_space=observation_space,
            docs=(
                "SO-101 Cartesian control. Use move_eef with absolute base-frame "
                "x, y, z targets in meters, optional complete unit orientation "
                "quaternion qw, qx, qy, qz, and optional absolute gripper target "
                "in 0..100 (0 closed, 100 open). Omitted position coordinates "
                "and orientation hold their exact measured values. The requested "
                "step count is a minimum; the agent tool adds smooth control "
                "steps as needed to reach the full target. The state eef_state is "
                "[x,y,z,rx,ry,rz,gripper]; eef_quat_wxyz is the measured "
                "absolute orientation quaternion. The "
                "arm has five joints, so some poses cannot be reached. Observe "
                "the measured result after each trajectory. Valid IK joint "
                "targets are sent unchanged "
                "through the calibrated joint driver; use the next measured "
                "EEF pose to plan again when the motor does not follow. "
                "The cameras do not "
                "provide calibrated 3D object coordinates. "
                + (arm.info.docs or "")
            ),
        )

    def __getattr__(self, name: str) -> Any:
        """Preserve optional CLIArm hooks, including operator-session integration."""
        return getattr(self._arm, name)

    def _fail(self, message: str) -> NoReturn:
        # CLIArm skips shutdown homing after a motion fault; this includes IK
        # rejection, because a surprise home ramp could itself be unsafe.
        self._arm._motion_failed = True
        raise ValueError("SO-101 EEF motion rejected: " + message)

    def _augment(self, observation: Observation) -> Observation:
        joints = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            self._fail("joint observation is invalid")
        pose = np.asarray(self._kinematics.forward_kinematics(joints[:5]), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            self._fail("forward kinematics returned an invalid pose")
        rotation_value = Rotation.from_matrix(pose[:3, :3])
        rotation = rotation_value.as_rotvec()
        quat_xyzw = rotation_value.as_quat()
        quat_wxyz = np.r_[quat_xyzw[3], quat_xyzw[:3]]
        eef = np.r_[pose[:3, 3], rotation, joints[5]]
        if not np.all(np.isfinite(eef)) or not np.all(np.isfinite(quat_wxyz)):
            self._fail("forward kinematics returned a non-finite pose")
        self._last_joint_observation = observation
        return replace(observation, state={
            "eef_state": eef, "eef_quat_wxyz": quat_wxyz, **observation.state,
        })

    def _wait_for_gripper(self, observation: Observation, commanded: float) -> Observation:
        """Give the gripper its own bounded settle window; upstream waits only for arm joints."""
        if abs(float(observation.state["joint_pos"][5]) - commanded) <= MAX_TRACKING_GRIPPER_ERROR:
            return observation
        timeout_s = self._arm._cfg.settle_timeout_s
        started = self._arm._clock()
        max_polls = max(1, math.ceil(timeout_s / 0.05))
        for _ in range(max_polls):
            if self._arm._clock() - started >= timeout_s:
                break
            self._arm._sleep(0.05)
            observation = self._augment(self._arm._observe(self._arm._instruction))
            if (abs(float(observation.state["joint_pos"][5]) - commanded)
                    <= MAX_TRACKING_GRIPPER_ERROR):
                break
        return observation

    @_mark_motion_failure
    def reset(self, scene: Any, *, seed: int | None = None) -> Observation:
        self._last_joint_observation = None
        self._expected_pose = None
        self._expected_gripper = None
        return self._augment(self._arm.reset(scene, seed=seed))

    @_mark_motion_failure
    def step(self, action: Action) -> StepResult:
        if self._last_joint_observation is None:
            self._fail("step called before reset")
        if self._arm._motion_failed:
            self._fail("a previous EEF or motor action failed")
        delta = np.asarray(action.data, dtype=np.float64)
        if delta.shape != (7,) or not np.all(np.isfinite(delta)):
            self._fail("expected seven finite EEF deltas")
        if np.any(np.abs(delta) > EEF_STEP_LIMIT + _EPS):
            self._fail("EEF delta exceeds the per-step action bounds")
        if np.linalg.norm(delta[:3]) > EEF_STEP_LIMIT[0] + _EPS:
            self._fail("EEF translation exceeds the 2 mm total step limit")
        measured = np.asarray(self._last_joint_observation.state["joint_pos"], dtype=np.float64)
        expected_gripper = (
            float(measured[5]) if self._expected_gripper is None else self._expected_gripper
        ) + float(delta[6])
        gripper_low = self._arm._cfg.joint_low[5]
        gripper_high = self._arm._cfg.joint_high[5]
        if (expected_gripper < gripper_low - _EPS
                or expected_gripper > gripper_high + _EPS):
            self._fail("gripper delta exceeds calibrated joint limits")
        expected_gripper = float(np.clip(expected_gripper, gripper_low, gripper_high))
        # Follow the complete agent trajectory even when the servo trails one
        # waypoint, while keeping each command within one native EEF step of
        # the latest measured gripper position.
        gripper = float(measured[5] + np.clip(
            expected_gripper - measured[5], -EEF_STEP_LIMIT[6], EEF_STEP_LIMIT[6]
        ))
        if gripper < gripper_low - _EPS or gripper > gripper_high + _EPS:
            self._fail("gripper delta exceeds calibrated joint limits")
        pose = np.asarray(self._kinematics.forward_kinematics(measured[:5]), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            self._fail("forward kinematics returned an invalid pose")
        expected = pose.copy() if self._expected_pose is None else self._expected_pose.copy()
        expected[:3, 3] += delta[:3]
        expected[:3, :3] = Rotation.from_rotvec(delta[3:6]).as_matrix() @ expected[:3, :3]
        ik_position_error: float | None = None
        ik_orientation_error: float | None = None
        if np.any(delta[:6] != 0):
            # Aim at the accumulated agent target from the latest measurement.
            desired = expected
            try:
                solved = np.asarray(
                    self._kinematics.inverse_kinematics(
                        measured[:5], desired, position_weight=1.0, orientation_weight=0.01
                    ),
                    dtype=np.float64,
                )
            except Exception as exc:
                self._fail(f"inverse kinematics failed: {exc}")
            if solved.shape != (5,) or not np.all(np.isfinite(solved)):
                self._fail("inverse kinematics returned invalid joint values")
        else:
            # Gripper-only moves must hold the measured arm joints exactly.
            # Solving an unchanged pose can still move joints near a limit.
            solved = measured[:5].copy()
        low = np.asarray(self._arm._cfg.joint_low[:5], dtype=np.float64)
        high = np.asarray(self._arm._cfg.joint_high[:5], dtype=np.float64)
        if np.any(solved < low) or np.any(solved > high):
            self._fail("inverse kinematics exceeds calibrated joint limits")
        if np.any(delta[:6] != 0):
            achieved = np.asarray(self._kinematics.forward_kinematics(solved), dtype=np.float64)
            if achieved.shape != (4, 4) or not np.all(np.isfinite(achieved)):
                self._fail("inverse kinematics returned an invalid achieved pose")
            ik_position_error = float(np.linalg.norm(achieved[:3, 3] - desired[:3, 3]))
            relative_rotation = achieved[:3, :3] @ desired[:3, :3].T
            cosine = float(np.clip((np.trace(relative_rotation) - 1) / 2, -1.0, 1.0))
            ik_orientation_error = math.acos(cosine)
            # A five-joint arm cannot satisfy every requested six-dimensional
            # pose. Record the residual, but publish the IK solution itself.
        commanded_pose = np.asarray(
            self._kinematics.forward_kinematics(solved), dtype=np.float64
        )
        if commanded_pose.shape != (4, 4) or not np.all(np.isfinite(commanded_pose)):
            self._fail("inverse kinematics returned an invalid commanded pose")
        target = np.r_[solved, gripper]
        result = self._arm.step(Action(data=target, meta=action.meta))
        observation = self._augment(result.observation)
        if result.terminated:
            self._expected_pose = None
            self._expected_gripper = None
            return replace(result, observation=observation, info={
                **result.info, "eef_commanded_joint_pos": target.tolist(),
            })
        if delta[6] != 0:
            observation = self._wait_for_gripper(observation, gripper)
        actual = np.asarray(observation.state["eef_state"], dtype=np.float64)
        measured_joints = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        position_error = float(np.linalg.norm(actual[:3] - commanded_pose[:3, 3]))
        actual_rotation = Rotation.from_rotvec(actual[3:6]).as_matrix()
        rotation_error = float(
            np.linalg.norm(
                Rotation.from_matrix(actual_rotation @ commanded_pose[:3, :3].T).as_rotvec()
            )
        )
        joint_errors = np.abs(measured_joints[:5] - target[:5])
        worst_joint = int(np.argmax(joint_errors))
        commanded_joint_move = float(np.max(np.abs(target[:5] - measured[:5])))
        observed_joint_move = float(np.max(np.abs(measured_joints[:5] - measured[:5])))
        warnings: list[str] = []
        if ik_position_error is not None and ik_position_error > IK_POSITION_WARNING_M:
            warnings.append("ik_position_residual")
        if (ik_orientation_error is not None
                and ik_orientation_error > IK_ORIENTATION_WARNING_RAD):
            warnings.append("ik_orientation_residual")
        if result.info.get("settled") is False:
            warnings.append("settle_timeout")
        if commanded_joint_move > 0.2 and observed_joint_move < 0.05:
            warnings.append("arm_no_motion")
        if joint_errors[worst_joint] > self._tracking_joint_tolerance_deg + _EPS:
            warnings.append("joint_tracking_error")
        # Each gripper command is bounded from the latest measured position.
        # The accumulated agent target can therefore stay ahead after normal
        # servo lag; compare tracking with the command actually sent instead.
        gripper_error = abs(float(actual[6]) - gripper)
        gripper_target_error = abs(float(actual[6]) - expected_gripper)
        gripper_progress = abs(float(actual[6]) - float(measured[5]))
        if (delta[6] != 0 and gripper_progress < 0.05
                and gripper_target_error > MAX_TRACKING_GRIPPER_ERROR):
            warnings.append("gripper_no_motion")
        if position_error > self._tracking_position_tolerance_m:
            warnings.append("eef_position_tracking_error")
        if rotation_error > self._tracking_orientation_tolerance_rad:
            warnings.append("eef_orientation_tracking_error")
        if gripper_error > MAX_TRACKING_GRIPPER_ERROR:
            warnings.append("gripper_tracking_error")
        if action.meta.get("chunk_final"):
            self._expected_pose = None
            self._expected_gripper = None
        else:
            self._expected_pose = expected
            self._expected_gripper = expected_gripper
        target_position_error = float(np.linalg.norm(actual[:3] - expected[:3, 3]))
        target_rotation_error = float(np.linalg.norm(
            Rotation.from_matrix(actual_rotation @ expected[:3, :3].T).as_rotvec()
        ))
        return replace(result, observation=observation, info={
            **result.info,
            "eef_tracking_warnings": warnings,
            "eef_tracking_position_error_m": position_error,
            "eef_tracking_orientation_error_rad": rotation_error,
            "eef_tracking_joint_error_deg": float(joint_errors[worst_joint]),
            "eef_tracking_worst_joint": MOTORS[worst_joint],
            "eef_tracking_gripper_error": gripper_error,
            "eef_tracking_commanded_joint_move_deg": commanded_joint_move,
            "eef_tracking_observed_joint_move_deg": observed_joint_move,
            "eef_ik_position_error_m": ik_position_error,
            "eef_ik_orientation_error_rad": ik_orientation_error,
            "eef_commanded_joint_pos": target.tolist(),
            "eef_target_position_error_m": target_position_error,
            "eef_target_orientation_error_rad": target_rotation_error,
            "eef_target_gripper_error": gripper_target_error,
        })

    def close(self) -> None:
        self._arm.close()
