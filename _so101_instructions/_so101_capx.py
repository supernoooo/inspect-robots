"""SO-101 joint-position profile for the CaP-X code-as-policy plugin.

The wrapped CLIArm keeps the calibration, home, camera, pacing, settling, and
disconnect lifecycle; actions stay absolute calibrated joint degrees, so the
Colosseum recording contract is unchanged. This adapter adds what ``capx``
needs on top: depth aligned to the RealSense color stream (meters), color
intrinsics, an optional camera-to-base transform, base-frame end-effector
state, and an in-process IK solver (``-P ik=embodiment``) over the same URDF
and calibrated limits as the EEF adapter.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots_so101 import MOTORS
from lerobot.utils.rotation import Rotation

from inspect_robots.spaces import StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

DEFAULT_IK_POSITION_TOLERANCE_M = 0.005
DEFAULT_IK_ORIENTATION_WEIGHT = 0.1
IK_MAX_ITERATIONS = 100
IK_CONVERGED_M = 5e-4
ORIENTATION_NOTE_DEG = 10.0
_RIGID_TOLERANCE = 1e-3


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
            f"SO-101 URDF is missing: {path}. "
            "Run python _so101_instructions/_fetch_so101_model.py first."
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
        if (
            low.shape != (5,)
            or high.shape != (5,)
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low >= high)
        ):
            raise ValueError("SO-101 calibrated IK joint limits are invalid")
        # LeRobot's placo solver enforces URDF limits, which can be narrower
        # than this follower's calibrated motor range. Use the same limits as
        # the driver and keep the independent pre-command range check.
        for name, lower, upper in zip(MOTORS[:5], low, high, strict=True):
            model.robot.set_joint_limits(name, math.radians(lower), math.radians(upper))
    return model


def load_extrinsics(path: str | Path) -> np.ndarray:
    """Read a camera-to-base 4x4 rigid transform (meters) from JSON or ``.npy``.

    JSON may be the bare nested list or ``{"camera_to_base": [[...], ...]}``.
    The rotation must be orthonormal with determinant +1 and the last row
    ``[0, 0, 0, 1]``; anything else is rejected before hardware connects.
    """
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(f"camera_extrinsics file is missing: {file}")
    if file.suffix == ".npy":
        value: Any = np.load(file, allow_pickle=False)
    else:
        value = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("camera_to_base")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_extrinsics must be a finite 4x4 camera-to-base matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=_RIGID_TOLERANCE)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=_RIGID_TOLERANCE)
        or abs(np.linalg.det(rotation) - 1.0) > _RIGID_TOLERANCE
    ):
        raise ValueError("camera_extrinsics must be a rigid transform (orthonormal, det +1)")
    return matrix


class RealSenseDepth:
    """Metric depth aligned to the color image of one LeRobot RealSense camera.

    ``install()`` must run after the driver connects. It wraps that camera's
    frame read with ``rs.align(rs.stream.color)`` so depth pixels index the
    same image SAM3 segments, then caches the depth scale and color
    intrinsics. ``depth()`` waits for a fresh frame, so it reflects the scene
    when model code asks, which is while the arm is holding still.
    """

    def __init__(self, arm: Any, camera_name: str):
        self._arm = arm
        self._name = camera_name
        self._scale: float | None = None
        self._intrinsics: np.ndarray | None = None

    def _camera(self) -> Any:
        driver = getattr(self._arm, "_driver", None)
        cameras = getattr(driver, "cameras", None) or {}
        if self._name not in cameras:
            raise RuntimeError(
                f"depth camera {self._name!r} is not connected; available: {sorted(cameras)}"
            )
        return cameras[self._name]

    def install(self) -> None:
        """Align depth to color on the live camera (idempotent per camera object)."""
        import pyrealsense2 as rs

        camera = self._camera()
        if not getattr(camera, "use_depth", False):
            raise RuntimeError(f"camera {self._name!r} was configured without a depth stream")
        if getattr(camera, "rotation", None) is not None:
            raise RuntimeError("depth alignment requires an unrotated RealSense stream")
        if not getattr(camera, "_capx_depth_aligned", False):
            align = rs.align(rs.stream.color)
            read = camera._read_from_hardware
            camera._read_from_hardware = lambda: align.process(read())
            camera._capx_depth_aligned = True
        profile = camera.rs_profile
        self._scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        k = stream.get_intrinsics()
        self._intrinsics = np.array(
            [[k.fx, 0.0, k.ppx], [0.0, k.fy, k.ppy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def depth(self) -> np.ndarray:
        """Return an (H, W) float32 depth map in meters; 0 marks missing depth."""
        if self._scale is None:
            raise RuntimeError("depth is unavailable before the embodiment is reset")
        raw = np.asarray(self._camera().read_depth(timeout_ms=0))
        return (raw.astype(np.float32) * np.float32(self._scale)).astype(np.float32)

    def intrinsics(self) -> np.ndarray:
        """Return the 3x3 color-camera matrix that also applies to aligned depth."""
        if self._intrinsics is None:
            raise RuntimeError("intrinsics are unavailable before the embodiment is reset")
        return self._intrinsics.copy()


def _quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def _pose(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_wxyz))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"quaternion_wxyz must be a unit quaternion, got norm {norm:.4f}")
    w, x, y, z = quaternion_wxyz / norm
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = position
    return pose


class SO101CapxEmbodiment:
    """Expose calibrated joint control plus the CaP-X perception and IK extras."""

    def __init__(
        self,
        arm: Any,
        kinematics: Any,
        *,
        depth_camera: str,
        depth_source: Any = None,
        extrinsics: np.ndarray | None = None,
        ik_position_tolerance_m: float = DEFAULT_IK_POSITION_TOLERANCE_M,
        ik_orientation_weight: float = DEFAULT_IK_ORIENTATION_WEIGHT,
        log: Callable[[str], None] | None = None,
    ):
        if not arm._cfg.use_degrees or arm._cfg.joints_are_delta:
            raise ValueError("SO-101 CaP-X control requires absolute joint targets in degrees")
        for name, value in (
            ("ik_position_tolerance_m", ik_position_tolerance_m),
            ("ik_orientation_weight", ik_orientation_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        cameras = arm.info.observation_space.camera_names
        if depth_camera not in cameras:
            raise ValueError(
                f"depth_camera {depth_camera!r} is not a configured camera; "
                f"available: {sorted(cameras)}"
            )
        self._arm = arm
        self._kinematics = kinematics
        self._depth = (
            depth_source if depth_source is not None else RealSenseDepth(arm, depth_camera)
        )
        self._extrinsics = None if extrinsics is None else np.array(extrinsics, dtype=np.float64)
        self._ik_position_tolerance_m = float(ik_position_tolerance_m)
        self._ik_orientation_weight = float(ik_orientation_weight)
        self._log = log if log is not None else _stderr
        self._low = np.asarray(arm._cfg.joint_low[:5], dtype=np.float64)
        self._high = np.asarray(arm._cfg.joint_high[:5], dtype=np.float64)

        joint_space = arm.info.observation_space
        fields = joint_space.state.fields if joint_space.state is not None else ()
        observation_space = replace(
            joint_space,
            state=StateSpec(
                fields=(
                    *fields,
                    StateField(key="eef_pos", shape=(3,), unit="m"),
                    StateField(key="eef_quat_wxyz", shape=(4,), unit="unit_quaternion"),
                )
            ),
            state_keys=frozenset({*joint_space.state_keys, "eef_pos", "eef_quat_wxyz"}),
        )
        others = sorted(set(cameras) - {depth_camera})
        extrinsics_docs = (
            "obs['extrinsics'] is the calibrated 4x4 camera-to-base transform; map a "
            "camera-frame point p with (obs['extrinsics'] @ [*p, 1])[:3] and a grasp pose "
            "G with obs['extrinsics'] @ G. "
            if self._extrinsics is not None
            else "CLOSED-LOOP MODE: no camera-to-base calibration exists, so obs['extrinsics'] "
            "is absent and camera-frame points or plan_grasp() poses cannot be used as IK "
            "targets directly. Locate the object relative to the gripper in images and "
            "correct every turn, moving at most 3 cm per step. An eye-in-hand camera "
            "(mounted on the gripper) keeps the jaws at a fixed spot in its image, so "
            "centering the object between the jaws aligns the grasp without any "
            "calibration; a fixed camera can compare gripper and object only when it "
            "actually sees both, so check segment() scores before trusting it. "
            f"obs['depth'] (meters) and obs['intrinsics'] belong to camera {depth_camera!r} "
            "only. obs['eef_rotation'] is the measured 3x3 tool-to-base rotation (its "
            "columns are the tool x, y, z axes in base coordinates), so a tool-frame step "
            "d becomes the base step obs['eef_rotation'] @ d. Learn how image directions "
            "map to base motion with one small probe (2 cm along a base axis from state "
            "eef_pos, same eef_quat_wxyz), compare the object's pixel position before and "
            "after next turn, and keep the result in variables (the namespace persists). "
            "Before closing the gripper confirm alignment"
            + (f" in camera {others[0]!r} (segment(..., camera={others[0]!r}))" if others else "")
            + "; after lifting, confirm the object moved with the gripper. "
        )
        self.info = replace(
            arm.info,
            name="so101_capx",
            observation_space=observation_space,
            docs=(
                "SO-101 for code-as-policy. move_to_joints takes the five arm joints "
                "[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in "
                "calibrated degrees; open_gripper/close_gripper drive the 0..100 gripper. "
                "solve_ik(position, quaternion_wxyz) solves the gripper_frame_link pose in "
                "the robot-base frame (meters; x forward, y left, z up) with the calibrated "
                "joint limits. It returns those five degrees, raises when the position "
                f"residual exceeds {self._ik_position_tolerance_m * 1000:.0f} mm, and prints "
                f"a note to stderr when orientation misses by more than "
                f"{ORIENTATION_NOTE_DEG:.0f} degrees: a five-joint arm cannot reach every "
                "orientation, and position is prioritized. The gripper approach axis is "
                "the tool frame's local +z (wrist toward fingertips), the same convention "
                "as Contact-GraspNet poses; a straight-down approach is close to "
                "quaternion_wxyz [0, 0, 1, 0], and wrist_roll turns about that axis. "
                "Contact-GraspNet places its pose origin roughly 0.10 m behind the "
                "fingertip contact along +z, while gripper_frame_link sits near the "
                "fingertips, so shift a grasp position forward along its +z before IK. "
                "state eef_pos and eef_quat_wxyz report the measured tool pose at the start "
                "of each turn; obs['eef_rotation'] is the same orientation as a matrix. "
                f"segment() and plan_grasp() use camera {depth_camera!r}; obs['depth'] is "
                "aligned to that image in meters (0 = no reading) and obs['intrinsics'] is "
                "its 3x3 matrix. " + extrinsics_docs + (arm.info.docs or "")
            ),
        )

    def __getattr__(self, name: str) -> Any:
        """Preserve optional CLIArm hooks, including operator-session integration."""
        return getattr(self._arm, name)

    def _fk(self, joints: np.ndarray) -> np.ndarray:
        pose = np.asarray(self._kinematics.forward_kinematics(joints), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("forward kinematics returned an invalid pose")
        return pose

    def _augment(self, observation: Observation) -> Observation:
        joints = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("SO-101 joint observation is invalid")
        pose = self._fk(joints[:5])
        extra: dict[str, Any] = {
            **observation.extra,
            "depth": self._depth.depth,
            "intrinsics": self._depth.intrinsics,
            "ik_solver": self.solve_ik,
            "eef_rotation": pose[:3, :3].copy(),
        }
        if self._extrinsics is not None:
            extra["extrinsics"] = self._extrinsics.copy()
        return replace(
            observation,
            state={
                **observation.state,
                "eef_pos": pose[:3, 3].copy(),
                "eef_quat_wxyz": _quat_wxyz(pose[:3, :3]),
            },
            extra=extra,
        )

    def solve_ik(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, seed: np.ndarray
    ) -> np.ndarray:
        """Return five calibrated arm degrees reaching a base-frame tool pose.

        The placo solver takes one regularized step per call, so it is
        iterated from ``seed`` until the position converges. An unreachable
        orientation would otherwise drag the tool off target, so the
        orientation weight is relaxed tenfold per pass (to position-only)
        until the position fits. Results outside the calibrated limits or
        beyond the position tolerance raise; the CaP-X sandbox turns that into
        stderr the model can react to.
        """
        target_position = np.asarray(position, dtype=np.float64).reshape(-1)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
        start = np.asarray(seed, dtype=np.float64).reshape(-1)
        if (
            target_position.shape != (3,)
            or quaternion.shape != (4,)
            or start.shape != (5,)
            or not np.all(np.isfinite(target_position))
            or not np.all(np.isfinite(start))
        ):
            raise ValueError("solve_ik needs a finite position (3,), quaternion (4,), seed (5,)")
        target = _pose(target_position, quaternion)
        weight = self._ik_orientation_weight
        for orientation_weight in (weight, weight / 10, weight / 100, 0.0):
            joints, achieved, error = self._iterate_ik(start, target, orientation_weight)
            if error <= self._ik_position_tolerance_m:
                break
        if error > self._ik_position_tolerance_m:
            raise ValueError(
                f"SO-101 IK cannot reach position {np.round(target_position, 4).tolist()} m: "
                f"residual {error * 1000:.1f} mm exceeds "
                f"{self._ik_position_tolerance_m * 1000:.1f} mm (out of reach or limits)"
            )
        if np.any(joints < self._low) or np.any(joints > self._high):
            raise ValueError("SO-101 IK solution exceeds the calibrated joint limits")
        relative = achieved[:3, :3] @ target[:3, :3].T
        cosine = float(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
        missed = math.degrees(math.acos(cosine))
        if missed > ORIENTATION_NOTE_DEG:
            self._log(
                f"solve_ik note: orientation residual {missed:.0f} deg; achieved "
                f"quaternion_wxyz {np.round(_quat_wxyz(achieved[:3, :3]), 3).tolist()}"
            )
        return joints

    def _iterate_ik(
        self, seed: np.ndarray, target: np.ndarray, orientation_weight: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        joints = seed
        achieved = self._fk(joints)
        error = math.inf
        for _ in range(IK_MAX_ITERATIONS):
            solved = np.asarray(
                self._kinematics.inverse_kinematics(
                    np.r_[joints, 0.0],
                    target,
                    position_weight=1.0,
                    orientation_weight=orientation_weight,
                ),
                dtype=np.float64,
            ).reshape(-1)[:5]
            if solved.shape != (5,) or not np.all(np.isfinite(solved)):
                raise ValueError("SO-101 IK returned invalid joint values")
            joints = solved
            achieved = self._fk(joints)
            error = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
            if error <= IK_CONVERGED_M:
                break
        return joints, achieved, error

    def reset(self, scene: Any, *, seed: int | None = None) -> Observation:
        """Reset the arm, then align depth on the freshly connected camera."""
        observation = self._arm.reset(scene, seed=seed)
        self._depth.install()
        return self._augment(observation)

    def step(self, action: Action) -> StepResult:
        """Send the joint target unchanged and augment the measured observation."""
        result = self._arm.step(action)
        return replace(result, observation=self._augment(result.observation))

    def close(self) -> None:
        """Close the wrapped arm (shutdown home ramp and disconnect)."""
        self._arm.close()


def _stderr(message: str) -> None:
    import sys

    print(message, file=sys.stderr, flush=True)


__all__ = [
    "RealSenseDepth",
    "SO101CapxEmbodiment",
    "load_extrinsics",
    "load_kinematics",
]
