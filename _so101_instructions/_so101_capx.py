"""SO-101 joint-position profile for the CaP-X code-as-policy plugin.

The wrapped CLIArm keeps the calibration, home, camera, pacing, settling, and
disconnect lifecycle; actions stay absolute calibrated joint degrees, so the
Colosseum recording contract is unchanged. The shared
``inspect_robots_capx.embodiment.CapxArmEmbodiment`` base adds what ``capx``
needs on top (depth, intrinsics, extrinsics, tool state, ``solve_ik``); this
module supplies the SO-101 parts: the placo URDF kinematics, depth aligned to
the RealSense color stream through the LeRobot camera, and the prompt notes.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots_capx.embodiment import (
    DEFAULT_IK_ORIENTATION_WEIGHT,
    DEFAULT_IK_POSITION_TOLERANCE_M,
    IK_CONVERGED_M,
    IK_MAX_ITERATIONS,
    ORIENTATION_NOTE_DEG,
    CapxArmEmbodiment,
    load_extrinsics,
)
from inspect_robots_so101 import MOTORS

ARM_DOF = 5


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
    missing = set(MOTORS[:ARM_DOF]) - joints
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
        str(path), target_frame_name="gripper_frame_link", joint_names=list(MOTORS[:ARM_DOF])
    )
    if (joint_low is None) != (joint_high is None):
        raise ValueError("SO-101 IK requires both calibrated joint limit vectors")
    if joint_low is not None and joint_high is not None:
        low = np.asarray(joint_low[:ARM_DOF], dtype=np.float64)
        high = np.asarray(joint_high[:ARM_DOF], dtype=np.float64)
        if (
            low.shape != (ARM_DOF,)
            or high.shape != (ARM_DOF,)
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low >= high)
        ):
            raise ValueError("SO-101 calibrated IK joint limits are invalid")
        # LeRobot's placo solver enforces URDF limits, which can be narrower
        # than this follower's calibrated motor range. Use the same limits as
        # the driver and keep the independent pre-command range check.
        for name, lower, upper in zip(MOTORS[:ARM_DOF], low, high, strict=True):
            model.robot.set_joint_limits(name, math.radians(lower), math.radians(upper))
    return model


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


class SO101CapxEmbodiment(CapxArmEmbodiment):
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
        self._kinematics = kinematics
        super().__init__(
            arm,
            name="so101_capx",
            label="SO-101",
            arm_dof=ARM_DOF,
            joint_low=arm._cfg.joint_low[:ARM_DOF],
            joint_high=arm._cfg.joint_high[:ARM_DOF],
            limits_label="calibrated joint limits",
            depth_camera=depth_camera,
            depth_source=(
                depth_source if depth_source is not None else RealSenseDepth(arm, depth_camera)
            ),
            extrinsics=extrinsics,
            ik_position_tolerance_m=ik_position_tolerance_m,
            ik_orientation_weight=ik_orientation_weight,
            log=log,
        )

    def _kinematics_docs(self) -> str:
        return (
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
            f"segment() and plan_grasp() use camera {self._depth_camera!r}; obs['depth'] is "
            "aligned to that image in meters (0 = no reading) and obs['intrinsics'] is "
            "its 3x3 matrix. "
        )

    def _fk(self, joints: np.ndarray) -> np.ndarray:
        pose = np.asarray(self._kinematics.forward_kinematics(joints), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("forward kinematics returned an invalid pose")
        return pose

    def _iterate_ik(
        self, seed: np.ndarray, target: np.ndarray, orientation_weight: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Iterate placo's single regularized step from ``seed`` until the position converges."""
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
            ).reshape(-1)[:ARM_DOF]
            if solved.shape != (ARM_DOF,) or not np.all(np.isfinite(solved)):
                raise ValueError("SO-101 IK returned invalid joint values")
            joints = solved
            achieved = self._fk(joints)
            error = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
            if error <= IK_CONVERGED_M:
                break
        return joints, achieved, error


__all__ = [
    "DEFAULT_IK_ORIENTATION_WEIGHT",
    "DEFAULT_IK_POSITION_TOLERANCE_M",
    "RealSenseDepth",
    "SO101CapxEmbodiment",
    "load_extrinsics",
    "load_kinematics",
]
