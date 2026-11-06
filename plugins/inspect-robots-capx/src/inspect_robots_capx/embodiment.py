"""Shared code-as-policy adapter that adds perception and IK to a joint-space arm.

A ``capx`` run with ``-P ik=embodiment`` needs more from an embodiment than
joint control: metric depth aligned to one camera image, that camera's
intrinsics, an optional camera-to-base transform, the measured tool pose, and
an in-process IK solver published as ``Observation.extra["ik_solver"]``.
:class:`CapxArmEmbodiment` adds all of that around an existing joint-space
embodiment and keeps the robot-specific parts behind small hooks: forward
kinematics, one IK pass, the prompt notes about joints and frames, and (for
multi-arm robots) how a single-arm action maps onto the full command.

Robot packages subclass it once per rig; the solver loop, the observation
augmentation, the closed-loop prompt, and the lifecycle are shared.
"""

from __future__ import annotations

import json
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

#: Largest accepted tool-position residual of a ``solve_ik`` result.
DEFAULT_IK_POSITION_TOLERANCE_M = 0.005
#: Orientation weight of the first IK pass; later passes relax it tenfold.
DEFAULT_IK_ORIENTATION_WEIGHT = 0.1
#: Solver passes per orientation weight before the residual is judged.
IK_MAX_ITERATIONS = 100
#: Position residual at which an IK pass stops early.
IK_CONVERGED_M = 5e-4
#: Orientation residual above which ``solve_ik`` prints a note for the model.
ORIENTATION_NOTE_DEG = 10.0
_RIGID_TOLERANCE = 1e-3

Pose = npt.NDArray[np.float64]
Vec = npt.NDArray[np.float64]


def load_extrinsics(path: str | Path) -> Pose:
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


def quat_wxyz_from_matrix(rotation: npt.ArrayLike) -> Vec:
    """Return the unit quaternion ``[w, x, y, z]`` of a 3x3 rotation matrix.

    Uses the largest-diagonal branch so every rotation, including half turns,
    converts without cancellation; the sign is normalized to ``w >= 0``.
    """
    m = np.asarray(rotation, dtype=np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    unit = np.asarray(q / np.linalg.norm(q), dtype=np.float64)
    return unit if unit[0] >= 0.0 else -unit


def pose_from_quat_wxyz(position: npt.ArrayLike, quaternion_wxyz: npt.ArrayLike) -> Pose:
    """Return the 4x4 pose of a position and a unit ``[w, x, y, z]`` quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"quaternion_wxyz must be a unit quaternion, got norm {norm:.4f}")
    w, x, y, z = quaternion / norm
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def rotation_angle_deg(achieved: npt.ArrayLike, target: npt.ArrayLike) -> float:
    """Return the angle in degrees between two 3x3 rotation matrices."""
    relative = np.asarray(achieved, dtype=np.float64) @ np.asarray(target, dtype=np.float64).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class DepthSource(Protocol):
    """Metric depth and intrinsics for the camera ``segment()`` and ``plan_grasp()`` use.

    ``install()`` runs once per ``reset()`` after the wrapped arm connected;
    ``depth()`` returns an ``(H, W)`` float32 array in meters aligned to that
    camera's image (0 marks no reading) and ``intrinsics()`` its 3x3 matrix.
    """

    def install(self) -> None:
        """Prepare the live camera; called after every ``reset()`` of the arm."""
        ...

    def depth(self) -> npt.NDArray[np.float32]:
        """Return the newest depth image in meters."""
        ...

    def intrinsics(self) -> Pose:
        """Return the 3x3 camera matrix matching ``depth()``."""
        ...


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _positive_finite(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


class CapxArmEmbodiment(ABC):
    """Joint control plus the CaP-X perception and IK extras over a wrapped arm.

    Subclasses set whatever their hooks need (kinematics, slices) before
    calling ``super().__init__``, then implement :meth:`_fk`,
    :meth:`_iterate_ik`, and :meth:`_kinematics_docs`. Multi-arm robots also
    override :meth:`_base_info`, :meth:`_project`, and :meth:`_expand` to
    present one arm to the single-arm ``capx`` profile. Every other method of
    the wrapped arm (operator hooks, task binding, guardrails) passes through
    unchanged via ``__getattr__``.
    """

    def __init__(
        self,
        arm: Any,
        *,
        name: str,
        label: str,
        arm_dof: int,
        joint_low: npt.ArrayLike,
        joint_high: npt.ArrayLike,
        depth_camera: str,
        depth_source: DepthSource,
        extrinsics: npt.ArrayLike | None = None,
        ik_position_tolerance_m: float = DEFAULT_IK_POSITION_TOLERANCE_M,
        ik_orientation_weight: float = DEFAULT_IK_ORIENTATION_WEIGHT,
        limits_label: str = "configured joint limits",
        state_key: str = "joint_pos",
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._ik_position_tolerance_m = _positive_finite(
            "ik_position_tolerance_m", ik_position_tolerance_m
        )
        self._ik_orientation_weight = _positive_finite(
            "ik_orientation_weight", ik_orientation_weight
        )
        low = np.asarray(joint_low, dtype=np.float64).reshape(-1)
        high = np.asarray(joint_high, dtype=np.float64).reshape(-1)
        if (
            arm_dof < 1
            or low.shape != (arm_dof,)
            or high.shape != (arm_dof,)
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low >= high)
        ):
            raise ValueError(f"{label} IK joint limits are invalid")
        self._arm = arm
        self._label = label
        self._limits_label = limits_label
        self._arm_dof = arm_dof
        self._low = low
        self._high = high
        self._state_key = state_key
        self._depth = depth_source
        self._depth_camera = depth_camera
        self._extrinsics = None if extrinsics is None else np.array(extrinsics, dtype=np.float64)
        self._log = log if log is not None else _stderr

        base = self._base_info()
        cameras = base.observation_space.camera_names
        if depth_camera not in cameras:
            raise ValueError(
                f"depth_camera {depth_camera!r} is not a configured camera; "
                f"available: {sorted(cameras)}"
            )
        fields = base.observation_space.state.fields if base.observation_space.state else ()
        observation_space = replace(
            base.observation_space,
            state=StateSpec(
                fields=(
                    *fields,
                    StateField(key="eef_pos", shape=(3,), unit="m"),
                    StateField(key="eef_quat_wxyz", shape=(4,), unit="unit_quaternion"),
                )
            ),
            state_keys=frozenset({*base.observation_space.state_keys, "eef_pos", "eef_quat_wxyz"}),
        )
        others = sorted(set(cameras) - {depth_camera})
        self.info = replace(
            base,
            name=name,
            observation_space=observation_space,
            docs=self._kinematics_docs() + self._perception_docs(others) + (base.docs or ""),
        )

    # -- hooks ----------------------------------------------------------------------

    @property
    def arm_dof(self) -> int:
        """Return the number of arm joints ``solve_ik`` returns (gripper excluded)."""
        return self._arm_dof

    @abstractmethod
    def _fk(self, joints: Vec) -> Pose:
        """Return the 4x4 base-frame tool pose of ``arm_dof`` arm joints."""

    @abstractmethod
    def _iterate_ik(
        self, seed: Vec, target: Pose, orientation_weight: float
    ) -> tuple[Vec, Pose, float]:
        """Run one solver pass and return ``(joints, achieved_pose, position_error_m)``.

        Raise ``ValueError`` (mentioning "invalid joint values") when the solver
        returns a non-finite or misshapen vector.
        """

    @abstractmethod
    def _kinematics_docs(self) -> str:
        """Return the robot-specific prompt notes: joints, units, frames, tool axes."""

    def _base_info(self) -> EmbodimentInfo:
        """Return the single-arm ``EmbodimentInfo`` this wrapper extends."""
        info: EmbodimentInfo = self._arm.info
        return info

    def _project(self, observation: Observation) -> Observation:
        """Map the wrapped arm's observation onto the single-arm profile."""
        return observation

    def _expand(self, action: Action) -> Action:
        """Map a single-arm action onto the wrapped arm's full command."""
        return action

    def _arm_joints(self, observation: Observation) -> Vec:
        """Return the ``arm_dof`` arm joints of a projected observation."""
        joints = np.asarray(observation.state[self._state_key], dtype=np.float64)
        if joints.shape != (self._arm_dof + 1,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"{self._label} joint observation is invalid")
        return joints[: self._arm_dof]

    # -- shared behaviour ---------------------------------------------------------

    def _perception_docs(self, other_cameras: list[str]) -> str:
        depth_camera = self._depth_camera
        if self._extrinsics is not None:
            return (
                "obs['extrinsics'] is the calibrated 4x4 camera-to-base transform; map a "
                "camera-frame point p with (obs['extrinsics'] @ [*p, 1])[:3] and a grasp pose "
                "G with obs['extrinsics'] @ G. "
            )
        return (
            "CLOSED-LOOP MODE: no camera-to-base calibration exists, so obs['extrinsics'] "
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
            + (
                f" in camera {other_cameras[0]!r} (segment(..., camera={other_cameras[0]!r}))"
                if other_cameras
                else ""
            )
            + "; after lifting, confirm the object moved with the gripper. "
        )

    def __getattr__(self, name: str) -> Any:
        """Preserve optional hooks of the wrapped arm, including operator-session integration."""
        return getattr(self._arm, name)

    def _augment(self, observation: Observation) -> Observation:
        pose = self._fk(self._arm_joints(observation))
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
                "eef_quat_wxyz": quat_wxyz_from_matrix(pose[:3, :3]),
            },
            extra=extra,
        )

    def solve_ik(
        self, position: npt.ArrayLike, quaternion_wxyz: npt.ArrayLike, seed: npt.ArrayLike
    ) -> Vec:
        """Return ``arm_dof`` arm joints reaching a base-frame tool pose.

        Each pass runs the subclass solver from ``seed``. An unreachable
        orientation would otherwise drag the tool off target, so the
        orientation weight is relaxed tenfold per pass (down to position-only)
        until the position fits. Results outside the joint limits or beyond
        the position tolerance raise; the CaP-X sandbox turns that into stderr
        the model can react to. A large orientation residual is only noted.
        """
        target_position = np.asarray(position, dtype=np.float64).reshape(-1)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
        start = np.asarray(seed, dtype=np.float64).reshape(-1)
        if (
            target_position.shape != (3,)
            or quaternion.shape != (4,)
            or start.shape != (self._arm_dof,)
            or not np.all(np.isfinite(target_position))
            or not np.all(np.isfinite(start))
        ):
            raise ValueError(
                f"solve_ik needs a finite position (3,), quaternion (4,), seed ({self._arm_dof},)"
            )
        target = pose_from_quat_wxyz(target_position, quaternion)
        weight = self._ik_orientation_weight
        for orientation_weight in (weight, weight / 10, weight / 100, 0.0):
            joints, achieved, error = self._iterate_ik(start, target, orientation_weight)
            if error <= self._ik_position_tolerance_m:
                break
        if error > self._ik_position_tolerance_m:
            rounded = np.round(target_position, 4).tolist()
            raise ValueError(
                f"{self._label} IK cannot reach position {rounded} m: residual "
                f"{error * 1000:.1f} mm exceeds {self._ik_position_tolerance_m * 1000:.1f} mm "
                "(out of reach or limits)"
            )
        if np.any(joints < self._low) or np.any(joints > self._high):
            raise ValueError(f"{self._label} IK solution exceeds the {self._limits_label}")
        missed = rotation_angle_deg(achieved[:3, :3], target[:3, :3])
        if missed > ORIENTATION_NOTE_DEG:
            self._log(
                f"solve_ik note: orientation residual {missed:.0f} deg; achieved "
                f"quaternion_wxyz {np.round(quat_wxyz_from_matrix(achieved[:3, :3]), 3).tolist()}"
            )
        return joints

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset the wrapped arm, prepare depth on the live camera, and augment."""
        observation = self._project(self._arm.reset(scene, seed=seed))
        self._depth.install()
        return self._augment(observation)

    def step(self, action: Action) -> StepResult:
        """Send the expanded joint target and augment the measured observation."""
        result: StepResult = self._arm.step(self._expand(action))
        return replace(result, observation=self._augment(self._project(result.observation)))

    def close(self) -> None:
        """Close the wrapped arm (its own shutdown motion and disconnect)."""
        self._arm.close()


__all__ = [
    "DEFAULT_IK_ORIENTATION_WEIGHT",
    "DEFAULT_IK_POSITION_TOLERANCE_M",
    "IK_CONVERGED_M",
    "IK_MAX_ITERATIONS",
    "ORIENTATION_NOTE_DEG",
    "CapxArmEmbodiment",
    "DepthSource",
    "load_extrinsics",
    "pose_from_quat_wxyz",
    "quat_wxyz_from_matrix",
    "rotation_angle_deg",
]
