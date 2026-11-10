"""URDF arm kinematics for ``ik="embodiment"`` when no embodiment adapter supplies IK.

Some hosts (for example a remote policy server that only receives joint
states and images) cannot use an embodiment's own solver. This module gives
them the same forward kinematics and iterative IK the SO-101 adapter uses, in
joint degrees, from a local URDF. It needs the optional ``placo`` dependency:
``pip install "inspect-robots-capx[kinematics]"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: Position residual below which an IK pass stops iterating, in metres.
IK_CONVERGED_M = 5e-4
IK_MAX_ITERATIONS = 100

_Vec = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IkSolution:
    """Arm joints (degrees) and the residuals of the pose they reach."""

    joints: _Vec
    position_error_m: float
    orientation_error_deg: float


def _rotation_from_wxyz(quaternion_wxyz: _Vec) -> _Vec:
    norm = float(np.linalg.norm(quaternion_wxyz))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"quaternion_wxyz must be a unit quaternion, got norm {norm:.4f}")
    w, x, y, z = quaternion_wxyz / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_wxyz(rotation: _Vec) -> _Vec:
    """Return the unit quaternion (w, x, y, z) of a 3x3 rotation matrix."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out = np.asarray(q, dtype=np.float64)
    return out / np.linalg.norm(out)


class ArmKinematics:
    """Forward kinematics and iterative IK for one URDF tool frame, in degrees.

    ``joint_low_deg``/``joint_high_deg`` replace the URDF limits inside the
    solver (calibrated rigs are often wider or narrower than the URDF) and
    are enforced on every returned solution.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        tip_frame: str,
        joint_names: tuple[str, ...],
        joint_low_deg: Any = None,
        joint_high_deg: Any = None,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self.tip_frame = tip_frame
        if (joint_low_deg is None) != (joint_high_deg is None):
            raise ValueError("give both joint_low_deg and joint_high_deg, or neither")
        dof = len(self.joint_names)
        self.low: _Vec | None = None
        self.high: _Vec | None = None
        if joint_low_deg is not None:
            low = np.asarray(joint_low_deg, dtype=np.float64).reshape(-1)
            high = np.asarray(joint_high_deg, dtype=np.float64).reshape(-1)
            if (
                low.shape != (dof,)
                or high.shape != (dof,)
                or not np.all(np.isfinite(low))
                or not np.all(np.isfinite(high))
                or np.any(low >= high)
            ):
                raise ValueError(f"joint limits must be {dof} finite increasing pairs")
            self.low, self.high = low, high
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF is missing: {path}")
        try:
            import placo  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError(
                'URDF kinematics needs placo: pip install "inspect-robots-capx[kinematics]"'
            ) from exc
        self._robot = placo.RobotWrapper(str(path))
        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.mask_fbase(True)
        self._task = self._solver.add_frame_task(tip_frame, np.eye(4))
        if self.low is not None and self.high is not None:
            for name, lo, hi in zip(self.joint_names, self.low, self.high, strict=True):
                self._robot.set_joint_limits(name, math.radians(lo), math.radians(hi))

    def _set(self, joints_deg: _Vec) -> None:
        for name, value in zip(self.joint_names, np.deg2rad(joints_deg), strict=True):
            self._robot.set_joint(name, float(value))
        self._robot.update_kinematics()

    def forward(self, joints_deg: Any) -> _Vec:
        """Return the 4x4 base-frame pose of the tool frame for arm joints in degrees."""
        joints = np.asarray(joints_deg, dtype=np.float64).reshape(-1)[: len(self.joint_names)]
        if joints.shape != (len(self.joint_names),) or not np.all(np.isfinite(joints)):
            raise ValueError(f"forward() needs {len(self.joint_names)} finite joint values")
        self._set(joints)
        pose = np.asarray(self._robot.get_T_world_frame(self.tip_frame), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("forward kinematics returned an invalid pose")
        return pose

    def _iterate(
        self, seed: _Vec, target: _Vec, orientation_weight: float
    ) -> tuple[_Vec, _Vec, float]:
        joints = seed
        achieved = self.forward(joints)
        error = math.inf
        self._task.configure(self.tip_frame, "soft", 1.0, orientation_weight)
        for _ in range(IK_MAX_ITERATIONS):
            self._set(joints)
            self._task.T_world_frame = target
            self._solver.solve(True)
            self._robot.update_kinematics()
            solved = np.rad2deg([self._robot.get_joint(name) for name in self.joint_names])
            if not np.all(np.isfinite(solved)):
                raise ValueError("IK returned invalid joint values")
            joints = np.asarray(solved, dtype=np.float64)
            achieved = self.forward(joints)
            error = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
            if error <= IK_CONVERGED_M:
                break
        return joints, achieved, error

    def solve(
        self,
        position: Any,
        quaternion: Any,
        seed_deg: Any,
        *,
        position_tolerance_m: float = 0.005,
        orientation_weight: float = 0.1,
    ) -> IkSolution:
        """Reach a base-frame tool pose from ``seed_deg``; position wins over orientation.

        An unreachable orientation would otherwise drag the tool off target,
        so the orientation weight is relaxed tenfold per pass (to
        position-only) until the position fits. Raises ``ValueError`` when the
        position stays beyond ``position_tolerance_m`` or the solution leaves
        the configured joint limits.
        """
        target_position = np.asarray(position, dtype=np.float64).reshape(-1)
        quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
        seed = np.asarray(seed_deg, dtype=np.float64).reshape(-1)
        if (
            target_position.shape != (3,)
            or quat.shape != (4,)
            or seed.shape != (len(self.joint_names),)
            or not np.all(np.isfinite(target_position))
            or not np.all(np.isfinite(seed))
        ):
            raise ValueError(
                f"solve() needs a finite position (3,), quaternion (4,), seed ({len(seed)},)"
            )
        target = np.eye(4)
        target[:3, :3] = _rotation_from_wxyz(quat)
        target[:3, 3] = target_position
        for weight in (orientation_weight, orientation_weight / 10, orientation_weight / 100, 0.0):
            joints, achieved, error = self._iterate(seed, target, weight)
            if error <= position_tolerance_m:
                break
        if error > position_tolerance_m:
            raise ValueError(
                f"IK cannot reach position {np.round(target_position, 4).tolist()} m: residual "
                f"{error * 1000:.1f} mm exceeds {position_tolerance_m * 1000:.1f} mm "
                "(out of reach or limits)"
            )
        if (
            self.low is not None
            and self.high is not None
            and (np.any(joints < self.low) or np.any(joints > self.high))
        ):
            raise ValueError("IK solution exceeds the configured joint limits")
        relative = achieved[:3, :3] @ target[:3, :3].T
        cosine = float(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
        return IkSolution(joints, error, math.degrees(math.acos(cosine)))
