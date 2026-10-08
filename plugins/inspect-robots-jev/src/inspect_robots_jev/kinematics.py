"""Static MJCF kinematics for the YAM 14-value wire state; no simulator state."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import numpy.typing as npt

GRIPPER_OPEN_POSITION = -0.0475  # Static MJCF finger hinge, not the YAM wire gripper.

Side = Literal["left", "right"]


def _wire(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError("joint_pos must be a finite 14-D wire state")
    if np.any(np.abs(q[np.r_[0:6, 7:13]]) > np.pi) or np.any(q[[6, 13]] < 0) or np.any(q[[6, 13]] > 1):
        raise ValueError("joint_pos exceeds wire limits")
    return q.copy()


@dataclass(frozen=True, eq=False)
class EndEffectorPose:
    position_base_m: npt.NDArray[np.float64]
    rotation_base_eef: npt.NDArray[np.float64]
    side: Side
    transform_chain: tuple[str, ...]
    mjcf_sha256: str


@dataclass(frozen=True, eq=False)
class IKResult:
    success: bool
    joint_pos: npt.NDArray[np.float64] | None
    reason: str | None
    iterations: int
    position_error_m: float
    rotation_error_rad: float
    mjcf_sha256: str

    def diagnostic(self) -> dict[str, object]:
        return {"success": self.success, "reason": self.reason, "iterations": self.iterations,
                "position_error_m": self.position_error_m,
                "rotation_error_rad": self.rotation_error_rad, "mjcf_sha256": self.mjcf_sha256}


def _rotation_error(target: npt.NDArray[np.float64], current: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """World-frame logarithm of target * current.T, stable near pi."""
    matrix = target @ current.T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.ravel())
    if quat[0] < 0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2 * np.arctan2(vector_norm, quat[0])
    return quat[1:] * angle / vector_norm


class YamKinematics:
    """FK/Jacobian/local pose IK backed by a user-supplied static YAM MJCF.

    The wire order is left joints 1..6, normalized left gripper, right joints
    1..6, normalized right gripper. Each gripper drives both MJCF finger joints
    to ``GRIPPER_OPEN_POSITION * wire_gripper``. The EEF body defaults to
    ``left_link_6`` / ``right_link_6`` and can be named explicitly.
    """

    def __init__(self, mjcf_path: str | Path, *, eef_bodies: dict[Side, str] | None = None) -> None:
        path = Path(mjcf_path)
        raw = path.read_bytes()
        if ET.fromstring(raw).find(".//include") is not None:
            raise ValueError("MJCF must be flattened so its fingerprint covers all kinematic definitions")
        self.mjcf_sha256 = hashlib.sha256(raw).hexdigest()
        self.model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bimanual_base")
        if self.base_body < 0 or self.model.body_jntnum[self.base_body]:
            raise ValueError("MJCF needs a fixed bimanual_base body")
        names = eef_bodies or {"left": "left_link_6", "right": "right_link_6"}
        if set(names) != {"left", "right"}:
            raise ValueError("eef_bodies must name both arms")
        self.body_ids: dict[Side, int] = {}
        self.arm_qpos: dict[Side, tuple[int, ...]] = {}
        self.arm_dofs: dict[Side, tuple[int, ...]] = {}
        self.arm_joint_ids: dict[Side, tuple[int, ...]] = {}
        self.finger_qpos: dict[Side, tuple[int, ...]] = {}
        for side in ("left", "right"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, names[side])
            if body_id < 0:
                raise ValueError(f"MJCF missing EEF body {names[side]}")
            self.body_ids[side] = body_id
            joints = tuple(self._joint_id(f"{side}_joint{i}") for i in range(1, 7))
            fingers = tuple(self._joint_id(f"{side}_{finger}_finger") for finger in ("left", "right"))
            self.arm_joint_ids[side] = joints
            self.arm_qpos[side] = tuple(int(self.model.jnt_qposadr[j]) for j in joints)
            self.arm_dofs[side] = tuple(int(self.model.jnt_dofadr[j]) for j in joints)
            self.finger_qpos[side] = tuple(int(self.model.jnt_qposadr[j]) for j in fingers)
            if len(set(self.arm_qpos[side])) != 6:
                raise ValueError("MJCF arm joint mapping is not unique")
            # The named EEF must actually descend from every named arm joint.
            ancestors: set[int] = set()
            current = body_id
            while current > 0:
                ancestors.add(current)
                current = int(self.model.body_parentid[current])
            if any(int(self.model.jnt_bodyid[j]) not in ancestors for j in joints):
                raise ValueError(f"{names[side]} is not downstream of all {side} arm joints")

    def _joint_id(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"MJCF missing hinge joint {name}")
        return joint

    def _data(self, wire: npt.ArrayLike) -> mujoco.MjData:
        q = _wire(wire)
        data = mujoco.MjData(self.model)
        for side, start, gripper_index in (("left", 0, 6), ("right", 7, 13)):
            for index, (joint, address) in enumerate(zip(self.arm_joint_ids[side], self.arm_qpos[side])):
                value = q[start + index]
                if self.model.jnt_limited[joint]:
                    lo, hi = self.model.jnt_range[joint]
                    if value < lo - 1e-9 or value > hi + 1e-9:
                        raise ValueError(f"{side}_joint{index + 1} exceeds MJCF limits")
                data.qpos[address] = value
            for address in self.finger_qpos[side]:
                data.qpos[address] = GRIPPER_OPEN_POSITION * q[gripper_index]
        mujoco.mj_forward(self.model, data)
        return data

    def forward(self, side: Side, joint_pos: npt.ArrayLike) -> EndEffectorPose:
        if side not in self.body_ids:
            raise ValueError("side must be left or right")
        data = self._data(joint_pos)
        body = self.body_ids[side]
        base_rotation = data.xmat[self.base_body].reshape(3, 3)
        position = base_rotation.T @ (data.xpos[body] - data.xpos[self.base_body])
        rotation = base_rotation.T @ data.xmat[body].reshape(3, 3)
        return EndEffectorPose(position.copy(), rotation.copy(), side,
                               ("wire_joint_pos", "MJCF_base", f"{side}_eef"), self.mjcf_sha256)

    def link_positions(self, side: Side, joint_pos: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Arm link centers in the base frame, for conservative swept-link checks."""
        if side not in self.body_ids:
            raise ValueError("side must be left or right")
        data = self._data(joint_pos)
        base_rotation = data.xmat[self.base_body].reshape(3, 3)
        base_position = data.xpos[self.base_body]
        bodies = [int(self.model.jnt_bodyid[joint]) for joint in self.arm_joint_ids[side]]
        return np.stack([base_rotation.T @ (data.xpos[body] - base_position) for body in bodies])

    def jacobian(self, side: Side, joint_pos: npt.ArrayLike) -> npt.NDArray[np.float64]:
        if side not in self.body_ids:
            raise ValueError("side must be left or right")
        data = self._data(joint_pos)
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, linear, angular, self.body_ids[side])
        base_rotation = data.xmat[self.base_body].reshape(3, 3)
        return np.vstack((base_rotation.T @ linear[:, self.arm_dofs[side]],
                          base_rotation.T @ angular[:, self.arm_dofs[side]]))

    def inverse(
        self, side: Side, target_position_base_m: npt.ArrayLike,
        target_rotation_base_eef: npt.ArrayLike, seed_joint_pos: npt.ArrayLike,
        *, max_iterations: int = 120, position_tolerance_m: float = 1e-4,
        rotation_tolerance_rad: float = 1e-3,
    ) -> IKResult:
        if side not in self.body_ids:
            raise ValueError("side must be left or right")
        target_pos = np.asarray(target_position_base_m, dtype=np.float64)
        target_rot = np.asarray(target_rotation_base_eef, dtype=np.float64)
        if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
            raise ValueError("target position must be finite 3-D")
        if target_rot.shape != (3, 3) or not np.isfinite(target_rot).all() or not np.allclose(target_rot.T @ target_rot, np.eye(3), atol=1e-5) or np.linalg.det(target_rot) < 0.999:
            raise ValueError("target rotation must be a proper 3x3 matrix")
        if max_iterations < 1 or position_tolerance_m <= 0 or rotation_tolerance_rad <= 0:
            raise ValueError("invalid IK iteration or tolerance")
        q = _wire(seed_joint_pos)
        start = 0 if side == "left" else 7
        last_pos = last_rot = float("inf")
        singular = False
        for iteration in range(max_iterations + 1):
            pose = self.forward(side, q)
            pos_error = target_pos - pose.position_base_m
            rot_error = _rotation_error(target_rot, pose.rotation_base_eef)
            last_pos, last_rot = float(np.linalg.norm(pos_error)), float(np.linalg.norm(rot_error))
            if last_pos <= position_tolerance_m and last_rot <= rotation_tolerance_rad:
                return IKResult(True, q.copy(), None, iteration, last_pos, last_rot, self.mjcf_sha256)
            if iteration == max_iterations:
                break
            jac = self.jacobian(side, q)
            singular = np.linalg.svd(jac, compute_uv=False)[-1] < 1e-5
            error = np.r_[pos_error, rot_error]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.0025 * np.eye(6), error)
            maximum = float(np.max(np.abs(step)))
            if maximum > 0.2:
                step *= 0.2 / maximum
            current_cost = float(np.linalg.norm(error))
            improved = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = q.copy()
                for index, joint in enumerate(self.arm_joint_ids[side]):
                    lo, hi = (-np.pi, np.pi)
                    if self.model.jnt_limited[joint]:
                        lo = max(lo, float(self.model.jnt_range[joint, 0]))
                        hi = min(hi, float(self.model.jnt_range[joint, 1]))
                    candidate[start + index] = np.clip(q[start + index] + fraction * step[index], lo, hi)
                candidate_pose = self.forward(side, candidate)
                candidate_cost = float(np.linalg.norm(np.r_[target_pos - candidate_pose.position_base_m,
                                          _rotation_error(target_rot, candidate_pose.rotation_base_eef)]))
                if candidate_cost < current_cost - 1e-10:
                    q = candidate
                    improved = True
                    break
            if not improved:
                break
        reason = "singular" if singular and last_pos <= 0.02 else ("unreachable" if last_pos > position_tolerance_m else "non_converged")
        return IKResult(False, None, reason, iteration, last_pos, last_rot, self.mjcf_sha256)
