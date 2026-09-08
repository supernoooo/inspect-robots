"""Static MolmoAct2-LIBERO action and observation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from inspect_robots import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec

ACTION_DIM = 7
STATE_DIM = 8
STATE_KEY = "state"
CAMERA_NAMES = ("image", "wrist_image")
RAW_CAMERA_NAMES = ("agentview_image", "robot0_eye_in_hand_image")
DIM_LABELS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def action_space() -> Box:
    """Return LIBERO's normalized 7-D relative end-effector action space."""
    return Box(
        shape=(ACTION_DIM,),
        low=np.full(ACTION_DIM, -1.0, dtype=np.float64),
        high=np.full(ACTION_DIM, 1.0, dtype=np.float64),
        semantics=ActionSemantics(
            control_mode="eef_delta_pose",
            rotation_repr="axis_angle",
            gripper="continuous",
            frame="base",
            dim_labels=DIM_LABELS,
        ),
    )


def observation_space(height: int = 256, width: int = 256) -> ObservationSpace:
    """Return the image, wrist image, and 8-D robot state expected by the model."""
    return ObservationSpace(
        cameras=tuple(CameraSpec(name, height, width, 3) for name in CAMERA_NAMES),
        state=StateSpec(
            fields=(
                StateField(
                    STATE_KEY,
                    (STATE_DIM,),
                    unit="m+axis_angle+gripper_joint_pos",
                    dtype="float32",
                ),
            )
        ),
    )


def quat_xyzw_to_axis_angle(quaternion: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Convert one LIBERO quaternion in ``xyzw`` order to a 3-D axis-angle vector."""
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {quat.shape}")
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math_sqrt_clamped(1.0 - w * w)
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return np.asarray(quat[:3] / denominator * angle, dtype=np.float32)


def math_sqrt_clamped(value: float) -> float:
    """Square-root a scalar after clamping tiny negative round-off to zero."""
    return float(np.sqrt(max(value, 0.0)))


def robot_state(raw: Mapping[str, object]) -> npt.NDArray[np.float32]:
    """Build the official 8-D LIBERO state from a raw simulator observation."""
    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in raw or raw[key] is None]
    if missing:
        raise ValueError(f"LIBERO observation is missing robot state fields: {missing}")
    position = np.asarray(raw[required[0]], dtype=np.float32).reshape(-1)
    quaternion = np.asarray(raw[required[1]], dtype=np.float32).reshape(-1)
    gripper = np.asarray(raw[required[2]], dtype=np.float32).reshape(-1)
    if position.shape != (3,) or gripper.shape != (2,):
        raise ValueError(
            f"unexpected LIBERO state shapes: eef_pos={position.shape}, gripper={gripper.shape}"
        )
    return cast(
        npt.NDArray[np.float32],
        np.concatenate((position, quat_xyzw_to_axis_angle(quaternion), gripper)).astype(np.float32),
    )
