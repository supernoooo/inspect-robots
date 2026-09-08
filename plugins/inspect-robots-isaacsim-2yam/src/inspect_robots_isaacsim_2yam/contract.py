"""Pure NumPy action and observation contracts for bimanual YAM simulation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from inspect_robots import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec

ARM_DOF = 6
ARM_WIDTH = 7
ACTION_DIM = 14
CAMERA_NAMES = ("top_cam", "left_cam", "right_cam")
STATE_KEY = "joint_pos"
GRIPPER_INDICES = (6, 13)
GRIPPER_OPEN_POSITION = -0.0475

DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(ARM_DOF)), "gripper")
)


def action_space() -> Box:
    """Return MolmoAct2-YAM's 14-D absolute joint target contract."""
    arm_low = (-np.pi,) * ARM_DOF + (0.0,)
    arm_high = (np.pi,) * ARM_DOF + (1.0,)
    return Box(
        shape=(ACTION_DIM,),
        low=np.asarray(arm_low * 2, dtype=np.float64),
        high=np.asarray(arm_high * 2, dtype=np.float64),
        semantics=ActionSemantics(
            control_mode="joint_pos",
            rotation_repr="none",
            gripper="continuous",
            frame="base",
            dim_labels=DIM_LABELS,
        ),
    )


def observation_space(height: int, width: int) -> ObservationSpace:
    """Return the three-camera and flat 14-D state contract used by the YAM server."""
    return ObservationSpace(
        cameras=tuple(CameraSpec(name, height, width, 3) for name in CAMERA_NAMES),
        state=StateSpec(fields=(StateField(STATE_KEY, (ACTION_DIM,), unit="rad+normalized"),)),
    )


def physical_joint_targets(
    action: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Split a wire action into arm targets and physical finger positions.

    The wire grippers use ``0=closed, 1=open``. The YAM MJCF finger joints use
    ``0=closed, -0.0475=open``.
    """
    vector = np.asarray(action, dtype=np.float64)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"expected action shape ({ACTION_DIM},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("action contains non-finite values")
    arms = np.concatenate((vector[:ARM_DOF], vector[ARM_WIDTH : ARM_WIDTH + ARM_DOF]))
    return (
        arms,
        float(GRIPPER_OPEN_POSITION * np.clip(vector[GRIPPER_INDICES[0]], 0.0, 1.0)),
        float(GRIPPER_OPEN_POSITION * np.clip(vector[GRIPPER_INDICES[1]], 0.0, 1.0)),
    )


def wire_state(
    left_arm: npt.ArrayLike,
    left_finger: npt.ArrayLike,
    right_arm: npt.ArrayLike,
    right_finger: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Pack Isaac joint positions into MolmoAct2-YAM's 14-D wire order."""
    left = np.asarray(left_arm, dtype=np.float64).reshape(-1)
    right = np.asarray(right_arm, dtype=np.float64).reshape(-1)
    if left.shape != (ARM_DOF,) or right.shape != (ARM_DOF,):
        raise ValueError("left_arm and right_arm must each contain six joint positions")
    left_gripper = np.clip(
        -float(np.asarray(left_finger).reshape(-1)[0]) / abs(GRIPPER_OPEN_POSITION), 0.0, 1.0
    )
    right_gripper = np.clip(
        -float(np.asarray(right_finger).reshape(-1)[0]) / abs(GRIPPER_OPEN_POSITION), 0.0, 1.0
    )
    return np.concatenate((left, [left_gripper], right, [right_gripper]))
