"""Real YAM 0.36.0 wire contract, exported from its hardware-free modules."""

from __future__ import annotations

import numpy as np

from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig, action_box
from inspect_robots_yam.config import observation_space as yam_observation_space
from inspect_robots_yam.packing import ARM_DOF, ARM_WIDTH, DIM_LABELS, STATE_KEY, TOTAL_DIM

ACTION_DIM = TOTAL_DIM
CAMERA_NAMES = DEFAULT_CAMERAS
GRIPPER_INDICES = (ARM_DOF, ARM_WIDTH + ARM_DOF)


def action_space():
    """Absolute joint targets with YAM's default 14-D safety bounds."""
    config = YamConfig()
    return action_box(low=np.asarray(config.joint_low, dtype=np.float64),
                      high=np.asarray(config.joint_high, dtype=np.float64),
                      control_interface="joints", joints_are_delta=False)


def observation_space(height: int, width: int):
    """Three RGB streams and the packed joint state at the rig's output size."""
    return yam_observation_space(height, width, CAMERA_NAMES, STATE_KEY,
                                 control_interface="joints")
