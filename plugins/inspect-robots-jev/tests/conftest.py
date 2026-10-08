"""Synthetic Batch 04 geometry and motion fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspect_robots_jev.contract import DecodedInput
from inspect_robots_jev.geometry import Calibration, Localization
from inspect_robots_jev.kinematics import EndEffectorPose, IKResult
from inspect_robots_jev.motion import MotionPlanner


class SyntheticKinematics:
    mjcf_sha256 = "synthetic-model"

    def __init__(self) -> None:
        self.fail_ik = False

    def forward(self, side: str, q: np.ndarray) -> EndEffectorPose:
        start = 0 if side == "left" else 7
        y = 0.2 if side == "left" else -0.2
        return EndEffectorPose(np.array([0.6 + q[start], y + q[start + 1], 0.2 + q[start + 2]]),
                               np.eye(3), side, ("synthetic",), self.mjcf_sha256)

    def link_positions(self, side: str, q: np.ndarray) -> np.ndarray:
        start = 0 if side == "left" else 7
        y = 0.2 if side == "left" else -0.2
        return np.stack([np.linspace(0.1, 0.6, 6) + q[start],
                         np.full(6, y + q[start + 1]),
                         np.full(6, 0.2 + q[start + 2])], axis=1)

    def inverse(self, side: str, target_position: np.ndarray,
                target_rotation: np.ndarray, seed: np.ndarray) -> IKResult:
        if self.fail_ik:
            return IKResult(False, None, "non_converged", 10, 0.1, 0.0, self.mjcf_sha256)
        q = seed.copy()
        start = 0 if side == "left" else 7
        q[start:start + 3] = np.asarray(target_position) - [0.6, 0.2 if side == "left" else -0.2, 0.2]
        return IKResult(True, q, None, 1, 0.0, 0.0, self.mjcf_sha256)


@pytest.fixture
def synthetic_kinematics() -> SyntheticKinematics:
    return SyntheticKinematics()


@pytest.fixture
def calibration() -> Calibration:
    path = Path(__file__).parent / "fixtures" / "synthetic_calibration_v1.json"
    return Calibration.load(path)


@pytest.fixture
def motion(synthetic_kinematics: SyntheticKinematics, calibration: Calibration) -> MotionPlanner:
    return MotionPlanner(synthetic_kinematics, calibration)


@pytest.fixture
def q0() -> np.ndarray:
    return np.r_[np.zeros(6), 1.0, np.zeros(6), 1.0]


def observation(q: np.ndarray, timestamp: float = 1.0) -> DecodedInput:
    images = {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in ("top_cam", "left_cam", "right_cam")}
    return DecodedInput(images, q.copy(), "place objects in box",
                        {name: timestamp for name in images}, timestamp)


def location(category: str, xyz: tuple[float, float, float], *, usable: bool = True) -> Localization:
    return Localization(category, usable, np.asarray(xyz, dtype=float) if usable else None,
                        None if usable else "high_uncertainty", 0.005 if usable else None,
                        ("top_cam",) if usable else (), (), (), "synthetic-v1", 1,
                        "synthetic-model", (), np.array([0.05, 0.04]) if category == "box" and usable else None)
