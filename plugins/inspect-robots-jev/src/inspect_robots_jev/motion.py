"""Short absolute YAM trajectories with whole-path geometric safety gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from inspect_robots import Action, ActionChunk
from inspect_robots_jev.yam_contract import action_space

from .geometry import Calibration, Localization
from .kinematics import Side, YamKinematics


@dataclass(frozen=True)
class MotionLimits:
    max_steps: int = 6
    max_arm_step_rad: float = 0.15
    max_gripper_step: float = 0.25
    interpolation_step_rad: float = 0.025
    interpolation_gripper_step: float = 0.05
    link_radius_m: float = 0.025
    table_margin_m: float = 0.015
    box_wall_margin_m: float = 0.015

    def __post_init__(self) -> None:
        values = (self.max_arm_step_rad, self.max_gripper_step,
                  self.interpolation_step_rad, self.interpolation_gripper_step,
                  self.link_radius_m, self.table_margin_m, self.box_wall_margin_m)
        if self.max_steps < 1 or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("motion limits must be positive and finite")


@dataclass(frozen=True, eq=False)
class MotionResult:
    trajectory: tuple[npt.NDArray[np.float64], ...]
    reason: str | None = None
    detail: str | None = None

    @property
    def safe(self) -> bool:
        return self.reason is None and bool(self.trajectory)

    def chunk(self, candidate_id: str) -> ActionChunk:
        if not self.safe:
            raise ValueError("filtered motion cannot be sent")
        return ActionChunk(actions=[Action(data=q.copy()) for q in self.trajectory],
                           meta={"candidate_id": candidate_id, "reobserve_after_chunk": True})


def _failed(reason: str, detail: str | None = None) -> MotionResult:
    return MotionResult((), reason, detail)


def _segment_distance(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
                      c: npt.NDArray[np.float64], d: npt.NDArray[np.float64]) -> float:
    """Closest distance between finite 3D link segments."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    denom = aa * cc - bb * bb
    s = 0.0 if denom < 1e-12 else float(np.clip((bb * ee - cc * dd) / denom, 0, 1))
    t = float(np.clip((bb * s + ee) / cc, 0, 1)) if cc > 1e-12 else 0.0
    s = float(np.clip((bb * t - dd) / aa, 0, 1)) if aa > 1e-12 else 0.0
    return float(np.linalg.norm(w + s * u - t * v))


class MotionPlanner:
    """Convert Batch 03 IK to bounded chunks and check every interpolated state."""

    def __init__(self, kinematics: YamKinematics, calibration: Calibration,
                 limits: MotionLimits = MotionLimits()) -> None:
        self.kinematics = kinematics
        self.calibration = calibration
        self.limits = limits

    def _collision(self, q: npt.NDArray[np.float64], box: Localization | None) -> str | None:
        try:
            left = self.kinematics.link_positions("left", q)
            right = self.kinematics.link_positions("right", q)
        except ValueError:
            return "joint_limit"
        if left.shape != (6, 3) or right.shape != (6, 3) or not np.isfinite(left).all() or not np.isfinite(right).all():
            return "invalid_kinematics"
        normal = self.calibration.table_normal_base
        floor = self.calibration.table_offset_m + self.limits.table_margin_m + self.limits.link_radius_m
        for links in (left, right):
            if np.any(links @ normal < floor):
                return "table_collision"
        for a, b in zip(left[:-1], left[1:]):
            for c, d in zip(right[:-1], right[1:]):
                if _segment_distance(a, b, c, d) < 2 * self.limits.link_radius_m:
                    return "arm_collision"
        if box is not None and box.usable and box.position_base_m is not None and box.opening_half_extents_m is not None:
            center = box.position_base_m
            inner = box.opening_half_extents_m
            outer = inner + self.calibration.box_lateral_margin_m + self.limits.box_wall_margin_m
            rim = center[2] - self.calibration.box_clearance_m
            for links in (left, right):
                for a, b in zip(links[:-1], links[1:]):
                    # Dense spatial sampling plus the configured wall margin covers
                    # the thin box rim in this short-link YAM model.
                    count = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.01)))
                    for fraction in np.linspace(0, 1, count + 1):
                        p = a + fraction * (b - a)
                        xy = np.abs(p[:2] - center[:2])
                        if p[2] <= rim + self.limits.link_radius_m and np.all(xy <= outer) and np.any(xy >= inner):
                            return "box_wall_collision"
        return None

    def check_trajectory(self, current: npt.ArrayLike, targets: Sequence[npt.ArrayLike],
                         side: Side | Literal["both"], *, box: Localization | None = None) -> MotionResult:
        if side not in ("left", "right", "both"):
            return _failed("invalid_side")
        if not targets or len(targets) > self.limits.max_steps:
            return _failed("invalid_horizon")
        try:
            previous = np.asarray(current, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return _failed("invalid_shape_or_value")
        if previous.shape != (14,) or not np.isfinite(previous).all():
            return _failed("invalid_shape_or_value")
        space = action_space()
        assert space.low is not None and space.high is not None
        if np.any(previous[[6, 13]] < 0) or np.any(previous[[6, 13]] > 1):
            return _failed("invalid_gripper_encoding")
        if np.any(previous < space.low) or np.any(previous > space.high):
            return _failed("joint_limit")
        inactive = slice(7, 14) if side == "left" else slice(0, 7) if side == "right" else None
        checked: list[npt.NDArray[np.float64]] = []
        for value in targets:
            try:
                q = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                return _failed("invalid_shape_or_value")
            if q.shape != (14,) or not np.isfinite(q).all():
                return _failed("invalid_shape_or_value")
            if np.any(q[[6, 13]] < 0) or np.any(q[[6, 13]] > 1):
                return _failed("invalid_gripper_encoding")
            if np.any(q < space.low) or np.any(q > space.high):
                return _failed("joint_limit")
            if inactive is not None and not np.array_equal(q[inactive], previous[inactive]):
                return _failed("inactive_arm_changed")
            delta = np.abs(q - previous)
            if np.max(delta[np.r_[0:6, 7:13]]) > self.limits.max_arm_step_rad + 1e-10 or np.max(delta[[6, 13]]) > self.limits.max_gripper_step + 1e-10:
                return _failed("step_too_large")
            steps = max(1, int(np.ceil(np.max(delta[np.r_[0:6, 7:13]]) / self.limits.interpolation_step_rad)),
                        int(np.ceil(np.max(delta[[6, 13]]) / self.limits.interpolation_gripper_step)))
            for fraction in np.linspace(0, 1, steps + 1):
                reason = self._collision(previous + fraction * (q - previous), box)
                if reason is not None:
                    return _failed(reason)
            checked.append(q.copy())
            previous = q
        return MotionResult(tuple(checked))

    def _interpolate(self, current: npt.NDArray[np.float64], goal: npt.NDArray[np.float64],
                     side: Side, box: Localization | None) -> MotionResult:
        if goal.shape != (14,) or not np.isfinite(goal).all():
            return _failed("invalid_shape_or_value")
        delta = goal - current
        scaled = max(float(np.max(np.abs(delta[np.r_[0:6, 7:13]]) / self.limits.max_arm_step_rad)),
                     float(np.max(np.abs(delta[[6, 13]]) / self.limits.max_gripper_step)))
        if scaled < 1e-8:
            return _failed("already_at_target")
        steps = min(self.limits.max_steps, max(1, int(np.ceil(scaled))))
        fraction = min(1.0, self.limits.max_steps / scaled)
        endpoint = current + fraction * delta
        points = [current + (endpoint - current) * (i / steps) for i in range(1, steps + 1)]
        return self.check_trajectory(current, points, side, box=box)

    def plan_pose(self, current: npt.ArrayLike, side: Side, target_position: npt.ArrayLike,
                  target_rotation: npt.ArrayLike, *, box: Localization | None = None) -> MotionResult:
        try:
            q = np.asarray(current, dtype=np.float64)
            result = self.kinematics.inverse(side, target_position, target_rotation, q)
        except (TypeError, ValueError, OverflowError) as exc:
            return _failed("invalid_target", str(exc))
        if not result.success or result.joint_pos is None:
            return _failed("ik_" + (result.reason or "failed"))
        return self._interpolate(q, result.joint_pos, side, box)

    def plan_gripper(self, current: npt.ArrayLike, side: Side, opening: float,
                     *, box: Localization | None = None) -> MotionResult:
        try:
            q = np.asarray(current, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return _failed("invalid_shape_or_value")
        if q.shape != (14,) or not np.isfinite(q).all():
            return _failed("invalid_shape_or_value")
        if not math.isfinite(opening) or opening < 0 or opening > 1:
            return _failed("invalid_gripper_encoding")
        goal = q.copy()
        goal[6 if side == "left" else 13] = opening
        return self._interpolate(q, goal, side, box)
