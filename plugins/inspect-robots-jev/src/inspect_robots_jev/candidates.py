"""Finite, inspectable task candidates and observation-gated stage progression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import numpy.typing as npt

from inspect_robots import ActionChunk

from .contract import DecodedInput, hold
from .geometry import Localization
from .kinematics import Side
from .motion import MotionPlanner, MotionResult


class Stage(str, Enum):
    APPROACH = "approach"
    ALIGN = "align"
    DESCEND = "descend"
    CLOSE = "close_gripper"
    LIFT = "lift"
    MOVE_BOX = "move_box"
    RELEASE = "release"
    REOBSERVE = "reobserve"


_NEXT = {Stage.APPROACH: Stage.ALIGN, Stage.ALIGN: Stage.DESCEND,
         Stage.DESCEND: Stage.CLOSE, Stage.CLOSE: Stage.LIFT,
         Stage.LIFT: Stage.MOVE_BOX, Stage.MOVE_BOX: Stage.RELEASE,
         Stage.RELEASE: Stage.REOBSERVE}
_OBJECTS = ("red_block", "yellow_ball")


@dataclass(frozen=True, eq=False)
class Candidate:
    id: str
    stage: Stage
    category: str | None
    side: Side | None
    operation: str
    eef_target_m: tuple[float, float, float] | None
    gripper_opening: float | None
    motion: MotionResult | None
    summary: Mapping[str, object]
    reference_target_m: tuple[float, float, float] | None = None

    def jev_summary(self) -> dict[str, object]:
        """Only detached JSON metadata; never observations or simulator objects."""
        source = "molmo" if self.operation == "molmo" else "eef" if self.motion else "control"
        return {"id": self.id, "stage": self.stage.value, "category": self.category,
                "side": self.side, "operation": self.operation,
                "source": source,
                "span": {"start": 0, "steps": len(self.motion.trajectory)} if self.motion else None,
                "eef_target_m": list(self.eef_target_m) if self.eef_target_m else None,
                "gripper_opening": self.gripper_opening, "summary": dict(self.summary),
                "trajectory_steps": len(self.motion.trajectory) if self.motion else 0}

    def chunk(self, joint_pos: npt.ArrayLike) -> ActionChunk:
        if self.operation in ("hold", "reobserve"):
            return hold(joint_pos, self.operation)
        if self.motion is None or not self.motion.safe:
            raise ValueError("filtered candidate cannot be dispatched")
        return self.motion.chunk(self.id)


@dataclass(frozen=True, eq=False)
class CandidateSet:
    available: tuple[Candidate, ...]
    filtered: Mapping[str, str]
    stage: Stage

    def by_id(self, candidate_id: str) -> Candidate:
        return next(item for item in self.available if item.id == candidate_id)

    def fallback(self, joint_pos: npt.ArrayLike) -> ActionChunk:
        return hold(joint_pos, "no_safe_candidate")


@dataclass(frozen=True, eq=False)
class _Pending:
    candidate: Candidate
    state_time: float
    image_times: tuple[float, ...]
    start_q: npt.NDArray[np.float64]
    target_position: npt.NDArray[np.float64] | None
    target_rotation: npt.NDArray[np.float64] | None


class CandidateGenerator:
    """Shared Batch 04 planner; a later Policy decides which ID to dispatch."""

    def __init__(self, motion: MotionPlanner) -> None:
        self.motion = motion
        self.stage = Stage.APPROACH
        self.category: str | None = None
        self.side: Side | None = None
        self._pending: _Pending | None = None
        self._latest_set: CandidateSet | None = None
        self._latest_q: npt.NDArray[np.float64] | None = None
        self._latest_times: tuple[float, ...] | None = None

    def reset(self) -> None:
        self.stage = Stage.APPROACH
        self.category = None
        self.side = None
        self._pending = None
        self._latest_set = None
        self._latest_q = None
        self._latest_times = None

    def _updated(self, decoded: DecodedInput, pending: _Pending) -> bool:
        return decoded.state_time > pending.state_time and all(
            decoded.image_times[name] > stamp for name, stamp in zip(
                ("top_cam", "left_cam", "right_cam"), pending.image_times))

    def observe(self, decoded: DecodedInput, locations: Mapping[str, Localization]) -> None:
        pending = self._pending
        if pending is None or not self._updated(decoded, pending):
            return
        self._pending = None
        if pending.candidate.stage == Stage.REOBSERVE:
            self.reset()
            return
        candidate = pending.candidate
        if candidate.side is None or candidate.category is None:
            return
        if candidate.stage in (Stage.APPROACH, Stage.ALIGN, Stage.DESCEND, Stage.CLOSE):
            target = locations.get(candidate.category)
            if not self._usable(target) or candidate.reference_target_m is None:
                return
            assert target is not None
            if np.linalg.norm(target.candidate_position() - candidate.reference_target_m) > 0.03:
                return
        if candidate.stage in (Stage.MOVE_BOX, Stage.RELEASE):
            box = locations.get("box")
            if not self._usable(box) or candidate.reference_target_m is None:
                return
            assert box is not None
            if np.linalg.norm(box.candidate_position() - candidate.reference_target_m) > 0.03:
                return
        q = decoded.joint_pos
        if candidate.operation == "gripper":
            index = 6 if candidate.side == "left" else 13
            reached = abs(q[index] - float(candidate.gripper_opening)) <= 0.05
            changed = abs(q[index] - pending.start_q[index]) > 0.02
        else:
            pose = self.motion.kinematics.forward(candidate.side, q)
            start = self.motion.kinematics.forward(candidate.side, pending.start_q)
            reached = (np.linalg.norm(pose.position_base_m - pending.target_position) <= 0.025
                       and np.linalg.norm(pose.rotation_base_eef - pending.target_rotation) <= 0.1)
            changed = np.linalg.norm(pose.position_base_m - start.position_base_m) > 0.005
        if reached and changed:
            self.stage = _NEXT[candidate.stage]

    def mark_dispatched(self, candidate: Candidate, decoded: DecodedInput,
                        candidates: CandidateSet) -> None:
        if (candidates is not self._latest_set or candidates.stage != self.stage
                or self._latest_q is None or not np.array_equal(decoded.joint_pos, self._latest_q)
                or self._latest_times != (decoded.state_time, *(decoded.image_times[name]
                    for name in ("top_cam", "left_cam", "right_cam")))
                or not any(item is candidate for item in candidates.available)):
            raise ValueError("candidate is not from the current observation")
        if candidate.operation == "hold":
            return
        if candidate.operation == "reobserve" and self.stage != Stage.REOBSERVE:
            return
        if self._pending is not None:
            raise ValueError("awaiting a new observation before another dispatch")
        target_rotation = None
        target_position = None
        if candidate.operation == "eef" and candidate.side is not None:
            target_rotation = self.motion.kinematics.forward(candidate.side, decoded.joint_pos).rotation_base_eef
            target_position = np.asarray(candidate.eef_target_m, dtype=np.float64)
        self.category = candidate.category
        self.side = candidate.side
        self._pending = _Pending(candidate, decoded.state_time,
                                 tuple(decoded.image_times[name] for name in ("top_cam", "left_cam", "right_cam")),
                                 decoded.joint_pos.copy(), target_position, target_rotation)

    def generate(self, decoded: DecodedInput,
                 locations: Mapping[str, Localization]) -> CandidateSet:
        self.observe(decoded, locations)
        available: list[Candidate] = []
        filtered: dict[str, str] = {}
        if self._pending is None and self.stage != Stage.REOBSERVE:
            tasks = ((self.category, self.side),) if self.category and self.side else (
                (category, side) for category in _OBJECTS for side in ("left", "right"))
            for category, side in tasks:
                assert category is not None and side is not None
                key = f"{category}:{side}:{self.stage.value}"
                target = locations.get(category)
                if self.stage in (Stage.APPROACH, Stage.ALIGN, Stage.DESCEND, Stage.CLOSE):
                    quality_reason = self._quality_reason(target)
                    if quality_reason is not None:
                        filtered[key] = quality_reason
                        continue
                box = locations.get("box")
                quality_reason = self._quality_reason(box)
                if quality_reason is not None:
                    filtered[key] = "unreliable_box:" + quality_reason
                    continue
                pose = self.motion.kinematics.forward(side, decoded.joint_pos)
                position: npt.NDArray[np.float64] | None = None
                opening: float | None = None
                if self.stage in (Stage.APPROACH, Stage.ALIGN, Stage.DESCEND):
                    assert target is not None
                    position = target.candidate_position()
                    assert position is not None
                    position[2] += {Stage.APPROACH: 0.12, Stage.ALIGN: 0.09, Stage.DESCEND: 0.045}[self.stage]
                elif self.stage == Stage.LIFT:
                    position = pose.position_base_m.copy()
                    position[2] += 0.12
                elif self.stage == Stage.MOVE_BOX:
                    assert box is not None
                    position = box.candidate_position()
                    assert position is not None
                    position[2] += 0.06
                elif self.stage == Stage.CLOSE:
                    opening = 0.0
                elif self.stage == Stage.RELEASE:
                    opening = 1.0
                if position is not None:
                    motion = self.motion.plan_pose(decoded.joint_pos, side, position,
                                                   pose.rotation_base_eef, box=box)
                    operation = "eef"
                else:
                    assert opening is not None
                    motion = self.motion.plan_gripper(decoded.joint_pos, side, opening, box=box)
                    operation = "gripper"
                if not motion.safe:
                    filtered[key] = motion.reason or "unsafe"
                    continue
                quality = (target if self.stage in (Stage.APPROACH, Stage.ALIGN, Stage.DESCEND, Stage.CLOSE)
                           else box if self.stage in (Stage.MOVE_BOX, Stage.RELEASE) else None)
                summary: dict[str, object] = {"risk": "checked", "quality": "usable" if quality else "not_required",
                                               "uncertainty_m": quality.uncertainty_m if quality else None,
                                               "source_cameras": list(quality.sources) if quality else [],
                                               "calibration_id": quality.calibration_id if quality else None,
                                               "mjcf_sha256": self.motion.kinematics.mjcf_sha256}
                available.append(Candidate(key, self.stage, category, side, operation,
                                           tuple(float(x) for x in position) if position is not None else None,
                                           opening, motion, summary,
                                           tuple(float(x) for x in quality.position_base_m)
                                           if quality is not None and quality.position_base_m is not None else None))
        available.extend((Candidate("hold", self.stage, None, None, "hold", None, None, None,
                                    {"risk": "none", "quality": "no_motion"}),
                          Candidate("reobserve", self.stage, self.category, self.side, "reobserve", None,
                                    None, None, {"risk": "none", "quality": "new_observation_required"})))
        result = CandidateSet(tuple(available), filtered, self.stage)
        self._latest_set = result
        self._latest_q = decoded.joint_pos.copy()
        self._latest_times = (decoded.state_time, *(decoded.image_times[name]
            for name in ("top_cam", "left_cam", "right_cam")))
        return result

    def _usable(self, target: Localization | None) -> bool:
        return self._quality_reason(target) is None

    def _quality_reason(self, target: Localization | None) -> str | None:
        if target is None:
            return "missing_localization"
        position = target.candidate_position()
        if position is None:
            return target.reason or "unreliable_target"
        if position.shape != (3,) or not np.isfinite(position).all():
            return "invalid_localization"
        if (target.calibration_id != self.motion.calibration.calibration_id
                or target.calibration_version != 1
                or target.mjcf_sha256 not in (None, self.motion.kinematics.mjcf_sha256)):
            return "provenance_mismatch"
        if target.category == "box":
            opening = target.opening_half_extents_m
            if (opening is None or opening.shape != (2,) or not np.isfinite(opening).all()
                    or np.any(opening <= 0)):
                return "invalid_box_opening"
        return None
