"""Local validation of Batch 01 proposals into short, selectable YAM actions.

This module only expands and checks in-memory chunks. It never owns a driver,
calls JEV, or executes an action against an embodiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from inspect_robots import Action, ActionChunk, Observation, PolicyInfo
from inspect_robots_agent._tools import build_toolset
from inspect_robots_agent.proposals import (
    ProposalBatch, ProposalCandidate, ProposalFailure, ProposalResult,
    ProposalTermination,
)
from inspect_robots_yam.collision import CollisionChecker, _config_from_yam
from inspect_robots_yam.config import YamConfig, action_box

from .contract import InputError, decode_observation, hold
from .geometry import BASE_FRAME, CALIBRATION_VERSION
from .kinematics import YamKinematics
from .motion import MotionPlanner
from .pairing import strict_yam_preflight
from .yam_contract import CAMERA_NAMES, DIM_LABELS, observation_space


@dataclass(frozen=True, eq=False)
class ValidatedActionCandidate:
    """Only the locally checked prefix is retained; no suffix is exposed."""

    id: str
    chunk: ActionChunk
    summary: Mapping[str, object]

    def jev_summary(self) -> dict[str, object]:
        return {"id": self.id, **self.summary}


@dataclass(frozen=True)
class FilteredProposal:
    id: str
    code: str
    detail: str


@dataclass(frozen=True, eq=False)
class ValidatedCandidateSet:
    """Stable Batch 04 handoff: checked prefixes, one hold, and rejections."""

    available: tuple[ValidatedActionCandidate, ...]
    hold: ActionChunk
    filtered: tuple[FilteredProposal, ...]
    observation_reason: str | None


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rotation_rad(start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]) -> float:
    cosine = float(np.clip((np.trace(start.T @ end) - 1) / 2, -1, 1))
    return float(math.acos(cosine))


class AgentCandidateValidator:
    """Bind a real YAM declaration to local expansion and optional geometry checks.

    ``embodiment`` is read only: only ``info`` and the frozen ``_cfg`` are used.
    A supplied checker permits deterministic fake-model tests, but must carry
    the collision configuration derived from that exact YAM config.
    """

    def __init__(self, embodiment: object, motion: MotionPlanner | None = None, *,
                 collision_checker: CollisionChecker | None = None,
                 max_prefix_steps: int = 3, max_image_age_s: float | None = 1.0,
                 max_skew_s: float | None = None,
                 max_speed_frac: float = 0.1) -> None:
        cfg = getattr(embodiment, "_cfg", None)
        if not isinstance(cfg, YamConfig):
            raise ValueError("actual rig YamConfig is required")
        if isinstance(max_prefix_steps, bool) or not isinstance(max_prefix_steps, int) or not 1 <= max_prefix_steps <= 3:
            raise ValueError("max_prefix_steps must be an integer from 1 to 3")
        policy = SimpleNamespace(info=PolicyInfo(
            "jev-agent-validator", action_box(cfg.low, cfg.high),
            observation_space(cfg.cam_height, cfg.cam_width)))
        report = strict_yam_preflight(policy, embodiment)
        if not report.ok:
            raise ValueError("YAM pairing failed: " + ", ".join(i.code for i in report.errors))
        checker = None
        calibration = None
        if motion is None:
            if collision_checker is not None:
                raise ValueError("collision_checker requires a motion model")
        else:
            if not cfg.collision_guardrail or not cfg.collision_table:
                raise ValueError("rig collision and table guards must be enabled")
            if any(getattr(cfg, name) is None for name in (
                "collision_left_base_pos", "collision_right_base_pos",
                "collision_left_base_yaw", "collision_right_base_yaw",
                "collision_table_height")):
                raise ValueError("measured rig base poses and table height are required")
            calibration = motion.calibration
            if any((camera.height, camera.width) != (cfg.cam_height, cfg.cam_width)
                   for camera in calibration.cameras.values()):
                raise ValueError("calibration camera size differs from rig")
            if cfg.collision_table_height is not None and not math.isclose(
                cfg.collision_table_height, calibration.table_offset_m, abs_tol=1e-6):
                raise ValueError("rig and calibration table heights differ")
            if not np.allclose(calibration.table_normal_base, [0, 0, 1], atol=1e-6):
                raise ValueError("YAM collision plane requires a level calibration table")
            if collision_checker is None and not isinstance(motion.kinematics, YamKinematics):
                raise ValueError("production validation requires a flattened YAM MJCF")
            if isinstance(motion.kinematics, YamKinematics):
                model = motion.kinematics.model
                for side in ("left", "right"):
                    joint = motion.kinematics.arm_joint_ids[side][0]
                    body = int(model.jnt_bodyid[joint])
                    expected = np.asarray(getattr(cfg, f"collision_{side}_base_pos"))
                    if not np.allclose(model.body_pos[body], expected, atol=1e-6):
                        raise ValueError(f"{side} MJCF base differs from measured rig config")
                    yaw = float(getattr(cfg, f"collision_{side}_base_yaw"))
                    expected_quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
                    actual_quat = model.body_quat[body]
                    if not (np.allclose(actual_quat, expected_quat, atol=1e-6) or
                            np.allclose(actual_quat, -expected_quat, atol=1e-6)):
                        raise ValueError(f"{side} MJCF yaw differs from measured rig config")
            expected_collision = _config_from_yam(cfg)
            checker = collision_checker if collision_checker is not None else CollisionChecker(expected_collision)
            if checker.config != expected_collision:
                raise ValueError("collision checker does not match actual rig config")
        self._cfg = cfg
        self._motion = motion
        self._checker = checker
        self._toolset = build_toolset(
            embodiment.info.action_space, observation_space(cfg.cam_height, cfg.cam_width),
            cfg.control_hz, max_speed_frac=max_speed_frac, images="always")
        self._max_prefix_steps = max_prefix_steps
        self._max_image_age_s = max_image_age_s
        self._max_skew_s = max_skew_s
        self._provenance = {
            "validation_mode": "measured" if motion is not None else "yam",
            "observation_age_limit_s": max_image_age_s,
            "observation_skew_limit_s": (max_skew_s if max_skew_s is not None
                                         else max_image_age_s),
            "mjcf_sha256": motion.kinematics.mjcf_sha256 if motion is not None else None,
            "base_frame": BASE_FRAME,
            "calibration_id": calibration.calibration_id if calibration is not None else None,
            "calibration_version": CALIBRATION_VERSION if calibration is not None else None,
            "rig_config_sha256": _sha({
                "joint_low": cfg.joint_low, "joint_high": cfg.joint_high,
                "step_limits": cfg.step_limits, "control_hz": cfg.control_hz,
                "gripper_open": cfg.gripper_open, "gripper_closed": cfg.gripper_closed,
                "collision_guardrail": cfg.collision_guardrail,
            }),
            "collision_model": ("not_checked_in_policy" if checker is None else
                                "inspect_robots_yam.collision.CollisionChecker"
                                if type(checker) is CollisionChecker else "test-double"),
            "collision_model_version": version("inspect-robots-yam"),
            "collision_model_sha256": (
                hashlib.sha256(CollisionChecker._read_model_xml().encode("utf-8")).hexdigest()
                if type(checker) is CollisionChecker else None),
            "collision_config_sha256": _sha(asdict(checker.config)) if checker is not None else None,
            "table_config_sha256": _sha({
                "normal_base": calibration.table_normal_base.tolist(),
                "offset_m": calibration.table_offset_m,
                "margin_m": motion.limits.table_margin_m,
                "link_radius_m": motion.limits.link_radius_m,
            }) if calibration is not None and motion is not None else None,
            "motion_limits_sha256": _sha(asdict(motion.limits)) if motion is not None else None,
            "agent_max_speed_frac": max_speed_frac,
        }

    def validate(self, proposal: ProposalResult, observation: Observation,
                 *, now: float | None = None) -> ValidatedCandidateSet:
        try:
            decoded = decode_observation(
                observation, height=self._cfg.cam_height, width=self._cfg.cam_width,
                max_image_age_s=self._max_image_age_s, max_skew_s=self._max_skew_s,
                now=now)
        except InputError as exc:
            if exc.joint_pos is None:
                raise  # An invalid encoder state cannot produce a safe hold.
            self._check_rig_state(exc.joint_pos)
            ids = (candidate.id for candidate in proposal.candidates) if isinstance(proposal, ProposalBatch) else ()
            return ValidatedCandidateSet((), hold(exc.joint_pos, exc.code),
                                         tuple(FilteredProposal(i, exc.code, str(exc)) for i in ids), exc.code)
        q0 = decoded.joint_pos
        self._check_rig_state(q0)
        if not isinstance(proposal, ProposalBatch):
            if not isinstance(proposal, (ProposalFailure, ProposalTermination)):
                raise TypeError("expected a Batch 01 proposal result")
            reason = proposal.code if isinstance(proposal, ProposalFailure) else proposal.status
            return ValidatedCandidateSet((), hold(q0, reason), (), reason)

        # The tool receives the same validated measurement for every candidate.
        source = Observation(images=decoded.images, state={"joint_pos": q0.copy()},
                             instruction=decoded.instruction, image_times=decoded.image_times,
                             state_time=decoded.state_time)
        available: list[ValidatedActionCandidate] = []
        filtered: list[FilteredProposal] = []
        seen: set[tuple[tuple[float, ...], ...]] = set()
        seen_ids: set[str] = set()
        for candidate in proposal.candidates:
            if candidate.id == "hold" or candidate.id in seen_ids:
                filtered.append(FilteredProposal(candidate.id, "invalid_id", "duplicate or reserved ID"))
                continue
            seen_ids.add(candidate.id)
            item, rejection = self._candidate(candidate, source, q0)
            if rejection is not None:
                filtered.append(rejection)
                continue
            assert item is not None
            fingerprint = tuple(tuple(float(x) for x in np.round(action.data, 9))
                                for action in item.chunk.actions)
            if fingerprint in seen:
                filtered.append(FilteredProposal(candidate.id, "duplicate_prefix",
                                                 "same verified prefix as an earlier candidate"))
                continue
            seen.add(fingerprint)
            available.append(item)
        return ValidatedCandidateSet(tuple(available), hold(q0, "independent_hold"),
                                     tuple(filtered), None)

    def _check_rig_state(self, q: npt.NDArray[np.float64]) -> None:
        if np.any(q < self._cfg.low) or np.any(q > self._cfg.high):
            raise InputError("invalid_joint_pos", "joint_pos exceeds actual rig bounds")

    def _candidate(self, candidate: ProposalCandidate, source: Observation,
                   q0: npt.NDArray[np.float64]
                   ) -> tuple[ValidatedActionCandidate | None, FilteredProposal | None]:
        def reject(code: str, detail: str) -> tuple[None, FilteredProposal]:
            return None, FilteredProposal(candidate.id, code, detail)

        if not candidate.targets or not isinstance(candidate.note, str) or not candidate.note.strip() or not isinstance(candidate.intended_effect, str) or not candidate.intended_effect.strip():
            return reject("invalid_proposal", "targets, note and intended_effect are required")
        try:
            arguments = json.dumps({"targets": candidate.targets, "note": candidate.note})
        except (TypeError, ValueError, OverflowError):
            return reject("invalid_proposal", "targets cannot be encoded")
        result = self._toolset.execute(SimpleNamespace(name="move_joints", arguments=arguments), source)
        if result.error is not None:
            code = ("unknown_joint" if result.error.startswith("unknown dimension") else
                    "joint_limit" if "is outside" in result.error else
                    "non_finite_target" if "must be a finite number" in result.error else
                    "agent_expansion")
            return reject(code, result.error)
        chunk = result.chunk
        if not isinstance(chunk, ActionChunk) or not chunk.actions or len(chunk) > 100 or (
            chunk.control_hz is not None and
            (isinstance(chunk.control_hz, bool) or
             not isinstance(chunk.control_hz, (int, float)) or
             not math.isfinite(chunk.control_hz) or chunk.control_hz <= 0)):
            return reject("invalid_chunk", "Agent returned an invalid chunk")
        states: list[npt.NDArray[np.float64]] = []
        previous = q0
        for action in chunk.actions:
            if not isinstance(action, Action):
                return reject("invalid_chunk", "chunk contains a non-action")
            try:
                q = np.asarray(action.data, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                return reject("invalid_chunk", "action is not a numeric joint vector")
            if q.shape != (14,) or not np.isfinite(q).all():
                return reject("invalid_chunk", "action must be finite 14-D")
            if np.any(q < self._cfg.low) or np.any(q > self._cfg.high):
                return reject("rig_joint_limit", "action exceeds actual rig bounds")
            if np.any(np.abs(q - previous) > np.asarray(self._cfg.step_limits) + 1e-10):
                return reject("rig_step_limit", "action exceeds actual rig step limits")
            states.append(q.copy())
            previous = q
        if np.array_equal(states[-1], q0):
            return reject("zero_displacement", "final target equals measured state")
        if self._motion is not None:
            arm_changed = [not np.array_equal(states[-1][start:start + 7], q0[start:start + 7])
                           for start in (0, 7)]
            side = "both" if all(arm_changed) else "left" if arm_changed[0] else "right"
            previous = q0
            for index in range(0, len(states), self._motion.limits.max_steps):
                segment = states[index:index + self._motion.limits.max_steps]
                checked = self._motion.check_trajectory(previous, segment, side)
                if not checked.safe:
                    return reject(checked.reason or "motion_rejected", checked.detail or
                                  f"trajectory segment starting at step {index + 1}")
                previous = segment[-1]
            # The rig's own collision model is checked on the whole chunk, including
            # substeps and the suffix that will later be discarded.
            previous = q0
            assert self._checker is not None
            for index, q in enumerate(states, start=1):
                delta = np.abs(q - previous)
                count = max(1, int(np.ceil(np.max(delta[np.r_[0:6, 7:13]]) /
                                               self._motion.limits.interpolation_step_rad)),
                            int(np.ceil(np.max(delta[[6, 13]]) /
                                        self._motion.limits.interpolation_gripper_step)))
                for fraction in np.linspace(0, 1, count + 1):
                    sample = previous + fraction * (q - previous)
                    try:
                        report = self._checker.check(sample)
                    except Exception as exc:
                        return reject("collision_check_error", f"step {index}: {type(exc).__name__}: {exc}")
                    collided = getattr(report, "collided", None)
                    if not isinstance(collided, (bool, np.bool_)):
                        return reject("collision_check_error", f"step {index}: invalid collision result")
                    if collided:
                        return reject("rig_collision", f"step {index}: {report.geom1}:{report.geom2}")
                previous = q
        prefix = states[:self._max_prefix_steps]
        last = prefix[-1]
        eef: dict[str, object] | None = None
        if self._motion is not None:
            try:
                eef = {}
                for arm in ("left", "right"):
                    start = self._motion.kinematics.forward(arm, q0)
                    end = self._motion.kinematics.forward(arm, last)
                    if any(pose.position_base_m.shape != (3,) or
                           pose.rotation_base_eef.shape != (3, 3) or
                           not np.isfinite(pose.position_base_m).all() or
                           not np.isfinite(pose.rotation_base_eef).all()
                           for pose in (start, end)):
                        return reject("invalid_kinematics", "non-finite or malformed EEF pose")
                    eef[arm] = {
                        "position_delta_m": (end.position_base_m - start.position_base_m).tolist(),
                        "rotation_delta_rad": _rotation_rad(start.rotation_base_eef,
                                                              end.rotation_base_eef),
                    }
            except (ValueError, TypeError, OverflowError) as exc:
                return reject("invalid_kinematics", str(exc))
        joint_delta = {name: float(last[i] - q0[i]) for i, name in enumerate(DIM_LABELS)
                       if last[i] != q0[i]}
        summary: dict[str, object] = {
            "model_intent": {"note": candidate.note,
                             "intended_effect": candidate.intended_effect,
                             "expected_visual_change": (candidate.expected_visual_change
                                                        or candidate.intended_effect)},
            "computed": {
                "joint_delta": joint_delta,
                "gripper": {arm: {"from": float(q0[i]), "to": float(last[i]),
                                  "delta": float(last[i] - q0[i])}
                            for arm, i in (("left", 6), ("right", 13))},
                "eef_delta": eef,
            },
            "local_checks": {"agent_expansion": "passed", "rig_bounds": "passed",
                             "rig_step_limits": "passed",
                             "full_trajectory": "passed" if self._motion is not None else "not_checked",
                             "interpolated_collision": "passed" if self._motion is not None else "not_checked",
                             "unknown_object_collision": "unchecked",
                             "finger_contact_collision": "unchecked"},
            "prefix_steps": len(prefix), "source_steps_checked": len(states),
            "provenance": dict(self._provenance),
        }
        short = ActionChunk(actions=[Action(data=q.copy()) for q in prefix],
                            control_hz=chunk.control_hz,
                            meta={"candidate_id": candidate.id,
                                  "reobserve_after_chunk": True})
        return ValidatedActionCandidate(candidate.id, short, summary), None
