"""Observation driven Agent proposals, local YAM filtering, and bounded JEV choice.

Returning a chunk records a dispatch *request*, never evidence of execution.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from inspect_robots import ActionChunk, Observation, PolicyConfig, PolicyInfo, Scene
from inspect_robots.approver import Approver
from inspect_robots_agent._llm import resolve_provider
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent.policy import _validated_effort
from inspect_robots_agent.proposals import (
    ACTING_ARMS, VISUAL_EFFECTS, AgentProposer, OBJECT_STATES, PHASES,
    VISUAL_RELATIONS, ProposalBatch, ProposalFailure, ProposalFeedback,
    ProposalTermination,
)
from inspect_robots_yam.config import YamConfig, action_box

from .agent_candidates import AgentCandidateValidator, FilteredProposal
from .audit import MAX_RECORD_BYTES, MAX_RECORDS, bounded_record, decision_file, load_decision, save_decision
from .contract import InputError, camera_frame_ids, decode_observation, frames_advanced, hold
from .geometry import Calibration
from .jev_choice import DEFAULT_JEV_MODEL, DEFAULT_JEV_URL, ChoiceError, ChoiceOption, JevChoiceClient
from .kinematics import YamKinematics
from .motion import MotionPlanner
from .pairing import strict_yam_preflight
from .yam_contract import action_space, observation_space


_CAMERAS = frozenset({"top_cam", "left_cam", "right_cam"})
_POINTS = ("target_object", "left_gripper", "right_gripper", "placement_region")


class _VisualContractError(ValueError):
    pass


def _field(value: object, name: str, default: object = None) -> object:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _bounded_float(value: object, low: float, high: float, field: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))):
        raise _VisualContractError(f"{field} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _VisualContractError(f"{field} is not finite") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise _VisualContractError(f"{field} is outside [{low}, {high}]")
    return number


def _point(value: object, field: str) -> dict[str, object]:
    visible = _field(value, "visible")
    if type(visible) is not bool:
        raise _VisualContractError(f"{field}.visible is not boolean")
    confidence = _bounded_float(_field(value, "confidence"), 0.0, 1.0,
                                field + ".confidence")
    raw_u, raw_v = _field(value, "u"), _field(value, "v")
    if visible:
        u = _bounded_float(raw_u, 0.0, 1.0, field + ".u")
        v = _bounded_float(raw_v, 0.0, 1.0, field + ".v")
    else:
        if raw_u is not None or raw_v is not None:
            raise _VisualContractError(f"{field} is invisible but carries coordinates")
        u = v = None
    if _field(value, "source", "agent_unverified") != "agent_unverified":
        raise _VisualContractError(f"{field}.source is invalid")
    return {"visible": visible, "u": u, "v": v, "confidence": confidence,
            "source": "agent_unverified"}


def _validate_visuals(proposal: ProposalBatch
                      ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Normalize the Agent-only 2-D contract; empty mappings are legacy unknowns."""
    raw_estimates = proposal.visual_estimates
    if not isinstance(raw_estimates, Mapping):
        raise _VisualContractError("visual_estimates is not a mapping")
    estimates: dict[str, dict[str, object]] = {}
    if raw_estimates:
        if set(raw_estimates) != _CAMERAS:
            raise _VisualContractError("visual_estimates cameras do not match YAM")
        for camera in sorted(_CAMERAS):
            item = raw_estimates[camera]
            if _field(item, "source", "agent_unverified") != "agent_unverified":
                raise _VisualContractError(f"visual_estimates.{camera}.source is invalid")
            estimates[camera] = {
                name: _point(_field(item, name), f"visual_estimates.{camera}.{name}")
                for name in _POINTS
            }
            estimates[camera]["source"] = "agent_unverified"
    candidate_visuals: dict[str, dict[str, object]] = {}
    for candidate in proposal.candidates:
        arm = candidate.acting_arm
        if arm not in (*ACTING_ARMS, "unknown"):
            raise _VisualContractError(f"candidate {candidate.id} has invalid acting_arm")
        effect = candidate.visual_effect
        if effect not in VISUAL_EFFECTS:
            raise _VisualContractError(f"candidate {candidate.id} has invalid visual_effect")
        confidence = candidate.prediction_confidence
        if confidence is not None:
            confidence = _bounded_float(confidence, 0.0, 1.0,
                                        f"candidate {candidate.id}.prediction_confidence")
        raw_deltas = candidate.predicted_visual_deltas
        if not isinstance(raw_deltas, Mapping):
            raise _VisualContractError(f"candidate {candidate.id} deltas are not a mapping")
        deltas: dict[str, dict[str, object]] = {}
        if raw_deltas:
            if set(raw_deltas) != _CAMERAS:
                raise _VisualContractError(f"candidate {candidate.id} cameras do not match YAM")
            for camera in sorted(_CAMERAS):
                delta = raw_deltas[camera]
                raw_du, raw_dv = _field(delta, "du"), _field(delta, "dv")
                if (raw_du is None) != (raw_dv is None):
                    raise _VisualContractError(
                        f"candidate {candidate.id}.{camera} needs both du and dv"
                    )
                du = None if raw_du is None else _bounded_float(
                    raw_du, -1.0, 1.0, f"candidate {candidate.id}.{camera}.du")
                dv = None if raw_dv is None else _bounded_float(
                    raw_dv, -1.0, 1.0, f"candidate {candidate.id}.{camera}.dv")
                delta_confidence = _bounded_float(
                    _field(delta, "confidence"), 0.0, 1.0,
                    f"candidate {candidate.id}.{camera}.confidence")
                if _field(delta, "source", "agent_unverified") != "agent_unverified":
                    raise _VisualContractError(
                        f"candidate {candidate.id}.{camera}.source is invalid"
                    )
                deltas[camera] = {"du": du, "dv": dv,
                                  "confidence": delta_confidence,
                                  "source": "agent_unverified"}
        if not isinstance(candidate.verifiable_result, str):
            raise _VisualContractError(f"candidate {candidate.id} result is not text")
        candidate_visuals[candidate.id] = {
            "acting_arm": arm, "visual_effect": effect,
            "prediction_confidence": confidence,
            "verifiable_result": candidate.verifiable_result or "unknown",
            "predicted_visual_deltas": deltas,
            "source": "agent_unverified",
        }
    return estimates, candidate_visuals


def _xy(point: Mapping[str, object]) -> np.ndarray | None:
    if not point.get("visible"):
        return None
    return np.asarray([point["u"], point["v"]], dtype=np.float64)


def _diagonal_distance(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector) / math.sqrt(2.0))


def _visual_relations(estimates: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for camera, item in estimates.items():
        target = _xy(item["target_object"])
        placement = _xy(item["placement_region"])
        camera_result: dict[str, object] = {}
        for arm in ("left", "right"):
            gripper = _xy(item[f"{arm}_gripper"])
            camera_result[f"{arm}_to_target"] = (
                {"du": float(target[0] - gripper[0]),
                 "dv": float(target[1] - gripper[1]),
                 "distance": _diagonal_distance(target - gripper)}
                if target is not None and gripper is not None else {"status": "unknown"}
            )
        camera_result["target_to_placement"] = (
            {"du": float(placement[0] - target[0]),
             "dv": float(placement[1] - target[1]),
             "distance": _diagonal_distance(placement - target)}
            if target is not None and placement is not None else {"status": "unknown"}
        )
        result[camera] = camera_result
    return result


def _candidate_error_change(
    phase: str, visual: Mapping[str, object], estimates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    carrying = visual.get("visual_effect") in {"carrying", "placing"} or phase in {"carry", "place"}
    arms = ("left", "right") if visual.get("acting_arm") == "both" else (visual.get("acting_arm"),)
    result: dict[str, object] = {}
    deltas = visual.get("predicted_visual_deltas")
    if not isinstance(deltas, Mapping):
        return result
    for camera, delta in deltas.items():
        if not isinstance(delta, Mapping) or delta.get("du") is None:
            result[camera] = {"status": "unknown"}
            continue
        shift = np.asarray([delta["du"], delta["dv"]], dtype=np.float64)
        item = estimates.get(camera)
        if item is None:
            result[camera] = {"status": "unknown"}
            continue
        entries: dict[str, object] = {}
        movers = ("object",) if carrying else tuple(arm for arm in arms if arm in {"left", "right"})
        for mover in movers:
            if mover == "object":
                moving = _xy(item["target_object"])
                destination = _xy(item["placement_region"])
            else:
                moving = _xy(item[f"{mover}_gripper"])
                destination = _xy(item["target_object"])
            if moving is None or destination is None:
                entries[mover] = {"status": "unknown"}
                continue
            current = _diagonal_distance(destination - moving)
            predicted = _diagonal_distance(destination - (moving + shift))
            entries[mover] = {"current_error": current, "predicted_error": predicted,
                              "error_reduction": current - predicted}
        result[camera] = entries or {"status": "unknown"}
    return result


def _safe_audit(row: dict[str, object], api_key_env: str | None) -> dict[str, object]:
    """Bound an allowlisted record and redact every configured provider key."""
    result = bounded_record(row, None)
    for name in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", api_key_env):
        if name and os.environ.get(name):
            result = bounded_record(result, os.environ[name])
    return result


def _positive_limit(value: object, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


class JevAgentPolicy:
    """A fail-closed short-prefix policy for the real absolute-joint YAM rig."""

    def __init__(
        self, *, calibration_path: str | Path | None = None,
        mjcf_path: str | Path | None = None, cam_height: int = 224,
        cam_width: int = 224, model: str = "openai/gpt-6-astra",
        wire: str = "responses", selector: str = "jev",
        validation_mode: str = "yam",
        freshness_mode: str = "frame_sequence",
        effort: str | float | None = None, max_speed_frac: float = 0.1,
        max_dispatch_age_s: float | None = None, max_image_age_s: float = 1.0,
        max_skew_s: float | None = None, inference_budget_s: float = 30.0,
        agent_timeout_s: float = 20.0, jev_timeout_s: float = 5.0,
        max_llm_calls: int = 100, jev_model: str = DEFAULT_JEV_MODEL,
        jev_url: str = DEFAULT_JEV_URL, base_url: str | None = None,
        api_key_env: str | None = None,
        proposer: AgentProposer | None = None,
        validator: AgentCandidateValidator | None = None,
        choice: JevChoiceClient | None = None,
        visual_progress_threshold: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if wire != "responses" or model != "openai/gpt-6-astra":
            raise ValueError("jev-agent supports only openai/gpt-6-astra on wire=responses")
        if selector not in {"jev", "agent_preferred"}:
            raise ValueError("selector must be jev or agent_preferred")
        if validation_mode not in {"measured", "yam"}:
            raise ValueError("validation_mode must be measured or yam")
        if freshness_mode not in {"frame_sequence", "capture_age"}:
            raise ValueError("freshness_mode must be frame_sequence or capture_age")
        if cam_height < 1 or cam_width < 1:
            raise ValueError("camera dimensions must be positive")
        _positive_limit(max_speed_frac, "max_speed_frac")
        _positive_limit(visual_progress_threshold, "visual_progress_threshold")
        if effort is not None:
            effort = _validated_effort(effort)
        for name, value in (("max_image_age_s", max_image_age_s),
                            ("inference_budget_s", inference_budget_s),
                            ("agent_timeout_s", agent_timeout_s),
                            ("jev_timeout_s", jev_timeout_s)):
            _positive_limit(value, name)
        if max_dispatch_age_s is not None:
            _positive_limit(max_dispatch_age_s, "max_dispatch_age_s")
        if max_skew_s is not None:
            _positive_limit(max_skew_s, "max_skew_s")
        self.info = PolicyInfo("jev-agent", action_space(), observation_space(cam_height, cam_width))
        self.config = PolicyConfig(action_horizon=3, replan_interval=None)
        self._height, self._width = cam_height, cam_width
        self._model, self._selector = model, selector
        self._validation_mode = validation_mode
        self._freshness_mode = freshness_mode
        self._wire = wire
        self._effort, self._max_speed_frac = effort, max_speed_frac
        self._visual_progress_threshold = float(visual_progress_threshold)
        self._max_dispatch_age_s = max_dispatch_age_s
        self._max_image_age_s = max_image_age_s
        self._max_skew_s = max_skew_s
        self._inference_budget_s = inference_budget_s
        self._agent_timeout_s, self._jev_timeout_s = agent_timeout_s, jev_timeout_s
        self._max_llm_calls = max_llm_calls
        self._jev_model, self._jev_url = jev_model, jev_url
        self._base_url, self._api_key_env = base_url, api_key_env
        self._calibration_path, self._mjcf_path = calibration_path, mjcf_path
        self._proposer, self._validator, self._choice = proposer, validator, choice
        self._clock = clock
        self._paired = False
        self._bound_embodiment: object | None = None
        self._rig_low: np.ndarray | None = None
        self._rig_high: np.ndarray | None = None
        self._scene: Scene | None = None
        self._episode_id = uuid.uuid4().hex
        self._calls = 0
        self._last_times: tuple[float, ...] | None = None
        self._last_frame_sample: tuple[float, dict[str, tuple[int, int]]] | None = None
        self._last_assessment: dict[str, object] | None = None
        self._last_progress_evaluation_id: str | None = None
        self._no_progress_rounds = 0
        self._consecutive_jev_holds = 0
        self._operator_review_required = False
        self._consecutive_give_ups = 0
        self._consecutive_done = 0
        self._visual_history: dict[tuple[str, str], deque[dict[str, object]]] = defaultdict(
            lambda: deque(maxlen=12)
        )
        self._visual_models: dict[tuple[str, str], np.ndarray] = {}
        self._records: list[dict[str, object]] = []
        self._audit_run_dir: Path | None = None
        self._audit_write_failures = 0
        self._audit_omitted = 0
        self._pending: list[dict[str, object]] = []
        self._feedback_row: dict[str, object] | None = None
        self._last_motion_row: dict[str, object] | None = None
        self._trial_started_at: float | None = None
        self._feedback_sent_id: str | None = None
        self._feedback_sent_observed = False
        self._candidate_approver: Approver | None = None
        self._candidate_guardrail_store: dict[str, object] | None = None

    def set_candidate_guardrail(self, approver: Approver, store: dict[str, object]) -> None:
        """Bind the execution approver for read-only candidate preflight.

        Each candidate receives a private copy of the current approval state.
        The actual approval state changes only at execution time.
        """
        if not callable(getattr(approver, "review", None)):
            raise ValueError("candidate approver must have review")
        self._candidate_approver = approver
        self._candidate_guardrail_store = store

    def pairing_preflight(self, embodiment: object):
        """Bind to the actual frozen YAM spaces without opening its driver."""
        if self._bound_embodiment is not None and embodiment is not self._bound_embodiment:
            raise ValueError("jev-agent cannot reuse a validator with another embodiment")
        cfg = getattr(embodiment, "_cfg", None)
        if isinstance(cfg, YamConfig):
            self.info = PolicyInfo("jev-agent", embodiment.info.action_space,
                                   observation_space(cfg.cam_height, cfg.cam_width))
        report = strict_yam_preflight(self, embodiment)
        self._paired = False
        if not report.ok:
            return report
        assert isinstance(cfg, YamConfig)
        if self._validator is None:
            motion = None
            if self._validation_mode == "measured":
                if self._calibration_path is None or self._mjcf_path is None:
                    raise ValueError("calibration_path and mjcf_path are required for validation_mode=measured")
                calibration = Calibration.load(self._calibration_path)
                motion = MotionPlanner(YamKinematics(self._mjcf_path), calibration)
            elif self._calibration_path is not None or self._mjcf_path is not None:
                raise ValueError("calibration_path and mjcf_path require validation_mode=measured")
            self._validator = AgentCandidateValidator(
                embodiment, motion, max_prefix_steps=3,
                max_image_age_s=(self._max_image_age_s if self._freshness_mode == "capture_age"
                                 else None),
                max_skew_s=(self._max_skew_s if self._freshness_mode == "capture_age"
                            else None),
                max_speed_frac=self._max_speed_frac)
        if self._proposer is None and self._max_dispatch_age_s is not None:
            provider = resolve_provider(self._model, self._base_url, self._api_key_env,
                                        dict(os.environ),
                                        native_wires=frozenset({"chat", "responses"}))
            client = ResponsesClient(provider, timeout_s=self._agent_timeout_s,
                                     max_retries=1)
            self._proposer = AgentProposer(
                model=self._model, action_space=self.info.action_space,
                observation_space=self.info.observation_space,
                base_url=self._base_url, api_key_env=self._api_key_env,
                client=client, max_llm_calls=self._max_llm_calls,
                candidate_count=3, effort=self._effort,
                embodiment_docs=getattr(embodiment.info, "docs", None))
        elif self._proposer is not None:
            bind_docs = getattr(self._proposer, "bind_embodiment_docs", None)
            if callable(bind_docs):
                bind_docs(getattr(embodiment.info, "docs", None))
        if self._choice is None and self._selector == "jev":
            self._choice = JevChoiceClient(model=self._jev_model, url=self._jev_url,
                                           timeout_s=self._jev_timeout_s)
        self._height, self._width = cfg.cam_height, cfg.cam_width
        self._rig_low = np.asarray(cfg.low, dtype=np.float64).copy()
        self._rig_high = np.asarray(cfg.high, dtype=np.float64).copy()
        self._bound_embodiment = embodiment
        self._paired = True
        return report

    def _assert_rig_state(self, q: np.ndarray) -> None:
        if (self._rig_low is not None and self._rig_high is not None and
                (np.any(q < self._rig_low) or np.any(q > self._rig_high))):
            raise InputError("invalid_joint_pos", "joint_pos exceeds actual rig bounds")

    @property
    def audit_records(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(json.dumps(row)) for row in self._records)

    def transcript(self) -> list[dict[str, object]]:
        return list(self.audit_records)

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
        del scene_id, epoch, run_id
        self._audit_run_dir = Path(log_dir)
        self._trial_started_at = self._clock()

    def _persist(self, row: dict[str, object]) -> None:
        if self._audit_run_dir is None:
            return
        try:
            save_decision(self._audit_run_dir, row)
        except (OSError, ValueError):
            self._audit_write_failures += 1
            row["audit_sidecar_error"] = "write_failed"
            row["audit_incomplete"] = True

    def on_trial_end(self, record: object, log_dir: str, run_id: str) -> None:
        """Link stored frames and expose missing evidence in the trial index."""
        del run_id
        refs_by_time: dict[float, dict[str, str]] = {}
        for step in getattr(record, "steps", ()):
            stamp = getattr(step.observation, "state_time", None)
            refs = getattr(step, "image_refs", None)
            if stamp is not None and refs:
                refs_by_time[stamp] = {name: "frames/" + Path(ref.path).name
                                       for name, ref in refs.items()
                                       if name in ("top_cam", "left_cam", "right_cam")
                                       and (Path(log_dir) / "frames" / Path(ref.path).name).is_file()}
        missing_files = 0
        missing_frames = 0
        pending_unverified = 0
        incomplete_rows = 0
        for index in range(1, self._calls + 1):
            decision_id = f"{self._episode_id}-{index:06d}"
            try:
                disk_row = load_decision(log_dir, decision_id) if self._audit_run_dir else None
                row = next((item for item in self._records
                            if item["decision_id"] == decision_id), disk_row)
                if row is None:
                    raise FileNotFoundError(decision_id)
            except (OSError, ValueError, json.JSONDecodeError, StopIteration):
                missing_files += 1
                for existing in self._records:
                    if existing.get("decision_id") == decision_id:
                        existing["audit_incomplete"] = True
                        existing["audit_sidecar_error"] = "missing_or_invalid"
                        execution = existing.get("execution")
                        if isinstance(execution, dict) and execution.get("status") == "awaiting_observation":
                            execution["status"] = "unverified_no_next_observation"
                            pending_unverified += 1
                        break
                continue
            execution = row.get("execution")
            if isinstance(execution, dict) and execution.get("status") == "awaiting_observation":
                execution["status"] = "unverified_no_next_observation"
                pending_unverified += 1
                approval = row.get("approval")
                if isinstance(approval, dict) and approval.get("status") == "awaiting_observation":
                    approval["status"] = "not_observed"
            requested = row.get("selected_prefix")
            source = row.get("observation")
            start_step = source.get("env_step") if isinstance(source, dict) else None
            if row.get("selected_for_dispatch") and isinstance(requested, list):
                reviewed: list[list[float]] = []
                if type(start_step) is int:
                    by_step = {step.t: step for step in getattr(record, "steps", ())
                               if type(getattr(step, "t", None)) is int}
                    for t in range(start_step, start_step + len(requested)):
                        step = by_step.get(t)
                        if step is None:
                            break
                        action = getattr(step, "action", None)
                        if getattr(action, "meta", {}).get("decision_id") != decision_id:
                            break
                        try:
                            values = np.asarray(action.data, dtype=np.float64)
                        except (AttributeError, TypeError, ValueError):
                            break
                        if values.shape != (14,) or not np.isfinite(values).all():
                            break
                        reviewed.append(values.tolist())
                row["controller_trace"] = {
                    "status": "reviewed_action_recorded" if reviewed else "not_linked",
                    "reviewed_prefix": reviewed,
                    "recorded_steps": len(reviewed),
                    "requested_steps": len(requested),
                    "changed_steps": [offset for offset, values in enumerate(reviewed)
                                      if not np.allclose(values, requested[offset], atol=1e-9)],
                }
            row["trial_end"] = {
                "rollout_status": getattr(record, "status", None),
                "termination_reason": getattr(record, "termination_reason", None),
                "operator_judgement": getattr(record, "operator_judgement", None),
                "abort_type": ("SafetyAbort" if str(getattr(record, "error", "")).startswith("SafetyAbort")
                               else None),
            }
            for field in ("observation", "observed_state"):
                observed = row.get(field)
                if not isinstance(observed, dict):
                    continue
                stamp = observed.get("state_time")
                refs = refs_by_time.get(stamp, {})
                observed["frame_refs"] = refs
                if len(refs) != 3:
                    observed["missing_frames"] = sorted(
                        {"top_cam", "left_cam", "right_cam"} - set(refs))
                    row["audit_incomplete"] = True
                    missing_frames += 1
            clean = _safe_audit(row, self._api_key_env)
            incomplete_rows += bool(clean.get("audit_incomplete"))
            for position, existing in enumerate(self._records):
                if existing.get("decision_id") == decision_id:
                    self._records[position] = clean
                    break
            self._persist(clean)
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            metadata["jev_audit"] = {
                "schema_version": 1,
                "episode_id": self._episode_id,
                "relative_dir": f"jev-audit/{self._episode_id}",
                "decision_count": self._calls,
                "record_limit_bytes": MAX_RECORD_BYTES,
                "missing_files": missing_files,
                "missing_frames": missing_frames,
                "unverified_decisions": pending_unverified,
                "incomplete_rows": incomplete_rows,
                "write_failures": self._audit_write_failures,
                "incomplete": bool(missing_files or missing_frames or incomplete_rows
                                   or self._audit_write_failures),
                "trial_duration_s": (max(0.0, self._clock() - self._trial_started_at)
                                     if self._trial_started_at is not None else None),
            }
        if isinstance(getattr(record, "policy_transcript", None), list):
            record.policy_transcript = self.transcript()
        self._audit_run_dir = None
        self._trial_started_at = None

    def reset(self, scene: Scene) -> None:
        self._scene = scene
        self._episode_id = uuid.uuid4().hex
        self._calls = 0
        self._last_times = None
        self._last_frame_sample = None
        self._last_assessment = None
        self._last_progress_evaluation_id = None
        self._no_progress_rounds = 0
        self._consecutive_jev_holds = 0
        self._operator_review_required = False
        self._consecutive_give_ups = 0
        self._consecutive_done = 0
        self._visual_history.clear()
        self._visual_models.clear()
        self._records.clear()
        self._audit_write_failures = 0
        self._audit_omitted = 0
        self._pending.clear()
        self._feedback_row = None
        self._last_motion_row = None
        self._feedback_sent_id = None
        self._feedback_sent_observed = False
        if self._proposer is not None:
            self._proposer.reset()

    def _resolve_previous(self, decoded: object, observation: Observation) -> None:
        self._pending = [row for row in self._pending
                         if not self._resolve_one(row, decoded, observation)]

    def _resolve_one(self, row: dict[str, object], decoded: object,
                     observation: Observation) -> bool:
        previous = row.get("observation")
        if not isinstance(previous, dict):
            return False
        old_times = previous.get("image_times")
        prior_step = previous.get("env_step")
        current_step = observation.extra.get("env_step")
        has_new_step = (type(prior_step) is int and type(current_step) is int and
                        current_step > prior_step)
        has_new_capture = (
            isinstance(old_times, dict) and
            decoded.state_time > previous["state_time"] and
            all(decoded.image_times[name] > old_times[name] for name in old_times))
        has_new_frames = False
        if self._freshness_mode == "frame_sequence":
            try:
                prior_ids = previous.get("frame_ids")
                current_ids = camera_frame_ids(observation)
                has_new_frames = (isinstance(prior_ids, dict) and
                                  frames_advanced(current_ids, prior_ids) and
                                  decoded.state_time > previous["state_time"])
            except (InputError, KeyError, TypeError):
                pass
        if not (has_new_frames if self._freshness_mode == "frame_sequence" else
                has_new_step if self._validation_mode == "yam" and
                type(prior_step) is int and type(current_step) is int
                else has_new_capture):
            return False
        old_q = np.asarray(previous["joint_pos"], dtype=np.float64)
        measured = {"joint_pos": decoded.joint_pos.tolist(),
                    "joint_delta": (decoded.joint_pos - old_q).tolist(),
                    "state_time": decoded.state_time,
                    "image_times": dict(decoded.image_times), "frame_refs": {},
                    "frame_ids": (dict(current_ids) if has_new_frames else None)}
        row["observed_state"] = measured
        approvals: list[dict[str, object]] = []
        mismatch = False
        raw_approvals = observation.extra.get("approvals", [])
        start_step = previous.get("env_step")
        end_step = observation.extra.get("env_step")
        if isinstance(raw_approvals, list):
            for approval in raw_approvals:
                if not isinstance(approval, dict):
                    continue
                if approval.get("decision_id", row["decision_id"]) != row["decision_id"]:
                    mismatch = True
                    continue
                step = approval.get("t")
                if (type(start_step) is int and type(end_step) is int and
                        (type(step) is not int or not start_step <= step < end_step)):
                    mismatch = True
                    continue
                detail = approval.get("detail")
                # The rollout's approval vocabulary is finite. Unknown text is
                # represented as a generic modification, never copied from extra.
                words = detail.split(", ") if isinstance(detail, str) else []
                flags = [flag for flag in ("clamped", "delta_clamped") if flag in words]
                approvals.append({"t": step if type(step) is int else None,
                                  "detail": ", ".join(flags) if flags else "modified"})
        elif raw_approvals is not None:
            mismatch = True
        row["approval"] = {"status": "id_mismatch" if mismatch else
                           "modified" if approvals else "none_reported",
                           "events": approvals}
        if mismatch:
            row["audit_incomplete"] = True
        execution = row["execution"]
        assert isinstance(execution, dict)
        execution["status"] = ("observed_approval_unverified" if mismatch else
                               "observed_after_dispatch" if row["selected_id"] not in (None, "hold")
                               else "observed_after_hold")
        selected_prefix = row.get("selected_prefix")
        if row["selected_id"] not in (None, "hold") and isinstance(selected_prefix, list):
            target = np.asarray(selected_prefix[-1], dtype=np.float64)
            execution["max_joint_error_to_requested"] = float(
                np.max(np.abs(decoded.joint_pos - target)))
        row["execution"] = execution
        clean = _safe_audit(row, self._api_key_env)
        row.clear()
        row.update(clean)
        self._persist(row)
        return True

    @staticmethod
    def _motion_visual(row: Mapping[str, object]) -> Mapping[str, object] | None:
        selected = row.get("selected_id")
        proposed = row.get("proposed")
        if not isinstance(selected, str) or not isinstance(proposed, list):
            return None
        for item in proposed:
            if isinstance(item, Mapping) and item.get("id") == selected:
                visual = item.get("visual_prediction")
                return visual if isinstance(visual, Mapping) else None
        return None

    @staticmethod
    def _observed_visual_deltas(
        before: Mapping[str, Mapping[str, object]],
        after: Mapping[str, Mapping[str, object]],
    ) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for camera in sorted(set(before) & set(after)):
            entities: dict[str, object] = {}
            for name in _POINTS:
                first = _xy(before[camera][name])
                second = _xy(after[camera][name])
                if first is None or second is None:
                    entities[name] = {"status": "unknown"}
                else:
                    delta = second - first
                    entities[name] = {"du": float(delta[0]), "dv": float(delta[1]),
                                      "distance": _diagonal_distance(delta)}
            result[camera] = entities
        return result

    def _learn_visual_mapping(self, row: Mapping[str, object],
                              actual: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        observed = row.get("observed_state")
        if not isinstance(observed, Mapping):
            return {}
        try:
            joint_delta = np.asarray(observed.get("joint_delta"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return {}
        if joint_delta.shape != (14,) or not np.isfinite(joint_delta).all():
            return {}
        for camera, entities in actual.items():
            for arm, indices in (("left", slice(0, 7)), ("right", slice(7, 14))):
                movement = entities.get(f"{arm}_gripper")
                if not isinstance(movement, Mapping) or movement.get("du") is None:
                    continue
                joints = joint_delta[indices]
                if float(np.linalg.norm(joints)) <= 1e-8:
                    continue
                visual = np.asarray([movement["du"], movement["dv"]], dtype=np.float64)
                self._visual_history[(camera, arm)].append({
                    "joint_delta": joints.tolist(), "visual_delta": visual.tolist(),
                    "decision_id": row.get("decision_id"),
                })
        summary: dict[str, object] = {}
        for key, samples in self._visual_history.items():
            camera, arm = key
            status = "unknown"
            if len(samples) >= 3:
                x = np.asarray([sample["joint_delta"] for sample in samples], dtype=np.float64)
                y = np.asarray([sample["visual_delta"] for sample in samples], dtype=np.float64)
                regularizer = 1e-4 * np.eye(7, dtype=np.float64)
                try:
                    model = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
                except np.linalg.LinAlgError:
                    model = np.linalg.pinv(x.T @ x + regularizer) @ x.T @ y
                if model.shape == (7, 2) and np.isfinite(model).all():
                    self._visual_models[key] = model
                    status = "estimated"
            summary[f"{camera}:{arm}"] = {"status": status, "sample_count": len(samples),
                                           "source": "observed_real_motion"}
        return summary

    @staticmethod
    def _direction_consistency(predicted: np.ndarray, actual: np.ndarray) -> str:
        if float(np.linalg.norm(predicted)) <= 1e-8 or float(np.linalg.norm(actual)) <= 1e-8:
            return "unknown"
        cosine = float(np.dot(predicted, actual) /
                       (np.linalg.norm(predicted) * np.linalg.norm(actual)))
        return "consistent" if cosine > 0.25 else "conflict" if cosine < -0.25 else "unknown"

    def _prediction_consistency(
        self, row: Mapping[str, object], actual: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        visual = self._motion_visual(row)
        if visual is None:
            return {"status": "unknown"}
        phase = _field(row.get("agent_assessment"), "phase", "unknown")
        carrying = visual.get("visual_effect") in {"carrying", "placing"} or phase in {
            "carry", "place"
        }
        arms = (("left", "right") if visual.get("acting_arm") == "both"
                else (visual.get("acting_arm"),))
        predictions = visual.get("predicted_visual_deltas")
        result: dict[str, object] = {}
        statuses: list[str] = []
        if not isinstance(predictions, Mapping):
            return {"status": "unknown"}
        for camera, prediction in predictions.items():
            if not isinstance(prediction, Mapping) or prediction.get("du") is None:
                result[camera] = {"status": "unknown"}
                continue
            predicted = np.asarray([prediction["du"], prediction["dv"]], dtype=np.float64)
            camera_actual = actual.get(camera, {})
            movers = ("target_object",) if carrying else tuple(
                f"{arm}_gripper" for arm in arms if arm in {"left", "right"}
            )
            camera_result: dict[str, str] = {}
            for mover in movers:
                movement = camera_actual.get(mover)
                if not isinstance(movement, Mapping) or movement.get("du") is None:
                    status = "unknown"
                else:
                    status = self._direction_consistency(
                        predicted, np.asarray([movement["du"], movement["dv"]], dtype=np.float64)
                    )
                camera_result[mover] = status
                statuses.append(status)
            result[camera] = camera_result or {"status": "unknown"}
        result["status"] = ("conflict" if "conflict" in statuses else
                            "consistent" if "consistent" in statuses else "unknown")
        return result

    def _co_motion(self, row: Mapping[str, object],
                   actual: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        visual = self._motion_visual(row)
        if visual is None:
            return {"status": "unknown"}
        arms = (("left", "right") if visual.get("acting_arm") == "both"
                else (visual.get("acting_arm"),))
        result: dict[str, object] = {}
        statuses: list[str] = []
        for camera, entities in actual.items():
            target = entities.get("target_object")
            camera_result: dict[str, str] = {}
            for arm in arms:
                gripper = entities.get(f"{arm}_gripper")
                if (arm not in {"left", "right"} or not isinstance(target, Mapping) or
                        not isinstance(gripper, Mapping) or target.get("du") is None or
                        gripper.get("du") is None):
                    status = "unknown"
                else:
                    difference = np.asarray(
                        [target["du"] - gripper["du"], target["dv"] - gripper["dv"]],
                        dtype=np.float64,
                    )
                    status = ("consistent" if _diagonal_distance(difference) <=
                              self._visual_progress_threshold else "conflict")
                camera_result[str(arm)] = status
                statuses.append(status)
            result[camera] = camera_result
        result["status"] = ("conflict" if "conflict" in statuses else
                            "consistent" if "consistent" in statuses else "unknown")
        return result

    def _historical_prediction(self, item: object, visual: Mapping[str, object],
                               q: np.ndarray) -> dict[str, object]:
        if visual.get("visual_effect") in {"carrying", "placing"}:
            return {"status": "unknown"}
        arms = (("left", "right") if visual.get("acting_arm") == "both"
                else (visual.get("acting_arm"),))
        try:
            last = np.asarray(item.chunk.actions[-1].data, dtype=np.float64)
        except (AttributeError, IndexError, TypeError, ValueError):
            return {"status": "unknown"}
        predictions = visual.get("predicted_visual_deltas")
        if not isinstance(predictions, Mapping):
            return {"status": "unknown"}
        result: dict[str, object] = {}
        statuses: list[str] = []
        for camera, prediction in predictions.items():
            if not isinstance(prediction, Mapping) or prediction.get("du") is None:
                continue
            agent_delta = np.asarray([prediction["du"], prediction["dv"]], dtype=np.float64)
            camera_result: dict[str, str] = {}
            for arm in arms:
                if arm not in {"left", "right"}:
                    continue
                model = self._visual_models.get((camera, arm))
                if model is None:
                    status = "unknown"
                else:
                    indices = slice(0, 7) if arm == "left" else slice(7, 14)
                    learned_delta = (last[indices] - q[indices]) @ model
                    status = self._direction_consistency(agent_delta, learned_delta)
                camera_result[arm] = status
                statuses.append(status)
            result[camera] = camera_result
        result["status"] = ("conflict" if "conflict" in statuses else
                            "consistent" if "consistent" in statuses else "unknown")
        return result

    def _visual_error_reduction(
        self, row: Mapping[str, object], before: Mapping[str, Mapping[str, object]],
        after: Mapping[str, Mapping[str, object]],
    ) -> float | None:
        visual = self._motion_visual(row)
        if visual is None:
            return None
        phase = _field(row.get("agent_assessment"), "phase", "unknown")
        carrying = visual.get("visual_effect") in {"carrying", "placing"} or phase in {
            "carry", "place"
        }
        arms = (("left", "right") if visual.get("acting_arm") == "both"
                else (visual.get("acting_arm"),))
        reductions: list[float] = []
        for camera in set(before) & set(after):
            pairs = (("target_object", "placement_region"),) if carrying else tuple(
                (f"{arm}_gripper", "target_object") for arm in arms
                if arm in {"left", "right"}
            )
            for moving_name, destination_name in pairs:
                old_moving, old_destination = (_xy(before[camera][moving_name]),
                                               _xy(before[camera][destination_name]))
                new_moving, new_destination = (_xy(after[camera][moving_name]),
                                               _xy(after[camera][destination_name]))
                if any(value is None for value in (
                        old_moving, old_destination, new_moving, new_destination)):
                    continue
                old_error = _diagonal_distance(old_destination - old_moving)
                new_error = _diagonal_distance(new_destination - new_moving)
                reductions.append(old_error - new_error)
        return float(np.mean(reductions)) if reductions else None

    def _assess_progress(
        self, proposal: ProposalBatch, estimates: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        current = {"phase": proposal.phase, "visual_relation": proposal.visual_relation,
                   "object_state": proposal.object_state, "visual_estimates": estimates}
        evidence: dict[str, object] = {
            "evaluated": False, "joint_progress": False,
            "reported_visual_progress": False, "counts_as_progress": False,
        }
        previous = self._last_assessment
        row = self._last_motion_row
        if (previous is not None and row is not None and row.get("selected_for_dispatch") and
                row.get("decision_id") != self._last_progress_evaluation_id):
            phases = ("approach", "align", "grasp", "lift", "carry", "place")
            relations = {"unknown": -1, "far": 0, "near": 1, "aligned": 2}
            phase_advanced = (previous["phase"] in phases and current["phase"] in phases and
                              phases.index(str(current["phase"])) >
                              phases.index(str(previous["phase"])))
            relation_improved = (current["phase"] == previous["phase"] and
                                 relations[str(current["visual_relation"])] >
                                 relations[str(previous["visual_relation"])] and
                                 current["visual_relation"] != "unknown")
            object_advanced = (previous["object_state"], current["object_state"]) in {
                ("on_table", "held"), ("held", "placed")}
            execution = row.get("execution")
            observed = row.get("observed_state")
            joint_progress = False
            if (isinstance(execution, Mapping) and
                    execution.get("status") == "observed_after_dispatch" and
                    isinstance(observed, Mapping)):
                try:
                    delta = np.asarray(observed.get("joint_delta"), dtype=np.float64)
                    joint_progress = (delta.shape == (14,) and np.isfinite(delta).all() and
                                      float(np.max(np.abs(delta))) > 1e-5)
                except (TypeError, ValueError, OverflowError):
                    pass
            before = row.get("visual_estimates")
            if estimates and isinstance(before, Mapping) and before:
                actual = self._observed_visual_deltas(before, estimates)
                row["actual_visual_change"] = actual
                row["prediction_consistency"] = self._prediction_consistency(row, actual)
                row["object_gripper_co_motion"] = self._co_motion(row, actual)
                row["local_visual_mapping"] = self._learn_visual_mapping(row, actual)
                row["visual_data_sources"] = {
                    "estimates": "agent_unverified",
                    "predictions": "agent_unverified",
                    "actual_change": "difference_of_consecutive_agent_unverified_estimates",
                    "local_mapping": "observed_real_motion",
                }
                reduction = self._visual_error_reduction(row, before, estimates)
                phase = _field(row.get("agent_assessment"), "phase", "unknown")
                if phase in {"grasp", "lift"}:
                    visual_progress = row["object_gripper_co_motion"].get("status") == "consistent"
                else:
                    visual_progress = (reduction is not None and
                                       reduction >= self._visual_progress_threshold)
                progress = joint_progress and visual_progress
                evidence = {
                    "evaluated": True, "joint_progress": joint_progress,
                    "reported_visual_progress": visual_progress,
                    "counts_as_progress": progress,
                    "visual_error_reduction": reduction,
                    "threshold": self._visual_progress_threshold,
                    "source": "agent_unverified_2d_comparison",
                }
                self._persist(_safe_audit(dict(row), self._api_key_env))
            else:
                visual_progress = phase_advanced or relation_improved or object_advanced
                progress = joint_progress and visual_progress
                evidence = {"evaluated": True, "joint_progress": joint_progress,
                            "reported_visual_progress": visual_progress,
                            "counts_as_progress": progress}
            self._no_progress_rounds = 0 if progress else self._no_progress_rounds + 1
            self._last_progress_evaluation_id = str(row["decision_id"])
        self._last_assessment = current
        if self._no_progress_rounds >= 3:
            self._operator_review_required = True
        return evidence

    def _jev_context(self, proposal: ProposalBatch,
                     estimates: Mapping[str, Mapping[str, object]],
                     relations: Mapping[str, object]) -> str:
        def compact(row: dict[str, object] | None) -> dict[str, object] | None:
            if row is None:
                return None
            execution = row.get("execution")
            return {"selection": row.get("selected_id") or "hold",
                    "reason": row.get("reason"),
                    "dispatch_status": row.get("dispatch_status"),
                    "execution_status": (execution.get("status")
                                         if isinstance(execution, dict) else None),
                    "max_joint_error": (execution.get("max_joint_error_to_requested")
                                        if isinstance(execution, dict) else None),
                    "approval": row.get("approval"),
                    "actual_visual_change": row.get("actual_visual_change"),
                    "prediction_consistency": row.get("prediction_consistency")}
        previous = self._feedback_row
        last_motion = (self._last_motion_row if self._last_motion_row is not previous else None)
        context = {"source": "agent_unverified",
                   "scene": proposal.scene_summary[:512],
                   "phase": proposal.phase,
                   "visual_relation": proposal.visual_relation,
                   "object_state": proposal.object_state,
                   "normalized_visual_points": estimates or {"status": "unknown"},
                   "derived_visual_relations": relations or {"status": "unknown"},
                   "previous_result": compact(previous),
                   "last_motion_result": compact(last_motion),
                   "guidance": ("During approach, the object normally remains still. Judge progress "
                                "from the gripper-to-object visual relation and actual execution "
                                "feedback. Select hold only when movement is unjustified; "
                                "local checks do not prove object or finger clearance.")}
        return json.dumps(context, separators=(",", ":"), ensure_ascii=False)

    def _feedback(self) -> ProposalFeedback | None:
        row = self._feedback_row
        if row is None:
            return None
        outcome = {
            "decision_id": row["decision_id"],
            "selection": row.get("selected_id") or "hold",
            "reason": row.get("reason"),
            "dispatch_status": row.get("dispatch_status"),
            "approval": row.get("approval"),
            "approval_effect": (
                "YAM reported a changed action; the requested prefix may differ from the controller action."
                if isinstance(row.get("approval"), dict) and row["approval"].get("status") == "modified"
                else "Approval could not be matched to this decision."
                if isinstance(row.get("approval"), dict) and row["approval"].get("status") == "id_mismatch"
                else None),
            "observed_state": row.get("observed_state"),
            "execution": row.get("execution"),
            "earlier_dispatch": (
                {"decision_id": self._last_motion_row["decision_id"],
                 "selection": self._last_motion_row["selected_id"],
                 "approval": self._last_motion_row.get("approval"),
                 "observed_state": self._last_motion_row.get("observed_state"),
                 "execution": self._last_motion_row.get("execution")}
                if self._last_motion_row is not None and self._last_motion_row is not row
                else None),
            "caution": "A dispatch request and joint readback do not prove task success; operator judges completion.",
        }
        return ProposalFeedback(
            selected_id=row.get("selected_id") if row.get("selected_id") != "hold" else None,
            outcome=json.dumps(outcome, separators=(",", ":"), ensure_ascii=False),
        )

    def _propose(self, observation: Observation) -> object:
        assert self._proposer is not None and self._scene is not None
        self._feedback_sent_id = None
        self._feedback_sent_observed = False
        feedback = self._feedback()
        if feedback is not None and "feedback" in inspect.signature(self._proposer.propose).parameters:
            self._feedback_sent_id = str(self._feedback_row["decision_id"])
            self._feedback_sent_observed = isinstance(self._feedback_row.get("observed_state"), dict)
            return self._proposer.propose(self._scene.instruction, observation, feedback=feedback)
        return self._proposer.propose(self._scene.instruction, observation)

    def _guard_candidates(self, checked: object) -> object:
        if self._candidate_approver is None or self._candidate_guardrail_store is None:
            return checked
        available = []
        filtered = list(checked.filtered)
        for item in checked.available:
            trial_store = deepcopy(self._candidate_guardrail_store)
            blocked = False
            for action in item.chunk.actions:
                original = np.asarray(action.data, dtype=np.float64).copy()
                try:
                    reviewed = self._candidate_approver.review(action, trial_store)
                    blocked = (reviewed is not action or
                               not np.array_equal(np.asarray(reviewed.data), original))
                except Exception:
                    blocked = True
                if blocked:
                    filtered.append(FilteredProposal(item.id, "yam_guardrail",
                                                     "execution approver modified or vetoed prefix"))
                    break
            if not blocked:
                available.append(item)
        return replace(checked, available=tuple(available), filtered=tuple(filtered))

    def act(self, observation: Observation) -> ActionChunk:
        started = self._clock()
        self._calls += 1
        decision_id = f"{self._episode_id}-{self._calls:06d}"
        row: dict[str, object] = {
            "audit_schema_version": 1,
            "decision_id": decision_id, "branch": self.info.name, "call": self._calls,
            "task": {"scene_id": self._scene.id if self._scene else None,
                      "instruction": self._scene.instruction if self._scene else None},
            "rig_fingerprint": dict(getattr(self._validator, "_provenance", {})),
            "selector": self._selector,
            "validation_mode": self._validation_mode,
            "observation_freshness": ("frame_sequence" if self._freshness_mode == "frame_sequence"
                                      else "capture_time_checked" if self._validation_mode == "measured"
                                      else "not_checked"),
            "dispatch_age_basis": ("decision_start" if self._freshness_mode == "frame_sequence"
                                   else "capture_time" if self._validation_mode == "measured"
                                   else "policy_start"),
            "decision_started_at": started,
            "candidate_guardrail": ("active" if self._candidate_approver is not None
                                    else "not_configured"),
            "validation_checks": {
                "finite_values": "checked", "yam_joint_ranges": "checked",
                "action_expansion": "checked", "speed_and_step_limits": "checked",
                "rollout_approver": ("checked" if self._candidate_approver is not None
                                      else "not_configured"),
                "full_trajectory": ("checked" if self._validation_mode == "measured"
                                    else "not_checked"),
                "interpolated_collision": ("checked" if self._validation_mode == "measured"
                                           else "not_checked"),
            },
            "proposed_ids": [], "candidates": [], "filtered": [],
            "proposed": [], "selected_for_dispatch": None,
            "selected_id": None, "selected_prefix": None, "reason": None,
            "mode": None, "agent_latency_s": None, "filter_latency_s": None,
            "jev_latency_s": None, "inference_budget_s": self._inference_budget_s,
            "max_dispatch_age_s": self._max_dispatch_age_s,
            "agent_model": self._model, "jev_model": self._jev_model,
            "agent_feedback_decision_id": None, "agent_feedback_observed": False,
            "wire": self._wire, "dispatch_status": "not_requested",
            "observation": None, "observed_state": None,
            "approval": {"status": "awaiting_observation", "events": []},
            "execution": {"status": "not_requested"},
            "controller_trace": {"status": "not_linked"},
        }

        def finish(chunk: ActionChunk, reason: str | None = None) -> ActionChunk:
            elapsed = max(0.0, self._clock() - started)
            row["reason"] = reason
            row["latency_s"] = elapsed
            if row["dispatch_status"] == "selected_for_dispatch":
                row["execution"] = {"status": "awaiting_observation"}
            elif row["dispatch_status"] == "hold_requested":
                row["execution"] = {"status": "hold_requested"}
            row["audit_file"] = decision_file(decision_id)
            clean = _safe_audit(row, self._api_key_env)
            self._records.append(clean)
            if len(self._records) > MAX_RECORDS:
                self._records.pop(0)
                self._audit_omitted += 1
            clean["audit_omitted_prior"] = self._audit_omitted
            self._persist(clean)
            if row["observation"] is not None:
                self._pending.append(clean)
            self._feedback_row = clean
            if row["dispatch_status"] == "selected_for_dispatch":
                self._last_motion_row = clean
            meta = {**chunk.meta, "decision_id": decision_id,
                    "proposed_ids": list(row["proposed_ids"]),
                    "candidate_ids": [item["id"] for item in row["candidates"]],
                    "filtered": list(row["filtered"]),
                    "selected_id": row["selected_id"], "decision_mode": row["mode"]}
            if row["dispatch_status"] == "selected_for_dispatch":
                chunk = replace(chunk, actions=[replace(action, meta={**action.meta,
                                                                     "decision_id": decision_id})
                                                for action in chunk.actions])
            return replace(chunk, meta=meta, inference_latency_s=elapsed)

        def record_error(code: str) -> None:
            row["reason"] = code
            row["latency_s"] = max(0.0, self._clock() - started)
            row["audit_file"] = decision_file(decision_id)
            clean = _safe_audit(row, self._api_key_env)
            self._records.append(clean)
            if len(self._records) > MAX_RECORDS:
                self._records.pop(0)
                self._audit_omitted += 1
            clean["audit_omitted_prior"] = self._audit_omitted
            self._persist(clean)
            self._feedback_row = clean

        def check_state(q: np.ndarray) -> None:
            try:
                self._assert_rig_state(q)
            except InputError as exc:
                record_error(exc.code)
                raise

        # Decode before any service request: an invalid encoder read cannot hold.
        try:
            decoded = decode_observation(
                observation, height=self._height, width=self._width,
                max_image_age_s=(self._max_image_age_s if self._freshness_mode == "capture_age" and
                                 self._validation_mode == "measured"
                                 else None),
                max_skew_s=(self._max_skew_s if self._freshness_mode == "capture_age" and
                            self._validation_mode == "measured"
                            else None), now=started)
        except InputError as exc:
            if exc.joint_pos is None:
                record_error(exc.code)
                raise
            check_state(exc.joint_pos)
            return finish(hold(exc.joint_pos, exc.code), exc.code)
        q = decoded.joint_pos
        check_state(q)
        self._resolve_previous(decoded, observation)
        frame_ids: dict[str, tuple[int, int]] | None = None
        if self._freshness_mode == "frame_sequence":
            try:
                frame_ids = camera_frame_ids(observation)
            except InputError as exc:
                return finish(hold(q, exc.code), exc.code)
        row["observation"] = {"joint_pos": q.tolist(), "state_time": decoded.state_time,
                              "image_times": dict(decoded.image_times), "frame_refs": {},
                              "frame_ids": dict(frame_ids) if frame_ids is not None else None,
                              "env_step": observation.extra.get("env_step")
                              if type(observation.extra.get("env_step")) is int else None}
        safe_observation = Observation(
            images=decoded.images, state={"joint_pos": q.copy()},
            instruction=decoded.instruction, image_times=decoded.image_times,
            state_time=decoded.state_time)
        stamps = (decoded.state_time, *decoded.image_times.values())
        if not self._paired or self._validator is None:
            return finish(hold(q, "pairing_not_ready"), "pairing_not_ready")
        if self._max_dispatch_age_s is None:
            return finish(hold(q, "dispatch_age_unconfigured"), "dispatch_age_unconfigured")
        if self._proposer is None:
            return finish(hold(q, "agent_not_ready"), "agent_not_ready")
        if self._freshness_mode == "frame_sequence":
            assert frame_ids is not None
            if self._last_frame_sample is None:
                self._last_frame_sample = (decoded.state_time, frame_ids)
                return finish(hold(q, "awaiting_fresh_frames"), "awaiting_fresh_frames")
            old_state_time, old_frame_ids = self._last_frame_sample
            if decoded.state_time <= old_state_time or not frames_advanced(frame_ids, old_frame_ids):
                return finish(hold(q, "observation_not_new"), "observation_not_new")
            self._last_frame_sample = (decoded.state_time, frame_ids)
        if (self._freshness_mode == "capture_age" and self._validation_mode == "measured" and
                self._last_times is not None and
                any(a <= b for a, b in zip(stamps, self._last_times))):
            return finish(hold(q, "observation_not_new"), "observation_not_new")
        dispatch_anchor = (min(stamps) if self._freshness_mode == "capture_age" and
                           self._validation_mode == "measured" else started)
        if started - dispatch_anchor > self._max_dispatch_age_s:
            return finish(hold(q, "dispatch_stale"), "dispatch_stale")
        if self._freshness_mode == "capture_age" and self._validation_mode == "measured":
            self._last_times = stamps
        if self._scene is None:
            return finish(hold(q, "scene_not_ready"), "scene_not_ready")
        if self._operator_review_required:
            stop = hold(q, "stalled_review")
            return finish(replace(stop, actions=[replace(stop.actions[0], meta={
                "request_stop": True, "stop_reason": "stalled_review"})]), "stalled_review")

        agent_started = self._clock()
        try:
            proposal = self._propose(safe_observation)
        except Exception:
            self._consecutive_give_ups = 0
            self._consecutive_done = 0
            row["agent_latency_s"] = max(0.0, self._clock() - agent_started)
            return finish(hold(q, "agent_error"), "agent_error")
        finally:
            row["agent_feedback_decision_id"] = self._feedback_sent_id
            row["agent_feedback_observed"] = self._feedback_sent_observed
        row["agent_latency_s"] = max(0.0, self._clock() - agent_started)
        if isinstance(proposal, (ProposalBatch, ProposalFailure, ProposalTermination)):
            row["agent_model"] = proposal.model
            row["agent_duration_s"] = proposal.duration_s
            row["agent_usage"] = proposal.usage
        if isinstance(proposal, ProposalTermination):
            row["termination"] = {"status": proposal.status, "summary": proposal.summary,
                                  "hindsight": proposal.hindsight}
            if proposal.status == "give_up":
                self._consecutive_give_ups += 1
                self._consecutive_done = 0
                count, required = self._consecutive_give_ups, 3
            else:
                self._consecutive_done += 1
                self._consecutive_give_ups = 0
                count, required = self._consecutive_done, 2
            row["termination"].update({"confirmation_count": count,
                                       "required_count": required,
                                       "operator_review_required": (
                                           proposal.status == "give_up" and count >= required
                                       )})
            if count < required:
                reason = proposal.status + "_confirmation_pending"
                return finish(hold(q, reason), reason)
            stop = hold(q, proposal.status)
            stop_meta = {"request_stop": True, "stop_reason": proposal.status,
                         "stop_detail": proposal.summary}
            if proposal.status == "give_up":
                stop_meta["operator_review_required"] = True
            action = replace(stop.actions[0], meta=stop_meta)
            return finish(replace(stop, actions=[action]), proposal.status)
        if isinstance(proposal, ProposalFailure):
            self._consecutive_give_ups = 0
            self._consecutive_done = 0
            row["agent_failure"] = {"code": proposal.code, "detail": proposal.detail}
            return finish(hold(q, "agent_" + proposal.code), "agent_" + proposal.code)
        if not isinstance(proposal, ProposalBatch):
            self._consecutive_give_ups = 0
            self._consecutive_done = 0
            return finish(hold(q, "agent_invalid_result"), "agent_invalid_result")
        self._consecutive_give_ups = 0
        self._consecutive_done = 0
        if (proposal.phase not in PHASES or proposal.visual_relation not in VISUAL_RELATIONS or
                proposal.object_state not in OBJECT_STATES):
            return finish(hold(q, "agent_invalid_assessment"), "agent_invalid_assessment")
        try:
            visual_estimates, candidate_visuals = _validate_visuals(proposal)
        except _VisualContractError as exc:
            row["visual_validation"] = {"status": "invalid", "detail": str(exc)}
            return finish(hold(q, "agent_invalid_visual_estimate"),
                          "agent_invalid_visual_estimate")
        relations = _visual_relations(visual_estimates)
        row["visual_validation"] = {"status": "passed" if visual_estimates else "unknown",
                                    "source": "agent_unverified"}
        row["visual_data_sources"] = {
            "estimates": "agent_unverified" if visual_estimates else "unknown",
            "predictions": "agent_unverified",
            "actual_change": "unknown",
            "local_mapping": "observed_real_motion_only",
        }
        row["visual_estimates"] = visual_estimates
        row["visual_relations"] = relations
        row["agent_assessment"] = {"source": "agent_unverified",
                                   "phase": proposal.phase,
                                   "visual_relation": proposal.visual_relation,
                                   "object_state": proposal.object_state,
                                   "scene_summary": proposal.scene_summary}
        row["progress_evidence"] = self._assess_progress(proposal, visual_estimates)
        row["no_progress_rounds"] = self._no_progress_rounds
        if self._operator_review_required:
            stop = hold(q, "stalled_review")
            return finish(replace(stop, actions=[replace(stop.actions[0], meta={
                "request_stop": True, "stop_reason": "stalled_review"})]), "stalled_review")
        row["proposed_ids"] = [item.id for item in proposal.candidates]
        row["proposed"] = [{"id": item.id, "targets": dict(item.targets),
                            "note": item.note, "intended_effect": item.intended_effect,
                            "expected_visual_change": item.expected_visual_change,
                            "visual_prediction": candidate_visuals[item.id],
                            "status": "proposed"} for item in proposal.candidates]
        if self._clock() - started > self._inference_budget_s:
            return finish(hold(q, "inference_timeout"), "inference_timeout")

        filter_started = self._clock()
        try:
            # Geometry validation is independent from the selected freshness mode.
            checked = self._validator.validate(proposal, safe_observation, now=started)
        except InputError as exc:
            if exc.joint_pos is None:
                record_error(exc.code)
                raise
            return finish(hold(exc.joint_pos, exc.code), exc.code)
        except Exception:
            return finish(hold(q, "validation_error"), "validation_error")
        checked = self._guard_candidates(checked)
        row["filter_latency_s"] = max(0.0, self._clock() - filter_started)
        row["filtered"] = [{"id": item.id, "code": item.code, "detail": item.detail}
                           for item in checked.filtered]
        candidate_decisions: dict[str, dict[str, object]] = {}
        for item in checked.available:
            visual = candidate_visuals[item.id]
            candidate_decisions[item.id] = {
                **visual,
                "predicted_error_change": _candidate_error_change(
                    proposal.phase, visual, visual_estimates
                ),
                "history_direction_consistency": self._historical_prediction(item, visual, q),
            }
        row["candidates"] = [{"id": item.id, "status": "proposed",
                              "summary": dict(item.summary),
                              "visual_decision": candidate_decisions[item.id],
                              "verified_prefix": [action.data.tolist()
                                                  for action in item.chunk.actions]}
                             for item in checked.available]
        row["mode"] = ("none" if not checked.available else
                       "gate" if len(checked.available) == 1 else "ranking")
        if checked.observation_reason is not None:
            return finish(hold(q, checked.observation_reason), checked.observation_reason)
        if not checked.available:
            return finish(hold(q, "no_safe_candidate"), "no_safe_candidate")
        if self._clock() - started > self._inference_budget_s:
            return finish(hold(q, "inference_timeout"), "inference_timeout")

        selected_id: str
        if self._selector == "agent_preferred":
            selected_id = proposal.preferred_id
            if selected_id not in {item.id for item in checked.available}:
                return finish(hold(q, "preferred_filtered"), "preferred_filtered")
        else:
            if self._choice is None:
                return finish(hold(q, "jev_not_ready"), "jev_not_ready")
            options = [ChoiceOption(item.id, {
                "operation": "move", "risk": ("path_checked" if self._validation_mode == "measured"
                                               else "path_not_checked"),
                "steps": len(item.chunk),
                "note": item.summary["model_intent"]["note"],
                "intended_effect": item.summary["model_intent"]["intended_effect"],
                "expected_visual_change": item.summary["model_intent"]["expected_visual_change"],
                **candidate_decisions[item.id],
            }) for item in checked.available]
            options.append(ChoiceOption("hold", {"operation": "hold", "risk": "none",
                                                "use_when": "movement is unjustified or evidence is uncertain"}))
            jev_started = self._clock()
            try:
                choice = self._choice.choose_generic(
                    instruction=decoded.instruction,
                    observation_context=self._jev_context(proposal, visual_estimates, relations),
                    candidates=options)
                selected_id = choice.selected_id
                row["jev_model"] = choice.model
                row["jev_id"] = selected_id
                row["jev_probabilities"] = (dict(choice.probabilities)
                                            if choice.probabilities is not None else None)
                row["jev_usage"] = dict(choice.usage) if choice.usage is not None else None
            except ChoiceError as exc:
                row["jev_latency_s"] = max(0.0, self._clock() - jev_started)
                return finish(hold(q, "jev_" + exc.code), "jev_" + exc.code)
            except Exception:
                row["jev_latency_s"] = max(0.0, self._clock() - jev_started)
                return finish(hold(q, "jev_service_error"), "jev_service_error")
            finally:
                row["jev_latency_s"] = max(0.0, self._clock() - jev_started)
        if self._clock() - started > self._inference_budget_s:
            return finish(hold(q, "inference_timeout"), "inference_timeout")
        if self._clock() - dispatch_anchor > self._max_dispatch_age_s:
            return finish(hold(q, "dispatch_stale"), "dispatch_stale")
        if selected_id == "hold":
            self._consecutive_jev_holds += 1
            row["selected_id"] = "hold"
            row["selected_prefix"] = [q.tolist()]
            row["dispatch_status"] = "hold_requested"
            row["consecutive_jev_holds"] = self._consecutive_jev_holds
            if self._consecutive_jev_holds >= 3:
                self._operator_review_required = True
                stop = hold(q, "stalled_review")
                return finish(replace(stop, actions=[replace(stop.actions[0], meta={
                    "request_stop": True, "stop_reason": "stalled_review"})]), "stalled_review")
            return finish(hold(q, "jev_hold"), "jev_hold")
        self._consecutive_jev_holds = 0
        selected = next((item for item in checked.available if item.id == selected_id), None)
        if selected is None:
            return finish(hold(q, "jev_unknown_candidate_id"), "jev_unknown_candidate_id")
        if self._validation_mode == "measured" and any(
            selected.summary["local_checks"].get(key) != "passed"
            for key in ("full_trajectory", "interpolated_collision")
        ):
            row["blocked_selection_id"] = selected_id
            row["selected_id"] = "hold"
            row["selected_prefix"] = [q.tolist()]
            row["dispatch_status"] = "hold_requested"
            return finish(hold(q, "measured_validation_required"), "measured_validation_required")
        row["selected_id"] = selected.id
        row["selected_for_dispatch"] = selected.id
        row["selected_prefix"] = [action.data.tolist() for action in selected.chunk.actions]
        row["dispatch_status"] = "selected_for_dispatch"
        return finish(selected.chunk)
