"""Batch 08 attended segment gate. No device is opened by this module.

The caller owns device lifecycle and supplies the same Inspect Robots approver
chain used for ordinary YAM rollout. A segment is prepared from a jev-agent
decision, reviewed by a human, then emitted through DefaultController. Every
action is reviewed by that approver before embodiment.step. The gate stops on
any discrepancy; it never silently substitutes a target or advances a stage.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from inspect_robots import Action, ActionChunk, Observation
from inspect_robots.approver import Approver, ClampApprover, DeltaLimitApprover
from inspect_robots.controller import DefaultController
from inspect_robots.errors import SafetyAbort
from inspect_robots.frames import FrameStore
from inspect_robots.types import StepResult
from inspect_robots_yam.collision import CollisionApprover

from .contract import InputError, camera_frame_ids, decode_observation, frames_advanced
from .yam_contract import CAMERA_NAMES


class StageStop(SafetyAbort):
    """A stage failed before further motor commands were allowed."""


@dataclass(frozen=True)
class StageLimits:
    """Values signed in Batch 07; never derive these from synthetic fixtures."""

    rig_fingerprint: Mapping[str, object]
    batch07_record: str
    batch07_operator_signature: str
    agent_model: str
    jev_model: str
    max_image_age_s: float
    max_skew_s: float
    max_dispatch_age_s: float
    max_joint_error: float
    max_micro_step: float = 0.01
    freshness_mode: str = "frame_sequence"

    def __post_init__(self) -> None:
        if not all((self.rig_fingerprint, self.batch07_record, self.batch07_operator_signature,
                    self.agent_model, self.jev_model)):
            raise ValueError("signed Batch 07 record, rig and model fingerprints required")
        for name in ("max_image_age_s", "max_skew_s", "max_dispatch_age_s",
                     "max_joint_error", "max_micro_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_micro_step > 0.01:
            raise ValueError("first micro motion cannot exceed 0.01 per step")
        if self.freshness_mode not in {"frame_sequence", "capture_age"}:
            raise ValueError("invalid stage freshness_mode")


class _ChunkPolicy:
    """Give the core controller only this reviewed prefix, once."""

    def __init__(self, chunk: ActionChunk) -> None:
        self.chunk = chunk
        self.calls = 0

    def act(self, observation: Observation) -> ActionChunk:
        del observation
        self.calls += 1
        if self.calls != 1:
            raise StageStop("old suffix or second inference requested")
        return self.chunk


def _has_approver(approver: object, kind: type, seen: set[int] | None = None) -> bool:
    """Find a required gate in a core chain or a transparent test wrapper."""
    if seen is None:
        seen = set()
    if id(approver) in seen:
        return False
    seen.add(id(approver))
    if isinstance(approver, kind):
        return True
    members = getattr(approver, "_approvers", ())
    if not isinstance(members, (tuple, list)):
        members = ()
    return any(_has_approver(member, kind, seen) for member in members) or (
        getattr(approver, "inner", None) is not None and
        _has_approver(approver.inner, kind, seen))


class StagedMotionGate:
    """Explicit prepare/permit/execute/observe/sign cycle for four stages.

    ``execute`` can drive a supplied embodiment. Batch 08 offline tests supply
    only a fake embodiment; real use requires separately signed Batch 07 and a
    field owner. A stopped segment must be retried from a new decision.
    """

    STAGES = ("hold", "micro", "jev_choice", "multi_round")
    _MICRO_GROUPS = ("left_arm", "right_arm", "left_gripper", "right_gripper")

    def __init__(self, limits: StageLimits, *, lock_path: Path,
                 frame_store: FrameStore, trial_id: str,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(frame_store, FrameStore) or not trial_id:
            raise ValueError("FrameStore and unique trial_id required")
        self.limits, self.lock_path, self.clock = limits, Path(lock_path), clock
        self.frame_store, self.trial_id = frame_store, trial_id
        self._frame_index = 0
        self.stage_index = 0
        self.signed: list[dict[str, object]] = []
        self.segments: list[dict[str, object]] = []
        self._prepared: tuple[ActionChunk, Observation, dict[str, object]] | None = None
        self._permit: dict[str, str] | None = None
        self._used_ids: set[str] = set()
        self._last_observation: Observation | None = None
        self._last_input: Observation | None = None
        self._stopped = False
        self._store: dict[str, Any] = {}
        self._bound_approver: Approver | None = None
        self._restart_stage = False
        if self.lock_path.is_symlink():
            raise StageStop("control lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise StageStop("another process holds the YAM control lock") from exc
        self._lock_fd: int | None = fd

    def close(self) -> None:
        """Release only the advisory lock; never call the YAM driver."""
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self) -> StagedMotionGate:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def stage(self) -> str:
        return self.STAGES[self.stage_index] if self.stage_index < 4 else "complete"

    def prepare_hold(self, observation: Observation, *, decision_id: str) -> str:
        """Stage 1: create one absolute hold from the current measured pose."""
        if self.stage != "hold":
            raise StageStop("hold preparation is only for stage 1")
        q = self._decode(observation, self.clock())
        row = {"decision_id": decision_id, "rig_fingerprint": dict(self.limits.rig_fingerprint),
               "agent_model": self.limits.agent_model, "jev_model": self.limits.jev_model,
               "decision_started_at": self.clock(),
               "selected_id": "hold", "selected_for_dispatch": None,
               "selected_prefix": [q.tolist()], "dispatch_status": "hold_requested",
               "candidates": [], "selector": "jev", "jev_id": "hold"}
        return self.prepare(ActionChunk([Action(q.copy(), {"decision_id": decision_id})]),
                            row, observation)

    def prepare_from_policy(self, policy: object, observation: Observation,
                            approver: Approver) -> str:
        """Use the existing jev-agent proposal/validation/choice pipeline."""
        from .agent_policy import JevAgentPolicy

        if not isinstance(policy, JevAgentPolicy) or self.stage == "hold":
            raise StageStop("stage needs a paired jev-agent policy")
        if self.stage == "complete" or self._prepared is not None or self._stopped:
            raise StageStop("stage unavailable or previous segment unresolved")
        if self.stage == "jev_choice" and any(
            row.get("stage") == self.stage and row.get("result") == "awaiting_operator_signature"
            for row in self.segments
        ):
            raise StageStop("stage awaits operator signature")
        if not all(_has_approver(approver, kind)
                   for kind in (ClampApprover, DeltaLimitApprover, CollisionApprover)):
            raise StageStop("candidate review needs the actual YAM approver chain")
        if self._bound_approver is not None and approver is not self._bound_approver:
            raise StageStop("YAM approver chain changed during session")
        if policy._freshness_mode != self.limits.freshness_mode:
            raise StageStop("policy and stage freshness modes differ")
        self._bound_approver = approver
        policy.set_candidate_guardrail(approver, self._store)
        chunk = policy.act(observation)
        row = policy.audit_records[-1]
        if chunk.meta.get("decision_id") != row.get("decision_id"):
            raise StageStop("policy chunk and audit decision ID differ")
        if any(action.meta.get("request_stop") for action in chunk.actions):
            raise StageStop("policy requires operator review")
        return self.prepare(chunk, row, observation)

    def _decode(self, observation: Observation, now: float) -> np.ndarray:
        if set(observation.images) != set(CAMERA_NAMES):
            raise StageStop("three cameras required")
        height, width = next(iter(observation.images.values())).shape[:2]
        decoded = decode_observation(
            observation, height=height, width=width,
            max_image_age_s=(self.limits.max_image_age_s
                             if self.limits.freshness_mode == "capture_age" else None),
            max_skew_s=(self.limits.max_skew_s
                        if self.limits.freshness_mode == "capture_age" else None), now=now)
        if self.limits.freshness_mode == "frame_sequence":
            try:
                camera_frame_ids(observation)
            except InputError as exc:
                raise StageStop(exc.code) from exc
        elif now - min(decoded.state_time, *decoded.image_times.values()) > self.limits.max_dispatch_age_s:
            raise StageStop("observation exceeded dispatch age")
        return decoded.joint_pos

    def _require_newer(self, current: Observation, previous: Observation) -> None:
        if current.state_time <= previous.state_time:
            raise StageStop("next joint sample is not new")
        if self.limits.freshness_mode == "frame_sequence":
            try:
                new_ids = camera_frame_ids(current)
                old_ids = camera_frame_ids(previous)
            except InputError as exc:
                raise StageStop(exc.code) from exc
            if not frames_advanced(new_ids, old_ids):
                raise StageStop("next camera frames are not new")
        elif any(current.image_times[name] <= previous.image_times[name]
                 for name in CAMERA_NAMES):
            raise StageStop("next observation is not new")

    def _reject_regression(self, current: Observation, previous: Observation) -> None:
        if current.state_time < previous.state_time:
            raise StageStop("next joint sample moved backward")
        if self.limits.freshness_mode == "frame_sequence":
            try:
                new_ids = camera_frame_ids(current)
                old_ids = camera_frame_ids(previous)
            except InputError as exc:
                raise StageStop(exc.code) from exc
            if any(new_ids[name][0] != old_ids[name][0] or
                   new_ids[name][1] < old_ids[name][1] for name in CAMERA_NAMES):
                raise StageStop("next camera frames moved backward")
        elif any(current.image_times[name] < previous.image_times[name]
                 for name in CAMERA_NAMES):
            raise StageStop("next observation moved backward")

    def _save_frames(self, observation: Observation) -> dict[str, str]:
        refs = {name: self.frame_store.put(self.trial_id, self._frame_index,
                                           name, observation.images[name]).path
                for name in CAMERA_NAMES}
        self._frame_index += 1
        return refs

    def prepare(self, chunk: ActionChunk, decision: Mapping[str, object],
                observation: Observation) -> str:
        if self._lock_fd is None or self.stage_index >= 4 or self._prepared is not None or self._stopped:
            raise StageStop("stage unavailable or previous segment unresolved")
        if self.stage in ("hold", "jev_choice") and any(
            row["stage"] == self.stage and row["result"] == "awaiting_operator_signature"
            for row in self.segments
        ):
            raise StageStop("stage awaits operator signature")
        if self._last_observation is not None:
            self._reject_regression(observation, self._last_observation)
        if self._last_input is not None:
            self._require_newer(observation, self._last_input)
        q = self._decode(observation, self.clock())
        decision_started = decision.get("decision_started_at")
        if (type(decision_started) not in (int, float) or
                not math.isfinite(decision_started)):
            raise StageStop("decision start missing or invalid")
        elapsed = self.clock() - decision_started
        if elapsed < 0 or elapsed > self.limits.max_dispatch_age_s:
            raise StageStop("decision expired before preparation")
        if dict(decision.get("rig_fingerprint", {})) != dict(self.limits.rig_fingerprint):
            raise StageStop("rig or model fingerprint changed")
        if decision.get("agent_model") != self.limits.agent_model or decision.get("jev_model") != self.limits.jev_model:
            raise StageStop("Agent or JEV model changed")
        decision_id = decision.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id or decision_id in self._used_ids:
            raise StageStop("missing or reused decision ID")
        if not 1 <= len(chunk) <= 3:
            raise StageStop("segment must contain one to three actions")
        vectors = [np.asarray(a.data, dtype=np.float64) for a in chunk.actions]
        if any(v.shape != (14,) or not np.isfinite(v).all() for v in vectors):
            raise StageStop("invalid 14-D action")
        if any(a.meta.get("decision_id") != decision_id for a in chunk.actions):
            raise StageStop("action decision ID mismatch")
        selected = decision.get("selected_id")
        expected = decision.get("selected_prefix")
        if not isinstance(expected, list) or len(expected) != len(vectors):
            raise StageStop("selected prefix missing or truncated")
        try:
            matches = all(np.array_equal(v, np.asarray(e, dtype=np.float64))
                          for v, e in zip(vectors, expected))
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise StageStop("controller prefix differs from selected prefix")
        if self.stage == "hold":
            if len(vectors) != 1 or not np.array_equal(vectors[0], q):
                raise StageStop("stage 1 requires one current-pose hold")
        else:
            if decision.get("validation_mode") != "measured":
                raise StageStop("measured validation required for motion")
            if decision.get("candidate_guardrail") != "active":
                raise StageStop("YAM candidate guardrail preflight missing")
            if decision.get("selected_for_dispatch") != selected or selected in (None, "hold"):
                raise StageStop("motion requires a selected safe candidate")
            available = decision.get("candidates")
            if not isinstance(available, list) or selected not in [c.get("id") for c in available if isinstance(c, dict)]:
                raise StageStop("selected candidate absent from local checks")
            selected_record = next(c for c in available if isinstance(c, dict) and c.get("id") == selected)
            checks = selected_record.get("summary", {}).get("local_checks", {})
            if any(checks.get(key) != "passed" for key in
                   ("full_trajectory", "interpolated_collision")):
                raise StageStop("full trajectory and interpolated collision checks required")
            if decision.get("dispatch_status") != "selected_for_dispatch":
                raise StageStop("policy did not select for dispatch")
            if self.stage in ("jev_choice", "multi_round"):
                if decision.get("selector") != "jev" or decision.get("jev_id") != selected:
                    raise StageStop("JEV selection missing or differs")
            if self.stage == "jev_choice" and len({c.get("id") for c in available if isinstance(c, dict)}) < 2:
                raise StageStop("stage 3 requires at least two safe candidates plus hold")
        group = None
        if self.stage == "micro":
            prior = q
            groups = set()
            moved_index: int | None = None
            direction: float | None = None
            for target in vectors:
                changed = np.flatnonzero(~np.isclose(target, prior, atol=1e-10))
                if len(changed) != 1 or abs(float(target[changed[0]] - prior[changed[0]])) > self.limits.max_micro_step + 1e-12:
                    raise StageStop("micro step must move one joint by at most 0.01")
                index = int(changed[0])
                delta = float(target[index] - prior[index])
                if moved_index is not None and (index != moved_index or delta * direction <= 0):
                    raise StageStop("micro segment must keep one joint and direction")
                moved_index, direction = index, delta
                groups.add("left_gripper" if index == 6 else "right_gripper" if index == 13
                           else "left_arm" if index < 6 else "right_arm")
                prior = target
            if len(groups) != 1:
                raise StageStop("micro segment mixes joints or sides")
            group = groups.pop()
        if self.stage == "multi_round":
            current_rounds = [r for r in self.segments if r.get("stage") == "multi_round"
                              and r.get("result") == "awaiting_operator_signature"]
            if current_rounds or not self._restart_stage:
                previous = current_rounds[-1] if current_rounds else self.segments[-1] if self.segments else None
                if (previous is None or previous.get("result") not in
                        ("passed", "awaiting_operator_signature") or
                        decision.get("agent_feedback_decision_id") != previous.get("decision_id") or
                        decision.get("agent_feedback_observed") is not True):
                    raise StageStop("Agent did not receive observed previous-decision feedback")
        self._prepared = (chunk, observation, {"decision_id": decision_id,
                          "selected_id": selected, "stage": self.stage,
                          "group": group, "prefix": [v.tolist() for v in vectors],
                          "candidate_ids": [c.get("id") for c in decision.get("candidates", [])
                                            if isinstance(c, dict)],
                          "jev_id": decision.get("jev_id"),
                          "agent_feedback_decision_id": decision.get("agent_feedback_decision_id"),
                          "restarted_stage": self._restart_stage,
                          "input_joint_pos": q.tolist(),
                          "input_state_time": observation.state_time,
                          "decision_started_at": decision_started,
                          "input_image_times": dict(observation.image_times),
                          "input_frame_ids": dict(observation.extra.get("camera_frame_ids", {})),
                          "input_frame_refs": self._save_frames(observation)})
        return decision_id

    def permit(self, decision_id: str, *, operator: str, estop_watcher: str,
               workspace_checked: bool, direction_checked: bool = False) -> None:
        if self._prepared is None or self._prepared[2]["decision_id"] != decision_id:
            raise StageStop("no matching prepared decision")
        if not operator.strip() or not estop_watcher.strip() or not workspace_checked:
            raise StageStop("operator, emergency-stop watcher and workspace check required")
        if self.stage == "micro" and not direction_checked:
            raise StageStop("micro joint/gripper direction requires operator review")
        self._permit = {"operator": operator, "estop_watcher": estop_watcher,
                        "direction_checked": str(direction_checked),
                        "permitted_at": str(self.clock())}

    def execute(self, embodiment: object, approver: Approver) -> dict[str, object]:
        if self._lock_fd is None or self._prepared is None or self._permit is None or self._stopped:
            raise StageStop("segment lacks current human permit")
        chunk, observation, row = self._prepared
        if self._bound_approver is not None and approver is not self._bound_approver:
            self.stop("YAM approver chain changed after candidate selection")
            raise StageStop("YAM approver chain changed after candidate selection")
        if not callable(getattr(approver, "review", None)) or not all(
            _has_approver(approver, kind)
            for kind in (ClampApprover, DeltaLimitApprover, CollisionApprover)
        ):
            self.stop("installed YAM clamp, delta and collision approver chain required")
            raise StageStop("installed YAM clamp, delta and collision approver chain required")
        controller = DefaultController()
        policy = _ChunkPolicy(chunk)
        state = self._store
        current = observation
        approvals: list[dict[str, object]] = []
        readbacks: list[list[float]] = []
        controller_trace: list[dict[str, object]] = []
        step_frame_refs: list[dict[str, str]] = []
        try:
            for index, expected in enumerate(row["prefix"]):
                if self._stopped:
                    raise StageStop("operator stopped segment")
                if self.clock() - row["decision_started_at"] > self.limits.max_dispatch_age_s:
                    raise StageStop("decision expired before action")
                action = controller.next_action(policy, current, index, state)
                if action.meta.get("decision_id") != row["decision_id"] or not np.array_equal(action.data, expected):
                    raise StageStop("controller emitted wrong decision or old suffix")
                reviewed = approver.review(action, state)
                if reviewed is not action or reviewed.meta.get("decision_id") != row["decision_id"] or not np.array_equal(reviewed.data, expected):
                    raise StageStop("YAM approver modified action; stop for review")
                if self._stopped:
                    raise StageStop("operator stopped segment")
                approvals.append({"step": index, "status": "approved", "decision_id": row["decision_id"]})
                result: StepResult = embodiment.step(reviewed)
                if not isinstance(result, StepResult) or result.terminated or result.truncated:
                    raise StageStop("embodiment terminated during segment")
                current = result.observation
                measured = self._decode(current, self.clock())
                self._require_newer(current, observation)
                error = float(np.max(np.abs(measured - np.asarray(expected))))
                readbacks.append(measured.tolist())
                controller_trace.append({"step": index, "decision_id": row["decision_id"],
                                         "reviewed_action": np.asarray(reviewed.data).tolist()})
                step_frame_refs.append(self._save_frames(current))
                if error > self.limits.max_joint_error:
                    raise StageStop("joint readback error or wrong direction")
                if any(result.info.get(flag) for flag in
                       ("collision_warning", "gripper_fault", "drift", "unexpected_observation")):
                    raise StageStop("embodiment reported anomaly")
                observation = current
            if self._stopped:
                raise StageStop("operator stopped segment")
            evidence = {**row, "permit": dict(self._permit), "approval": approvals,
                        "joint_readbacks": readbacks,
                        "controller_trace": controller_trace,
                        "step_frame_refs": step_frame_refs,
                        "next_state_time": current.state_time,
                        "next_image_times": dict(current.image_times),
                        "next_frame_ids": dict(current.extra.get("camera_frame_ids", {})),
                        "max_joint_error": max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                                               for a, b in zip(readbacks, row["prefix"])),
                        "result": "awaiting_operator_signature"}
            self.segments.append(evidence)
            self._used_ids.add(str(row["decision_id"]))
            self._last_observation = current
            self._last_input = self._prepared[1]
            self._restart_stage = False
            self._prepared = None
            self._permit = None
            return evidence
        except Exception as exc:
            self._stopped = True
            self.segments.append({**row, "permit": dict(self._permit),
                                  "approval": approvals, "joint_readbacks": readbacks,
                                  "controller_trace": controller_trace,
                                  "step_frame_refs": step_frame_refs,
                                  "result": "failed", "stop_reason": str(exc)})
            raise

    def sign_stage(self, *, operator: str, estop_watcher: str,
                   evidence_reviewed: bool) -> None:
        if self._prepared is not None or self._stopped or not evidence_reviewed:
            raise StageStop("unresolved or failed segment")
        rows = [r for r in self.segments if r["stage"] == self.stage and r["result"] == "awaiting_operator_signature"]
        if not rows or not operator.strip() or not estop_watcher.strip():
            raise StageStop("signed stage needs successful evidence and two names")
        if self.stage == "micro" and set(self._MICRO_GROUPS) - {r["group"] for r in rows}:
            raise StageStop("both arms and both grippers need direction checks")
        if self.stage == "multi_round" and len(rows) < 2:
            raise StageStop("multi-round stage needs two new decisions with Agent feedback")
        for row in rows:
            row["result"] = "passed"
        self.signed.append({"stage": self.stage, "operator": operator,
                            "estop_watcher": estop_watcher,
                            "decision_ids": [r["decision_id"] for r in rows],
                            "signed_at": self.clock()})
        self.stage_index += 1

    def recover(self, observation: Observation, *, operator: str, estop_watcher: str,
                fault_reviewed: bool) -> None:
        """Resample the rig, reset approval state and retry the same stage."""
        if not self._stopped or not fault_reviewed or not operator.strip() or not estop_watcher.strip():
            raise StageStop("fault review and two operators required")
        self._decode(observation, self.clock())
        if self._last_input is not None:
            self._require_newer(observation, self._last_input)
        for row in self.segments:
            if row.get("stage") == self.stage and row.get("result") == "awaiting_operator_signature":
                row["result"] = "invalidated_by_fault"
        self._last_observation = observation
        self._store.clear()
        self._prepared = None
        self._permit = None
        self._restart_stage = True
        self._stopped = False

    def stop(self, reason: str) -> None:
        """Latch an operator-reported wrong direction, warning or scene anomaly."""
        if not reason.strip():
            raise ValueError("stop reason required")
        self._stopped = True
        self.segments.append({"stage": self.stage, "result": "failed",
                              "stop_reason": reason, "decision_id":
                              self._prepared[2]["decision_id"] if self._prepared else None})

    def save_record(self, path: Path) -> None:
        """Atomically save software evidence; attach raw logs and human forms separately."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "batch07_record": self.limits.batch07_record,
                   "batch07_operator_signature": self.limits.batch07_operator_signature,
                   "rig_fingerprint": dict(self.limits.rig_fingerprint),
                   "agent_model": self.limits.agent_model, "jev_model": self.limits.jev_model,
                   "limits": {name: getattr(self.limits, name) for name in (
                       "max_image_age_s", "max_skew_s", "max_dispatch_age_s",
                       "max_joint_error", "max_micro_step", "freshness_mode")},
                   "current_stage": self.stage, "stopped": self._stopped,
                   "segments": self.segments, "signed_stages": self.signed}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
        fd, temp = tempfile.mkstemp(prefix=".batch08-", suffix=".json", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
