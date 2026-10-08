"""Jev policy entry points; the direct branch uses bounded Choice decisions."""

from __future__ import annotations

import math
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from inspect_robots_jev.yam_contract import CAMERA_NAMES

from inspect_robots import ActionChunk, Observation, PolicyConfig, PolicyInfo, Scene
from inspect_robots_jev.yam_contract import action_space, observation_space
from inspect_robots_jev.contract import DecodedInput, InputError, decode_observation, hold
from inspect_robots_jev.pairing import strict_yam_preflight
from inspect_robots_jev.audit import (
    MAX_RECORD_BYTES, append_record, bounded_record, decision_file, load_decision,
    save_decision,
)
from inspect_robots_jev.candidates import Candidate, CandidateGenerator, CandidateSet
from inspect_robots_jev.geometry import Calibration, Localization, localize_targets
from inspect_robots_jev.jev_choice import DEFAULT_JEV_MODEL, DEFAULT_JEV_URL, ChoiceError, JevChoiceClient
from inspect_robots_jev.kinematics import YamKinematics
from inspect_robots_jev.motion import MotionPlanner
from inspect_robots_jev.vision_client import VisionClient


class _CandidateFailure(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _HoldPolicy:
    def __init__(
        self,
        name: str,
        *,
        cam_height: int = 360,
        cam_width: int = 640,
        max_image_age_s: float = 1.0,
        max_skew_s: float | None = None,
    ) -> None:
        if cam_height < 1 or cam_width < 1:
            raise ValueError("cam_height and cam_width must be >= 1")
        if not math.isfinite(max_image_age_s) or max_image_age_s <= 0:
            raise ValueError("max_image_age_s must be finite and > 0")
        self.info = PolicyInfo(
            name=name,
            action_space=action_space(),
            observation_space=observation_space(cam_height, cam_width),
        )
        self.config = PolicyConfig(action_horizon=1, replan_interval=1)
        self._cam_height = cam_height
        self._cam_width = cam_width
        self.max_image_age_s = max_image_age_s
        self.max_skew_s = max_image_age_s if max_skew_s is None else max_skew_s
        if not math.isfinite(self.max_skew_s) or self.max_skew_s <= 0:
            raise ValueError("max_skew_s must be finite and > 0")
        self._episode_calls = 0
        self._diagnostics: list[dict[str, str]] = []

    @property
    def diagnostics(self) -> tuple[dict[str, str], ...]:
        """Return detached, JSON-safe diagnostics for the current episode."""
        return tuple(dict(item) for item in self._diagnostics)

    @property
    def episode_calls(self) -> int:
        """Number of calls since reset."""
        return self._episode_calls

    def reset(self, scene: Scene) -> None:
        """Clear all episode-local counters and diagnostics."""
        self._episode_calls = 0
        self._diagnostics.clear()

    def pairing_preflight(self, embodiment: object):
        """Require the full YAM wire contract before a real rollout begins."""
        if embodiment.info.name == "yam_arms":
            return strict_yam_preflight(self, embodiment)
        return None

    def act(self, observation: Observation) -> ActionChunk:
        """Validate the whitelist and hold at the observed pose for one step."""
        self._episode_calls += 1
        try:
            decoded = decode_observation(
                observation,
                height=self._cam_height,
                width=self._cam_width,
                max_image_age_s=self.max_image_age_s,
                max_skew_s=self.max_skew_s,
            )
        except InputError as exc:
            self._diagnostics.append({"code": exc.code, "detail": str(exc)})
            if exc.joint_pos is None:
                raise
            return hold(exc.joint_pos, exc.code)
        return hold(decoded.joint_pos, "batch01_scaffold")


class JevDirectPolicy(_HoldPolicy):
    """Replan every short chunk from new RGB and encoder observations."""

    def __init__(
        self, *, vision_url: str = "http://127.0.0.1:8765/v1/detect",
        calibration_path: str | Path, mjcf_path: str | Path,
        jev_model: str = DEFAULT_JEV_MODEL, max_image_age_s: float = 1.0,
        max_skew_s: float | None = None,
        jev_url: str = DEFAULT_JEV_URL, jev_timeout_s: float = 5.0,
        vision_timeout_s: float = 5.0,
        cam_height: int = 360, cam_width: int = 640,
    ) -> None:
        super().__init__(
            "jev-direct",
            cam_height=cam_height,
            cam_width=cam_width,
            max_image_age_s=max_image_age_s,
            max_skew_s=max_skew_s,
        )
        for label, path in (("calibration_path", calibration_path), ("mjcf_path", mjcf_path)):
            if not isinstance(path, (str, Path)) or not Path(path).is_file():
                raise ValueError(f"{label} must name an existing file")
        self.calibration = Calibration.load(calibration_path)
        if any((camera.height, camera.width) != (cam_height, cam_width)
               for camera in self.calibration.cameras.values()):
            raise ValueError("calibration camera dimensions differ from policy")
        self.kinematics = YamKinematics(mjcf_path)
        self.candidates = CandidateGenerator(MotionPlanner(self.kinematics, self.calibration))
        self.vision = VisionClient(vision_url, timeout_s=vision_timeout_s,
                                   max_image_age_s=max_image_age_s,
                                   required_classes=frozenset())
        self.jev = JevChoiceClient(model=jev_model, url=jev_url, timeout_s=jev_timeout_s)
        self.config = PolicyConfig(action_horizon=self.candidates.motion.limits.max_steps,
                                   replan_interval=None)
        self._last_times: tuple[float, ...] | None = None
        self._audit: list[dict[str, object]] = []
        self._audit_omitted = 0
        self._episode_id = uuid.uuid4().hex
        self._mask_dir: Path | None = None
        self._audit_run_dir: Path | None = None
        self._audit_write_failures = 0
        self._pending_execution: dict[str, object] | None = None

    @property
    def audit_records(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(json.dumps(row)) for row in self._audit)

    def transcript(self) -> list[dict[str, object]]:
        return list(self.audit_records)

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
        """Use the allocated run directory for optional compact mask sidecars."""
        del scene_id, epoch, run_id
        self._mask_dir = Path(log_dir) / "jev-masks"
        self._audit_run_dir = Path(log_dir)

    def _persist_audit(self, row: dict[str, object]) -> None:
        if self._audit_run_dir is None:
            return
        try:
            save_decision(self._audit_run_dir, row)
        except (OSError, ValueError):
            self._audit_write_failures += 1
            row["audit_sidecar_error"] = "write_failed"

    def on_trial_end(self, record: object, log_dir: str, run_id: str) -> None:
        """Link every sidecar decision to rollout frames and publish its index."""
        del run_id
        transcript = getattr(record, "policy_transcript", None)
        steps = getattr(record, "steps", ())
        refs_by_time = {}
        for step in steps:
            stamp = getattr(step.observation, "state_time", None)
            refs = getattr(step, "image_refs", None)
            if stamp is not None and refs and stamp not in refs_by_time:
                refs_by_time[stamp] = {name: "frames/" + Path(ref.path).name
                                       for name, ref in refs.items() if name in CAMERA_NAMES}
        if isinstance(transcript, list):
            for index, row in enumerate(transcript):
                if not isinstance(row, dict):
                    continue
                observation = row.get("observation")
                if isinstance(observation, dict):
                    observation["frame_refs"] = refs_by_time.get(observation.get("state_time"), {})
                transcript[index] = bounded_record(row, os.environ.get("TYPESAFE_API_KEY"))
            record.policy_transcript = transcript
        missing = 0
        if self._audit_run_dir is not None:
            for index in range(1, self._episode_calls + 1):
                decision_id = f"{self._episode_id}-{index:06d}"
                try:
                    row = load_decision(log_dir, decision_id)
                except (OSError, ValueError, json.JSONDecodeError):
                    missing += 1
                    continue
                observation = row.get("observation")
                if isinstance(observation, dict):
                    observation["frame_refs"] = refs_by_time.get(observation.get("state_time"), {})
                    self._persist_audit(bounded_record(row, os.environ.get("TYPESAFE_API_KEY")))
            metadata = getattr(record, "metadata", None)
            if isinstance(metadata, dict):
                metadata["jev_audit"] = {
                    "episode_id": self._episode_id,
                    "relative_dir": f"jev-audit/{self._episode_id}",
                    "decision_count": self._episode_calls,
                    "record_limit_bytes": MAX_RECORD_BYTES,
                    "missing_files": missing,
                    "write_failures": self._audit_write_failures,
                }
        self._mask_dir = None
        self._audit_run_dir = None

    def reset(self, scene: Scene) -> None:
        super().reset(scene)
        self.candidates.reset()
        self._last_times = None
        self._audit.clear()
        self._audit_omitted = 0
        self._episode_id = uuid.uuid4().hex
        self._audit_write_failures = 0
        self._pending_execution = None

    def _mask_reference(self, decision_id: str, camera: str, index: int,
                        detection: object) -> dict[str, object]:
        request_id = detection.request_id
        result: dict[str, object] = {"request_id": request_id, "index": index,
                                     "mask_area_px": int(np.count_nonzero(detection.mask)),
                                     "mask_file": None}
        if self._mask_dir is not None:
            filename = f"{decision_id}_{camera}_{index:02d}.npz"
            try:
                self._mask_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self._mask_dir / filename, mask=detection.mask)
                result["mask_file"] = "jev-masks/" + filename
            except OSError:
                result["mask_file_error"] = "write_failed"
        return result

    def _resolve_execution(self, decoded: DecodedInput) -> None:
        row = self._pending_execution
        if row is None:
            return
        execution = row.get("execution")
        trajectory = row.get("trajectory")
        if not isinstance(execution, dict) or not isinstance(trajectory, list) or not trajectory:
            return
        target = np.asarray(trajectory[-1], dtype=np.float64)
        error = np.abs(decoded.joint_pos - target)
        execution.update(status="reached_joint_target" if np.max(error) <= 0.05
                         else "not_reached_joint_target",
                         observed_joint_pos=decoded.joint_pos.tolist(),
                         observed_state_time=decoded.state_time,
                         max_joint_error=float(np.max(error)))
        resolved = bounded_record(row, os.environ.get("TYPESAFE_API_KEY"))
        for index, existing in enumerate(self._audit):
            if existing.get("decision_id") == row.get("decision_id"):
                self._audit[index] = resolved
                break
        self._persist_audit(resolved)
        self._pending_execution = None

    def _augment_candidates(self, decoded: DecodedInput,
                            locations: dict[str, Localization],
                            candidates: CandidateSet,
                            audit: dict[str, object]) -> CandidateSet:
        """Hybrid extends the same finite Choice set after shared planning."""
        return candidates

    def _initial_audit(self, audit: dict[str, object]) -> None:
        """Add branch-specific provenance before any service can fail."""

    def act(self, observation: Observation) -> ActionChunk:
        started = time.monotonic()
        self._episode_calls += 1
        decision_id = f"{self._episode_id}-{self._episode_calls:06d}"
        audit: dict[str, object] = {"decision_id": decision_id, "branch": self.info.name,
                                    "model": self.jev.model, "call": self._episode_calls,
                                    "selected_id": None, "probabilities": None,
                                    "probabilities_status": "not_returned",
                                    "trajectory": None, "candidates": [], "filtered": {},
                                    "vision": {}, "localization": {}, "reason": None,
                                    "jev_latency_s": None, "selected_source": None,
                                    "observation": None, "execution": None,
                                    "audit_omitted_prior": self._audit_omitted}
        if self._audit_run_dir is not None:
            audit["audit_file"] = decision_file(decision_id)
        self._initial_audit(audit)

        def finish(chunk: ActionChunk, reason: str | None = None) -> ActionChunk:
            elapsed = time.monotonic() - started
            audit["reason"] = reason
            audit["latency_s"] = elapsed
            omitted = append_record(self._audit, audit, os.environ.get("TYPESAFE_API_KEY"))
            self._audit_omitted += omitted
            if omitted:
                self._audit[-1]["audit_omitted_prior"] = self._audit_omitted
            self._persist_audit(self._audit[-1])
            execution = self._audit[-1].get("execution")
            if isinstance(execution, dict) and execution.get("status") == "awaiting_observation":
                self._pending_execution = self._audit[-1]
            if reason is not None:
                self._diagnostics.append({"code": reason, "detail": reason})
            return replace(chunk, inference_latency_s=elapsed)

        try:
            decoded = decode_observation(observation, height=self._cam_height,
                                         width=self._cam_width,
                                         max_image_age_s=self.max_image_age_s,
                                         max_skew_s=self.max_skew_s)
        except InputError as exc:
            if exc.joint_pos is None:
                self._diagnostics.append({"code": exc.code, "detail": exc.code})
                raise
            return finish(hold(exc.joint_pos, exc.code), exc.code)
        audit["observation"] = {"state_time": decoded.state_time,
                                "image_times": dict(decoded.image_times),
                                "state_age_s": max(0.0, started - decoded.state_time),
                                "image_ages_s": {name: max(0.0, started - stamp)
                                                 for name, stamp in decoded.image_times.items()},
                                "capture_skew_s": max((decoded.state_time, *decoded.image_times.values()))
                                                  - min((decoded.state_time, *decoded.image_times.values())),
                                "joint_pos": decoded.joint_pos.tolist(), "frame_refs": {}}
        stamps = (decoded.state_time, *(decoded.image_times[name] for name in CAMERA_NAMES))
        if self._last_times is not None and any(new <= old for new, old in zip(stamps, self._last_times)):
            return finish(hold(decoded.joint_pos, "observation_not_new"), "observation_not_new")
        self._resolve_execution(decoded)
        self._last_times = stamps

        outcomes = {}
        for camera in CAMERA_NAMES:
            vision_started = time.monotonic()
            try:
                outcome = self.vision.observe(decoded.images[camera], camera, decoded.image_times[camera])
            except Exception:
                return finish(hold(decoded.joint_pos, "vision_error"), "vision_error")
            audit["vision"][camera] = {"state": outcome.state, "failure": outcome.failure,
                                        "latency_s": time.monotonic() - vision_started,
                                        "detection_count": len(outcome.detections),
                                        "detections": [{"category": d.category,
                                                        "box": list(d.box),
                                                        "mask_ref": self._mask_reference(decision_id, camera, i, d),
                                                        "detection_score": d.detection_score,
                                                        "segmentation_score": d.segmentation_score,
                                                        "occluded": d.occluded,
                                                        "model_version": d.model_version}
                                                       for i, d in enumerate(outcome.detections)]}
            if outcome.state != "ready" and outcome.failure != "empty_detection":
                return finish(hold(decoded.joint_pos, "vision_" + (outcome.failure or "error")),
                              "vision_" + (outcome.failure or "error"))
            outcomes[camera] = outcome
        if not any(outcome.state == "ready" for outcome in outcomes.values()):
            return finish(hold(decoded.joint_pos, "vision_empty_detection"), "vision_empty_detection")
        if time.monotonic() - min(stamps) > self.max_image_age_s:
            return finish(hold(decoded.joint_pos, "stale_time"), "stale_time")
        try:
            planning_started = time.monotonic()
            locations = localize_targets(decoded, outcomes, self.calibration, self.kinematics)
            audit["localization"] = {name: item.diagnostic() for name, item in locations.items()}
            base_candidates = self.candidates.generate(decoded, locations)
            audit["candidates"] = [item.jev_summary() for item in base_candidates.available]
            audit["filtered"] = dict(base_candidates.filtered)
            candidates = self._augment_candidates(decoded, locations, base_candidates, audit)
            audit["candidates"] = [item.jev_summary() for item in candidates.available]
            audit["filtered"] = dict(candidates.filtered)
            audit["planning_latency_s"] = time.monotonic() - planning_started
            audit["stage"] = candidates.stage.value
        except _CandidateFailure as exc:
            return finish(hold(decoded.joint_pos, exc.code), exc.code)
        except Exception:
            return finish(hold(decoded.joint_pos, "planning_error"), "planning_error")
        motions = [item for item in candidates.available if item.motion is not None and item.motion.safe]
        if not motions:
            return finish(candidates.fallback(decoded.joint_pos), "no_safe_candidate")
        if time.monotonic() - min(stamps) > self.max_image_age_s:
            return finish(hold(decoded.joint_pos, "stale_time"), "stale_time")
        jev_started = time.monotonic()
        jev_failure = None
        try:
            choice = self.jev.choose(instruction=decoded.instruction, candidates=candidates)
        except ChoiceError as exc:
            jev_failure = "jev_" + exc.code
        except Exception:
            jev_failure = "jev_service_error"
        finally:
            audit["jev_latency_s"] = time.monotonic() - jev_started
        if jev_failure is not None:
            return finish(hold(decoded.joint_pos, jev_failure), jev_failure)
        audit["selected_id"] = choice.selected_id
        audit["probabilities"] = dict(choice.probabilities) if choice.probabilities is not None else None
        audit["probabilities_status"] = "returned" if choice.probabilities is not None else "missing"
        if time.monotonic() - min(stamps) > self.max_image_age_s:
            return finish(hold(decoded.joint_pos, "stale_time"), "stale_time")
        candidate = candidates.by_id(choice.selected_id)
        audit["selected_source"] = candidate.jev_summary()["source"]
        if candidate.operation in ("hold", "reobserve"):
            try:
                chunk = candidate.chunk(decoded.joint_pos)
                if candidate.operation == "reobserve":
                    self.candidates.mark_dispatched(candidate, decoded, candidates)
            except ValueError:
                return finish(hold(decoded.joint_pos, "candidate_invalid"), "candidate_invalid")
            audit["trajectory"] = [action.data.tolist() for action in chunk.actions]
            return finish(chunk, candidate.operation)
        try:
            chunk = candidate.chunk(decoded.joint_pos)
            if candidate.operation != "molmo":
                self.candidates.mark_dispatched(candidate, decoded, base_candidates)
        except ValueError:
            return finish(hold(decoded.joint_pos, "candidate_invalid"), "candidate_invalid")
        audit["trajectory"] = [action.data.tolist() for action in chunk.actions]
        audit["execution"] = {"status": "awaiting_observation",
                               "target_joint_pos": audit["trajectory"][-1],
                               "observed_joint_pos": None, "max_joint_error": None}
        return finish(chunk)


class JevHybridPolicy(JevDirectPolicy):
    """Choose one checked Molmo prefix or one checked shared EEF correction."""

    def __init__(
        self, *, molmo_url: str = "http://127.0.0.1:8202", molmo_prefix_steps: int = 3,
        molmo_timeout_s: float = 120.0, **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.info = replace(self.info, name="jev-hybrid")
        if (type(molmo_prefix_steps) is not int or molmo_prefix_steps < 1 or
                molmo_prefix_steps > self.candidates.motion.limits.max_steps):
            raise ValueError("molmo_prefix_steps must be within the shared short-motion horizon")
        from inspect_robots_jev.molmo_client import MolmoActClient
        self.molmo = MolmoActClient(molmo_url, timeout_s=molmo_timeout_s)
        self.molmo_prefix_steps = molmo_prefix_steps

    @property
    def server_url(self) -> str:
        return self.molmo.url.removesuffix("/act")

    @property
    def server_metadata_url(self) -> str:
        return self.molmo.url

    def _initial_audit(self, audit: dict[str, object]) -> None:
        identity = self.molmo.identity
        audit.update(molmo_checkpoint=identity[0] if identity else None,
                     molmo_revision=identity[1] if identity else None,
                     molmo_latency_s=None, molmo_server_latency_ms=None)

    def _augment_candidates(self, decoded: DecodedInput,
                            locations: dict[str, Localization],
                            candidates: CandidateSet,
                            audit: dict[str, object]) -> CandidateSet:
        from inspect_robots_jev.molmo_client import MolmoError
        try:
            result = self.molmo.infer(decoded)
        except MolmoError as exc:
            identity = self.molmo.identity
            if identity is not None:
                audit.update(molmo_checkpoint=identity[0], molmo_revision=identity[1])
            audit["filtered"]["molmo:prefix"] = "molmo_" + exc.code
            raise _CandidateFailure("molmo_" + exc.code) from None
        audit.update(molmo_checkpoint=result.checkpoint, molmo_revision=result.revision,
                     molmo_latency_s=result.latency_s,
                     molmo_server_latency_ms=result.server_latency_ms)
        prefix = [action.data for action in result.chunk.actions[:self.molmo_prefix_steps]]
        checked = self.candidates.motion.check_trajectory(
            decoded.joint_pos, prefix, "both", box=locations.get("box"))
        if not checked.safe:
            audit["filtered"]["molmo:prefix"] = checked.reason or "unsafe"
            raise _CandidateFailure("molmo_" + (checked.reason or "unsafe"))
        molmo = Candidate(
            "molmo:prefix", candidates.stage, None, None, "molmo", None, None,
            checked,
            {"risk": "checked", "source": "molmo", "start_step": 0,
             "prefix_steps": len(prefix), "returned_steps": len(result.chunk.actions),
             "checkpoint": result.checkpoint, "revision": result.revision},
        )
        return CandidateSet((*candidates.available, molmo), candidates.filtered, candidates.stage)
