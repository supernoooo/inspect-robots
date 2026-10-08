"""Offline, proposal-only replay of saved YAM observations.

This module has no embodiment, rollout, controller, or action dispatch path.
The only rig object is a data declaration consumed by local candidate checks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from inspect_robots import Observation
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots_agent.proposals import (
    AgentProposer, ProposalBatch, ProposalCandidate, ProposalFailure,
    ProposalFeedback, ProposalTermination,
)
from inspect_robots_yam.config import YamConfig, action_box, observation_space

from .agent_candidates import AgentCandidateValidator
from .audit import bounded_record, decision_file, save_decision
from .contract import InputError, _joint_pos, decode_observation
from .geometry import Calibration
from .jev_choice import ChoiceError, ChoiceOption, JevChoiceClient
from .kinematics import YamKinematics
from .motion import MotionPlanner
from .yam_contract import CAMERA_NAMES, DIM_LABELS


_EPISODE = re.compile(r"[0-9a-f]{32}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_CAMERAS = frozenset(CAMERA_NAMES)
_JOINTS = frozenset(DIM_LABELS)


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _clean_audit(row: dict[str, object]) -> dict[str, object]:
    clean = bounded_record(row, None)
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            clean = bounded_record(clean, secret)
    return clean


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size > 32_000_000:
        raise ValueError("missing_or_oversized_rgb")
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            image = np.load(stream, allow_pickle=False)
    else:
        try:
            from PIL import Image
        except ImportError:
            raise ValueError("invalid_rgb") from None

        with Image.open(path) as source:
            if source.mode != "RGB":
                raise ValueError("invalid_rgb")
            image = np.asarray(source)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("invalid_rgb")
    return image.copy()


def _observation(entry: Mapping[str, object], base: Path, now: float) -> tuple[Observation, dict[str, object]]:
    replay_at = _positive(entry.get("replay_at"), "replay_at")
    raw_times = entry.get("image_times")
    raw_files = entry.get("rgb")
    if not isinstance(raw_times, dict) or not isinstance(raw_files, dict):
        raise ValueError("missing_camera")
    if set(raw_times) != _CAMERAS or set(raw_files) != _CAMERAS:
        raise ValueError("missing_camera")
    state_time = _positive(entry.get("state_time"), "state_time")
    image_times = {name: _positive(raw_times[name], f"image_times.{name}") for name in CAMERA_NAMES}
    images: dict[str, np.ndarray] = {}
    refs: dict[str, str] = {}
    for name in CAMERA_NAMES:
        filename = raw_files[name]
        if not isinstance(filename, str) or not filename:
            raise ValueError("missing_camera")
        path = base / filename
        try:
            images[name] = _load_rgb(path)
        except (OSError, ValueError):
            raise ValueError("missing_rgb" if not path.is_file() else "invalid_rgb") from None
        refs[name] = filename
    rebased_times = {name: now - (replay_at - stamp) for name, stamp in image_times.items()}
    rebased_state = now - (replay_at - state_time)
    instruction = entry.get("instruction")
    q = entry.get("joint_pos")
    observation = Observation(
        images=images, state={"joint_pos": q}, instruction=instruction,
        image_times=rebased_times, state_time=rebased_state,
    )
    source = {"state_time": state_time, "image_times": image_times,
              "state_age_s": replay_at - state_time,
              "image_ages_s": {name: replay_at - stamp for name, stamp in image_times.items()},
              "frame_refs": refs}
    return observation, source


def _fixture_proposal(entry: Mapping[str, object]) -> ProposalBatch | ProposalFailure | ProposalTermination:
    fixture = entry.get("agent")
    if not isinstance(fixture, dict):
        return ProposalFailure("missing_fixture", "", "fixture", 0.0, None, None)
    status = fixture.get("status", "proposed")
    if status in ("done", "give_up"):
        return ProposalTermination(status, "fixture stop", "", "fixture", 0.0, None, {})
    if status == "failed":
        return ProposalFailure("request_error", "", "fixture", 0.0, None, None)
    raw = fixture.get("candidates")
    if not isinstance(raw, list) or not raw:
        return ProposalFailure("invalid_fixture", "", "fixture", 0.0, None, None)
    candidates = []
    for item in raw:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or _ID.fullmatch(item["id"]) is None):
            return ProposalFailure("invalid_fixture", "", "fixture", 0.0, None, None)
        candidates.append(ProposalCandidate(
            item["id"], item.get("targets", {}),
            item.get("note", "fixture proposal"),
            item.get("intended_effect", "fixture effect")))
    preferred = fixture.get("preferred_id", candidates[0].id)
    if not isinstance(preferred, str) or _ID.fullmatch(preferred) is None:
        return ProposalFailure("invalid_fixture", "", "fixture", 0.0, None, None)
    return ProposalBatch("saved fixture", tuple(candidates), preferred,
                         "fixture", 0.0, None, {})


def make_data_validator(rig: Mapping[str, object], calibration: str | Path,
                        mjcf: str | Path, *, max_image_age_s: float = 1.0,
                        max_skew_s: float = 0.2) -> AgentCandidateValidator:
    """Build local geometry from configuration and files, with no hardware object."""
    cfg = YamConfig(**rig)
    declaration = SimpleNamespace(
        _cfg=cfg,
        info=EmbodimentInfo("yam_arms", action_box(cfg.low, cfg.high),
                            observation_space(cfg.cam_height, cfg.cam_width,
                                              CAMERA_NAMES), is_simulated=False),
    )
    motion = MotionPlanner(YamKinematics(mjcf), Calibration.load(calibration))
    return AgentCandidateValidator(
        declaration, motion, max_image_age_s=max_image_age_s, max_skew_s=max_skew_s)


def replay(manifest: Mapping[str, object], *, base: Path, validator: AgentCandidateValidator,
           clock: Callable[[], float] = time.monotonic,
           proposer: AgentProposer | None = None,
           choice: JevChoiceClient | None = None,
           out_dir: Path | None = None) -> list[dict[str, object]]:
    """Compare Agent and JEV choices over one checked set; write proposal-only sidecars."""
    episode = manifest.get("episode_id")
    rounds = manifest.get("rounds")
    if not isinstance(episode, str) or _EPISODE.fullmatch(episode) is None:
        raise ValueError("episode_id must be 32 lowercase hex digits")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("rounds must be a nonempty array")
    max_age = _positive(manifest.get("max_image_age_s", 1.0), "max_image_age_s")
    max_skew = _positive(manifest.get("max_skew_s", 0.2), "max_skew_s")
    max_dispatch = _positive(manifest.get("max_dispatch_age_s", 0.5), "max_dispatch_age_s")
    budget = _positive(manifest.get("inference_budget_s", 30.0), "inference_budget_s")
    if min(max_age, max_skew, max_dispatch, budget) <= 0:
        raise ValueError("age and budget limits must be positive")
    rows: list[dict[str, object]] = []
    last_times: tuple[float, ...] | None = None
    last_replay_at: float | None = None
    last_frames: tuple[bytes, ...] | None = None
    previous: dict[str, object] | None = None
    pending: dict[str, object] | None = None
    for index, entry in enumerate(rounds, 1):
        if not isinstance(entry, dict):
            raise ValueError("round must be an object")
        started = clock()
        decision_id = f"{episode}-{index:06d}"
        row: dict[str, object] = {
            "audit_schema_version": 1, "decision_id": decision_id,
            "branch": "jev-agent-replay", "call": index, "proposed_only": True,
            "task": {"scene_id": manifest.get("scene_id"),
                     "instruction": entry.get("instruction")},
            "rig_fingerprint": dict(getattr(validator, "_provenance", {})),
            "proposed_ids": [], "proposed": [], "candidates": [], "filtered": [],
            "agent_preferred_id": None, "agent_selection": None,
            "jev_id": None, "jev_selection": None, "jev_probabilities": None,
            "selected_id": None, "selected_for_dispatch": None,
            "selected_prefix": None, "dispatch_status": "not_requested",
            "reason": None, "mode": None, "observation": None,
            "observed_state": None,
            "approval": {"status": "not_applicable", "events": [], "proposed_only": True},
            "execution": {"status": "proposal_only", "proposed_only": True},
            "controller_trace": {"status": "not_linked", "proposed_only": True},
        }

        def finish(reason: str | None = None, *, stop: bool = False) -> None:
            row["reason"] = reason
            row["latency_s"] = max(0.0, clock() - started)
            row["audit_file"] = decision_file(decision_id)
            if reason is not None:
                row["dispatch_status"] = "proposed_hold"
                q = row.get("observation")
                if isinstance(q, dict) and isinstance(q.get("joint_pos"), list):
                    row["selected_prefix"] = [q["joint_pos"]]
            if stop:
                row["termination"] = {"status": reason, "proposed_only": True}
            nonlocal previous
            clean = _clean_audit(row)
            rows.append(clean)
            previous = clean
            if out_dir is not None:
                save_decision(out_dir, clean)

        try:
            safe_q = _joint_pos(entry.get("joint_pos"))
            validator._check_rig_state(safe_q)
            row["observation"] = {"joint_pos": safe_q.tolist()}
        except InputError as exc:
            finish(exc.code)
            continue
        try:
            obs, source = _observation(entry, base, started)
            row["observation"] = {**source, "joint_pos": safe_q.tolist()}
            decoded = decode_observation(obs, height=validator._cfg.cam_height,
                                         width=validator._cfg.cam_width,
                                         max_image_age_s=max_age, max_skew_s=max_skew,
                                         now=started)
            validator._check_rig_state(decoded.joint_pos)
        except InputError as exc:
            finish(exc.code)
            continue
        except (OSError, ValueError, TypeError) as exc:
            code = str(exc)
            if code in {"missing_rgb", "invalid_rgb", "missing_camera"}:
                finish(code)
            elif code.startswith(("replay_at ", "state_time ", "image_times.")):
                finish("invalid_time")
            else:
                finish("invalid_manifest")
            continue
        raw_stamps = (source["state_time"], *(source["image_times"][name] for name in CAMERA_NAMES))
        replay_at = _positive(entry["replay_at"], "replay_at")
        if last_replay_at is not None and replay_at <= last_replay_at:
            finish("replay_time_not_new")
            continue
        if last_times is not None and any(new <= old for new, old in zip(raw_stamps, last_times)):
            finish("observation_not_new")
            continue
        frames = tuple(hashlib.sha256(decoded.images[name].tobytes()).digest()
                       for name in CAMERA_NAMES)
        if last_frames is not None and frames == last_frames:
            finish("duplicate_frame")
            continue
        last_replay_at = replay_at
        last_times = raw_stamps
        last_frames = frames
        if pending is not None:
            target = np.asarray(pending["selected_prefix"][-1], dtype=np.float64)
            error = float(np.max(np.abs(decoded.joint_pos - target)))
            pending["observed_state"] = {"joint_pos": decoded.joint_pos.tolist(),
                                          "state_time": source["state_time"],
                                          "image_times": source["image_times"],
                                          "max_joint_error_to_proposed": error,
                                          "comparison_only": True}
            pending["execution"] = {"status": "comparison_only",
                                     "proposed_only": True,
                                     "matches_proposed_target": error <= 0.05}
            if out_dir is not None:
                save_decision(out_dir, _clean_audit(pending))
            pending = None
            if error > 0.05:
                finish("next_state_mismatch")
                continue
        if max(source["state_age_s"], *source["image_ages_s"].values()) > max_dispatch:
            finish("dispatch_stale")
            continue
        approvals = entry.get("approvals", [])
        if not isinstance(approvals, list):
            finish("invalid_approval")
            continue
        if previous is not None:
            mismatched = any(not isinstance(item, dict) or item.get("decision_id") != previous["decision_id"]
                             for item in approvals)
            if mismatched:
                row["approval"] = {"status": "id_mismatch", "events": [], "proposed_only": True}
                previous["approval"] = row["approval"]
                previous["audit_incomplete"] = True
                if out_dir is not None:
                    save_decision(out_dir, _clean_audit(previous))
                finish("approval_id_mismatch")
                continue
            row["approval"] = {"status": "reported" if approvals else "none_reported",
                               "proposed_only": True,
                               "events": [{"detail": item.get("detail") if item.get("detail") in
                                           ("clamped", "delta_clamped", "rejected") else "modified"}
                                          for item in approvals]}
            previous["approval"] = row["approval"]
            if out_dir is not None:
                save_decision(out_dir, _clean_audit(previous))
        feedback = None
        if previous is not None:
            feedback = ProposalFeedback(previous.get("selected_id"), json.dumps({
                "decision_id": previous["decision_id"], "selection": previous.get("selected_id"),
                "reason": previous.get("reason"), "proposed_only": True,
                "comparison": previous.get("observed_state"),
                "approval": row["approval"],
            }, separators=(",", ":")))
        agent_started = clock()
        try:
            if proposer is None:
                proposal = _fixture_proposal(entry)
            elif "feedback" in inspect.signature(proposer.propose).parameters:
                proposal = proposer.propose(decoded.instruction, obs, feedback=feedback)
            else:
                proposal = proposer.propose(decoded.instruction, obs)
        except Exception:
            row["agent_latency_s"] = max(0.0, clock() - agent_started)
            finish("agent_error")
            continue
        row["agent_latency_s"] = max(0.0, clock() - agent_started)
        if isinstance(proposal, (ProposalBatch, ProposalFailure, ProposalTermination)):
            row["agent_model"] = proposal.model
            row["agent_duration_s"] = proposal.duration_s
            row["agent_usage"] = proposal.usage
        if isinstance(proposal, ProposalTermination):
            finish(proposal.status, stop=True)
            break
        if isinstance(proposal, ProposalFailure):
            finish("agent_" + proposal.code)
            continue
        if not isinstance(proposal, ProposalBatch):
            finish("agent_invalid_result")
            continue
        if (any(not isinstance(item.id, str) or _ID.fullmatch(item.id) is None
                for item in proposal.candidates)
                or not isinstance(proposal.preferred_id, str)
                or _ID.fullmatch(proposal.preferred_id) is None):
            finish("agent_invalid_candidate_id")
            continue
        row["proposed_ids"] = [item.id for item in proposal.candidates]
        row["proposed"] = [{"id": item.id,
                            "targets": {name: float(value) for name, value in item.targets.items()
                                        if name in _JOINTS and type(value) in (int, float)
                                        and math.isfinite(value)},
                            "status": "proposed", "proposed_only": True}
                           for item in proposal.candidates]
        row["agent_preferred_id"] = proposal.preferred_id
        if clock() - started > budget:
            finish("inference_timeout")
            continue
        filter_started = clock()
        try:
            checked = validator.validate(proposal, obs, now=clock())
        except InputError as exc:
            finish(exc.code)
            continue
        except Exception:
            finish("validation_error")
            continue
        row["filter_latency_s"] = max(0.0, clock() - filter_started)
        row["filtered"] = [{"id": item.id, "code": item.code, "proposed_only": True}
                           for item in checked.filtered]
        row["candidates"] = [{"id": item.id, "status": "proposed",
                              "verified_prefix": [action.data.tolist() for action in item.chunk.actions],
                              "proposed_only": True} for item in checked.available]
        row["mode"] = ("none" if not checked.available else
                       "gate" if len(checked.available) == 1 else "ranking")
        if checked.observation_reason is not None:
            finish(checked.observation_reason)
            continue
        if not checked.available:
            finish("no_safe_candidate")
            continue
        by_id = {item.id: item for item in checked.available}
        row["agent_selection"] = (proposal.preferred_id if proposal.preferred_id in by_id else "hold")
        row["agent_selection_reason"] = (None if proposal.preferred_id in by_id else "preferred_filtered")
        options = [ChoiceOption(item.id, {"operation": "move", "risk": "locally_checked",
                                          "steps": len(item.chunk),
                                          "note": item.summary["model_intent"]["note"],
                                          "intended_effect": item.summary["model_intent"]["intended_effect"]})
                   for item in checked.available]
        options.append(ChoiceOption("hold", {"operation": "hold", "risk": "none"}))
        jev_started = clock()
        try:
            if choice is None:
                selected = entry.get("jev_id", "hold")
                if not isinstance(selected, str):
                    raise ChoiceError("invalid_choice")
                probabilities = None
                model = "fixture"
                usage = None
            else:
                result = choice.choose_generic(instruction=decoded.instruction,
                                               observation_context="Current observation; candidate motions locally checked",
                                               candidates=options)
                selected, probabilities, model, usage = (result.selected_id, result.probabilities,
                                                         result.model, result.usage)
        except ChoiceError as exc:
            row["jev_latency_s"] = max(0.0, clock() - jev_started)
            finish("jev_" + exc.code)
            continue
        except Exception:
            row["jev_latency_s"] = max(0.0, clock() - jev_started)
            finish("jev_service_error")
            continue
        row["jev_latency_s"] = max(0.0, clock() - jev_started)
        row["jev_model"] = model
        if not isinstance(selected, str) or _ID.fullmatch(selected) is None:
            finish("jev_invalid_candidate_id")
            continue
        row["jev_id"] = selected
        row["jev_probabilities"] = (
            {key: float(value) for key, value in probabilities.items()
             if key in {option.id for option in options} and type(value) in (int, float)
             and math.isfinite(value)} if probabilities is not None else None)
        row["jev_usage"] = dict(usage) if usage is not None else None
        if clock() - started > budget:
            finish("inference_timeout")
            continue
        if clock() - min(obs.state_time, *obs.image_times.values()) > max_dispatch:
            finish("dispatch_stale")
            continue
        if selected not in by_id and selected != "hold":
            finish("jev_unknown_candidate_id")
            continue
        row["jev_selection"] = selected
        row["selected_id"] = selected
        if selected == "hold":
            finish("jev_hold")
            continue
        row["selected_prefix"] = [action.data.tolist() for action in by_id[selected].chunk.actions]
        row["dispatch_status"] = "proposed_only"
        finish()
        pending = rows[-1]
    if pending is not None:
        pending["execution"] = {"status": "unverified_no_next_observation", "proposed_only": True}
        if out_dir is not None:
            save_decision(out_dir, _clean_audit(pending))
    return rows
