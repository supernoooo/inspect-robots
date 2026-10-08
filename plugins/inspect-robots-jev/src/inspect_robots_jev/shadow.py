"""Read-only live sampling and proposal-only JEV decisions for Batch 07.

The reader callbacks are supplied by a separately audited device owner. This
module neither creates a YAM embodiment nor owns a driver lifecycle. In
particular, a read callback is *not* evidence that connecting it is motionless.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from inspect_robots import Observation
from inspect_robots_agent.proposals import (
    AgentProposer, ProposalBatch, ProposalFailure, ProposalFeedback, ProposalTermination,
)
from inspect_robots_yam.config import YamConfig

from .agent_candidates import AgentCandidateValidator
from .audit import decision_file, save_decision
from .contract import InputError, _joint_pos, decode_observation
from .jev_choice import ChoiceError, ChoiceOption, JevChoiceClient
from .replay import _clean_audit
from .yam_contract import CAMERA_NAMES


@dataclass(frozen=True)
class CameraRead:
    images: Mapping[str, np.ndarray]
    image_times: Mapping[str, float]
    frame_ids: Mapping[str, str]


@dataclass(frozen=True)
class JointRead:
    joint_pos: np.ndarray
    state_time: float


class ReadOnlySource(Protocol):
    """Only sample methods; connection and shutdown need separate field audit."""

    def read_cameras(self) -> CameraRead: ...

    def read_joints(self) -> JointRead: ...


def _limit(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_shadow_rig(cfg: YamConfig) -> None:
    """Reject unsafe runtime settings before any field source is loaded."""
    if (cfg.control_interface != "joints" or cfg.joints_are_delta or
            not cfg.collision_guardrail or not cfg.collision_table or
            cfg.auto_start or cfg.unattended or
            cfg.gripper_closed != 0.0 or cfg.gripper_open != 1.0):
        raise ValueError("shadow rig safety configuration is not active")
    if any(getattr(cfg, field) is None for field in (
        "collision_left_base_pos", "collision_right_base_pos",
        "collision_left_base_yaw", "collision_right_base_yaw",
        "collision_table_height")):
        raise ValueError("shadow requires measured rig geometry")


class ShadowRunner:
    """One sampled observation per decision; no action dispatch capability."""

    def __init__(self, source: ReadOnlySource, validator: AgentCandidateValidator,
                 proposer: AgentProposer, choice: JevChoiceClient, *, out_dir: Path,
                 max_dispatch_age_s: float, max_image_age_s: float,
                 max_skew_s: float, inference_budget_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic,
                 episode_id: str | None = None) -> None:
        self.source, self.validator = source, validator
        self.proposer, self.choice = proposer, choice
        self.out_dir, self.clock = Path(out_dir), clock
        self.max_dispatch_age_s = _limit(max_dispatch_age_s, "max_dispatch_age_s")
        self.max_image_age_s = _limit(max_image_age_s, "max_image_age_s")
        self.max_skew_s = _limit(max_skew_s, "max_skew_s")
        self.inference_budget_s = _limit(inference_budget_s, "inference_budget_s")
        validate_shadow_rig(validator._cfg)
        self.episode_id = episode_id or uuid.uuid4().hex
        if len(self.episode_id) != 32 or any(c not in "0123456789abcdef" for c in self.episode_id):
            raise ValueError("episode_id must be 32 lowercase hex digits")
        self.rows: list[dict[str, object]] = []
        self._last_times: tuple[float, ...] | None = None

    def run_round(self, instruction: str, *, scene_change: str) -> dict[str, object]:
        if scene_change not in {"unchanged", "changed", "unknown"}:
            raise ValueError("scene_change must be unchanged, changed, or unknown")
        index = len(self.rows) + 1
        decision_id = f"{self.episode_id}-{index:06d}"
        started = self.clock()
        row: dict[str, object] = {
            "audit_schema_version": 1, "branch": "jev-agent-shadow",
            "decision_id": decision_id, "call": index, "proposed_only": True,
            "task": {"instruction": instruction},
            "rig_fingerprint": dict(self.validator._provenance),
            "scene_change": scene_change, "observation": None,
            "proposed_ids": [], "filtered": [], "candidates": [],
            "agent_preferred_id": None, "jev_id": None, "selected_id": None,
            "selected_for_dispatch": None, "selected_prefix": None,
            "dispatch_status": "proposed_hold", "reason": None,
            "approval": {"status": "not_applicable", "proposed_only": True},
            "execution": {"status": "proposal_only", "proposed_only": True},
            "controller_trace": {"status": "not_linked", "proposed_only": True},
            "max_dispatch_age_s": self.max_dispatch_age_s,
            "max_image_age_s": self.max_image_age_s,
            "max_skew_s": self.max_skew_s,
        }
        observation: Observation | None = None
        safe_joint: list[float] | None = None
        capture_started = self.clock()

        def finish(reason: str | None) -> dict[str, object]:
            row["reason"] = reason
            if reason is not None and safe_joint is not None:
                row["selected_prefix"] = [safe_joint]
            finished = self.clock()
            row["latency_s"] = max(0.0, finished - started)
            if observation is not None:
                stamps = (observation.state_time, *observation.image_times.values())
                if all(isinstance(t, (int, float)) and math.isfinite(t) for t in stamps):
                    row["dispatch_age_s"] = max(0.0, finished - min(stamps))
            row["audit_file"] = decision_file(decision_id)
            clean = _clean_audit(row)
            save_decision(self.out_dir, clean)
            self.rows.append(clean)
            return clean

        try:
            cameras = self.source.read_cameras()
            joints = self.source.read_joints()
            row["capture_latency_s"] = max(0.0, self.clock() - capture_started)
            if set(cameras.images) != set(CAMERA_NAMES) or set(cameras.image_times) != set(CAMERA_NAMES):
                return finish("missing_camera")
            if set(cameras.frame_ids) != set(CAMERA_NAMES):
                return finish("missing_frame_id")
            images = {name: np.asarray(cameras.images[name]).copy() for name in CAMERA_NAMES}
            observation = Observation(images=images, state={"joint_pos": joints.joint_pos},
                                      instruction=instruction,
                                      image_times=dict(cameras.image_times),
                                      state_time=joints.state_time)
            checked_joint = _joint_pos(joints.joint_pos)
            self.validator._check_rig_state(checked_joint)
            safe_joint = checked_joint.tolist()
            row["observation"] = {
                "state_time": joints.state_time,
                "image_times": dict(cameras.image_times),
                "frame_ids": dict(cameras.frame_ids),
                "joint_pos": safe_joint,
            }
            decoded = decode_observation(
                observation, height=self.validator._cfg.cam_height,
                width=self.validator._cfg.cam_width,
                max_image_age_s=self.max_image_age_s, max_skew_s=self.max_skew_s,
                now=self.clock())
            self.validator._check_rig_state(decoded.joint_pos)
            stamps = (decoded.state_time, *(decoded.image_times[name] for name in CAMERA_NAMES))
            if self._last_times is not None and any(a <= b for a, b in zip(stamps, self._last_times)):
                return finish("observation_not_new")
            self._last_times = stamps
            refs = {}
            for name in CAMERA_NAMES:
                frame = decoded.images[name]
                frame_path = self.out_dir / "shadow-frames" / self.episode_id / f"{index:06d}-{name}.npy"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                with frame_path.open("wb") as stream:
                    np.save(stream, frame, allow_pickle=False)
                refs[name] = {"path": str(frame_path.relative_to(self.out_dir)),
                              "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                              "frame_id": cameras.frame_ids[name], "capture_time": decoded.image_times[name]}
            row["observation"]["frame_refs"] = refs
            if self.clock() - min(stamps) > self.max_dispatch_age_s:
                return finish("dispatch_stale")
            if scene_change != "unchanged":
                return finish("scene_" + scene_change)
            agent_started = self.clock()
            previous = self.rows[-1] if self.rows else None
            feedback = (ProposalFeedback(previous.get("selected_id"),
                                         '{"proposed_only":true}') if previous else None)
            try:
                if "feedback" in inspect.signature(self.proposer.propose).parameters:
                    proposal = self.proposer.propose(instruction, observation, feedback=feedback)
                else:
                    proposal = self.proposer.propose(instruction, observation)
            except Exception:
                return finish("agent_error")
            finally:
                row["agent_latency_s"] = max(0.0, self.clock() - agent_started)
            if isinstance(proposal, ProposalTermination):
                return finish(proposal.status)
            if isinstance(proposal, ProposalFailure):
                return finish("agent_" + proposal.code)
            if not isinstance(proposal, ProposalBatch):
                return finish("agent_invalid_result")
            row["agent_model"] = proposal.model
            row["agent_usage"] = proposal.usage
            row["proposed_ids"] = [item.id for item in proposal.candidates]
            row["agent_preferred_id"] = proposal.preferred_id
            if self.clock() - started > self.inference_budget_s:
                return finish("inference_timeout")
            filter_started = self.clock()
            try:
                checked = self.validator.validate(proposal, observation, now=self.clock())
            except InputError as exc:
                return finish(exc.code)
            except Exception:
                return finish("validation_error")
            finally:
                row["filter_latency_s"] = max(0.0, self.clock() - filter_started)
            row["filtered"] = [{"id": item.id, "code": item.code, "proposed_only": True}
                               for item in checked.filtered]
            row["candidates"] = [{"id": item.id,
                                  "verified_prefix": [action.data.tolist() for action in item.chunk.actions],
                                  "proposed_only": True} for item in checked.available]
            if checked.observation_reason:
                return finish(checked.observation_reason)
            if not checked.available:
                return finish("no_safe_candidate")
            by_id = {item.id: item for item in checked.available}
            options = [ChoiceOption(item.id, {"operation": "move", "risk": "locally_checked",
                                              "steps": len(item.chunk),
                                              "note": item.summary["model_intent"]["note"],
                                              "intended_effect": item.summary["model_intent"]["intended_effect"]})
                       for item in checked.available]
            options.append(ChoiceOption("hold", {"operation": "hold", "risk": "none"}))
            jev_started = self.clock()
            try:
                result = self.choice.choose_generic(
                    instruction=instruction,
                    observation_context="Current observation; candidate motions locally checked",
                    candidates=options)
            except ChoiceError as exc:
                return finish("jev_" + exc.code)
            except Exception:
                return finish("jev_service_error")
            finally:
                row["jev_latency_s"] = max(0.0, self.clock() - jev_started)
            row["jev_model"] = result.model
            row["jev_usage"] = result.usage
            row["jev_id"] = result.selected_id
            if self.clock() - started > self.inference_budget_s:
                return finish("inference_timeout")
            if self.clock() - min(stamps) > self.max_dispatch_age_s:
                return finish("dispatch_stale")
            if result.selected_id not in by_id and result.selected_id != "hold":
                return finish("jev_unknown_candidate_id")
            row["selected_id"] = result.selected_id
            if result.selected_id == "hold":
                return finish("jev_hold")
            row["selected_prefix"] = [action.data.tolist() for action in by_id[result.selected_id].chunk.actions]
            row["dispatch_status"] = "proposed_only"
            return finish(None)
        except InputError as exc:
            return finish(exc.code)
        except (OSError, ValueError, TypeError):
            return finish("observation_error")
        except Exception:
            # A reader must never turn an unexpected device fault into a proposal.
            return finish("read_error")

    def summary(self) -> dict[str, object]:
        """Latency statistics are descriptive; field staff sign the limits."""
        def stats(field: str) -> dict[str, float | None]:
            values = [float(row[field]) for row in self.rows if isinstance(row.get(field), (int, float))]
            return {"p95_s": float(np.percentile(values, 95)) if values else None,
                    "max_s": max(values) if values else None}
        count = len(self.rows)
        over_age = sum(row.get("reason") in {"dispatch_stale", "stale_time"}
                       for row in self.rows)
        return {"proposed_only": True, "episode_id": self.episode_id,
                "rounds": count, "latency": {field: stats(field) for field in (
                    "latency_s", "capture_latency_s", "agent_latency_s",
                    "filter_latency_s", "jev_latency_s", "dispatch_age_s")},
                "over_age_hold_count": over_age,
                "over_age_hold_rate": over_age / count if count else None,
                "scene_changes": {state: sum(row.get("scene_change") == state for row in self.rows)
                                  for state in ("unchanged", "changed", "unknown")},
                "max_dispatch_age_s": self.max_dispatch_age_s,
                "max_image_age_s": self.max_image_age_s,
                "max_skew_s": self.max_skew_s}
