"""One-call, observation-driven joint proposals for a Responses model.

This module never executes a tool or emits an ActionChunk. Every candidate is
unvalidated model output; the caller owns validation, selection and execution.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

import httpx
import numpy as np

from inspect_robots.spaces import Box, ObservationSpace
from inspect_robots.types import Observation

from ._llm import ENV_MODEL, resolve_provider
from ._responses import ResponsesClient, _parse_response
from ._tools import build_toolset
from .policy import _observation_content


class _RawResponsesClient(Protocol):
    """The narrow Responses surface required by the proposer."""

    def complete_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | float | None = None,
    ) -> dict[str, Any]:
        """Return a single raw Responses payload."""
        ...


@dataclass(frozen=True)
class VisualPoint:
    """One Agent-estimated normalized image point, never a measured coordinate."""

    visible: bool
    u: float | None
    v: float | None
    confidence: float
    source: Literal["agent_unverified"] = "agent_unverified"


@dataclass(frozen=True)
class CameraVisualEstimate:
    """Task-relevant points estimated independently in one camera image."""

    target_object: VisualPoint
    left_gripper: VisualPoint
    right_gripper: VisualPoint
    placement_region: VisualPoint
    source: Literal["agent_unverified"] = "agent_unverified"


@dataclass(frozen=True)
class PredictedVisualDelta:
    """Predicted normalized image displacement for a candidate's moving entity."""

    du: float | None
    dv: float | None
    confidence: float
    source: Literal["agent_unverified"] = "agent_unverified"


@dataclass(frozen=True)
class ProposalCandidate:
    """An unvalidated absolute named-joint target, with a batch-local stable ID."""

    id: str
    targets: dict[str, float]
    note: str
    intended_effect: str
    status: Literal["proposed"] = "proposed"
    expected_visual_change: str = ""
    acting_arm: Literal["left", "right", "both", "unknown"] = "unknown"
    predicted_visual_deltas: Mapping[str, PredictedVisualDelta] = field(default_factory=dict)
    visual_effect: Literal[
        "closer", "aligning", "grasping", "lifting", "carrying", "placing", "unknown"
    ] = "unknown"
    prediction_confidence: float | None = None
    verifiable_result: str = "unknown"


@dataclass(frozen=True)
class ProposalBatch:
    """A successful model proposal; `preferred_id` is advice, not a command."""

    scene_summary: str
    candidates: tuple[ProposalCandidate, ...]
    preferred_id: str
    model: str
    duration_s: float
    usage: dict[str, int] | None
    raw_response: dict[str, Any]
    status: Literal["proposed"] = "proposed"
    phase: str = "approach"
    visual_relation: str = "unknown"
    object_state: str = "unknown"
    visual_estimates: Mapping[str, CameraVisualEstimate] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalTermination:
    """Model-requested end state; the caller decides how to end its trial."""

    status: Literal["done", "give_up"]
    summary: str
    hindsight: str
    model: str
    duration_s: float
    usage: dict[str, int] | None
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ProposalFailure:
    """Auditable refusal to use a malformed, over-budget or failed response."""

    code: str
    detail: str
    model: str
    duration_s: float
    usage: dict[str, int] | None
    raw_response: dict[str, Any] | None
    status: Literal["failed"] = "failed"


ProposalResult = ProposalBatch | ProposalTermination | ProposalFailure


@dataclass(frozen=True)
class ProposalFeedback:
    """Caller report about the previous decision; None means hold/no candidate selected.

    The report is evidence from the caller, not a tool execution result. The
    fresh `Observation` passed to `propose` supplies the next camera/state data.
    """

    selected_id: str | None
    outcome: str


class _InvalidProposal(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


PHASES = frozenset({"approach", "align", "grasp", "lift", "carry", "place"})
VISUAL_RELATIONS = frozenset({"far", "near", "aligned", "unknown"})
OBJECT_STATES = frozenset({"on_table", "held", "placed", "unknown"})
ACTING_ARMS = frozenset({"left", "right", "both"})
VISUAL_EFFECTS = frozenset({
    "closer", "aligning", "grasping", "lifting", "carrying", "placing", "unknown",
})


def _enum(value: Any, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _InvalidProposal(
            "invalid_scene_assessment", f"{field} must be one of {sorted(allowed)}"
        )
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _InvalidProposal("missing_field", f"{field} must be a nonempty string")
    return value


def _finite_number(value: Any, field: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(number) or not low <= number <= high):
        raise _InvalidProposal(
            "invalid_visual_estimate", f"{field} must be finite and in [{low}, {high}]"
        )
    return number


def _nullable_number(value: Any, field: str, low: float, high: float) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field, low, high)


def _visual_point(value: Any, field: str) -> VisualPoint:
    if not isinstance(value, dict) or type(value.get("visible")) is not bool:
        raise _InvalidProposal(
            "invalid_visual_estimate", f"{field} must contain a boolean visible field"
        )
    visible = value["visible"]
    confidence = _finite_number(value.get("confidence"), f"{field}.confidence", 0.0, 1.0)
    u = _nullable_number(value.get("u"), f"{field}.u", 0.0, 1.0)
    v = _nullable_number(value.get("v"), f"{field}.v", 0.0, 1.0)
    if visible and (u is None or v is None):
        raise _InvalidProposal(
            "invalid_visual_estimate", f"{field} needs u and v when visible"
        )
    if not visible and (u is not None or v is not None):
        raise _InvalidProposal(
            "invalid_visual_estimate", f"{field} cannot contain coordinates when not visible"
        )
    return VisualPoint(visible, u, v, confidence)


def _visual_estimates(value: Any, camera_names: frozenset[str]
                      ) -> dict[str, CameraVisualEstimate]:
    if not isinstance(value, dict) or set(value) != camera_names:
        raise _InvalidProposal(
            "invalid_visual_estimate", "visual_estimates must contain exactly the declared cameras"
        )
    result: dict[str, CameraVisualEstimate] = {}
    point_names = ("target_object", "left_gripper", "right_gripper", "placement_region")
    for camera in sorted(camera_names):
        estimate = value[camera]
        if not isinstance(estimate, dict) or any(name not in estimate for name in point_names):
            raise _InvalidProposal(
                "invalid_visual_estimate", f"visual_estimates.{camera} is incomplete"
            )
        result[camera] = CameraVisualEstimate(*(
            _visual_point(estimate[name], f"visual_estimates.{camera}.{name}")
            for name in point_names
        ))
    return result


def _visual_deltas(value: Any, camera_names: frozenset[str], field: str
                   ) -> dict[str, PredictedVisualDelta]:
    if not isinstance(value, dict) or set(value) != camera_names:
        raise _InvalidProposal(
            "invalid_visual_prediction", f"{field} must contain exactly the declared cameras"
        )
    result: dict[str, PredictedVisualDelta] = {}
    for camera in sorted(camera_names):
        delta = value[camera]
        if not isinstance(delta, dict):
            raise _InvalidProposal(
                "invalid_visual_prediction", f"{field}.{camera} must be an object"
            )
        try:
            du = _nullable_number(delta.get("du"), f"{field}.{camera}.du", -1.0, 1.0)
            dv = _nullable_number(delta.get("dv"), f"{field}.{camera}.dv", -1.0, 1.0)
            confidence = _finite_number(
                delta.get("confidence"), f"{field}.{camera}.confidence", 0.0, 1.0
            )
        except _InvalidProposal as exc:
            raise _InvalidProposal("invalid_visual_prediction", str(exc)) from exc
        if (du is None) != (dv is None):
            raise _InvalidProposal(
                "invalid_visual_prediction", f"{field}.{camera} needs both du and dv or neither"
            )
        result[camera] = PredictedVisualDelta(du, dv, confidence)
    return result


def _usage(payload: dict[str, Any]) -> dict[str, int] | None:
    value = payload.get("usage")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _InvalidProposal("invalid_usage", "usage must be an object")
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        count = value.get(key)
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise _InvalidProposal(
                    "invalid_usage", f"usage.{key} must be a nonnegative integer"
                )
            result[key] = count
    return result


class AgentProposer:
    """Stateful call budget and audit log around stateless Responses requests.

    Supports only 14-D absolute joint spaces and three always-present cameras.
    Every call sends the latest observation; earlier images and joint values are
    retained in neither the next request nor a fabricated tool-result message.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        action_space: Box,
        observation_space: ObservationSpace,
        base_url: str | None = None,
        api_key_env: str | None = None,
        env: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        client: _RawResponsesClient | None = None,
        max_llm_calls: int = 100,
        candidate_count: int = 3,
        temperature: float | None = None,
        effort: str | float | None = None,
        embodiment_docs: str | None = None,
    ) -> None:
        if (
            isinstance(max_llm_calls, bool)
            or not isinstance(max_llm_calls, int)
            or max_llm_calls < 1
        ):
            raise ValueError("max_llm_calls must be an integer >= 1")
        if isinstance(candidate_count, bool) or candidate_count not in (2, 3, 4):
            raise ValueError("candidate_count must be 2, 3 or 4")
        semantics = action_space.semantics
        if (
            action_space.shape != (14,)
            or semantics is None
            or semantics.control_mode != "joint_pos"
            or len(observation_space.cameras) != 3
        ):
            raise ValueError("proposals require 14-D absolute joints and three cameras")
        toolset = build_toolset(action_space, observation_space, None, images="always")
        move, done, give_up = toolset.schemas()[:3]
        if move["function"]["name"] != "move_joints":
            raise ValueError("proposals require the move_joints tool")
        labels = tuple(semantics.dim_labels or ())
        if len(labels) != 14 or len(set(labels)) != 14:
            raise ValueError("proposals require 14 unique named joint dimensions")
        state_labels = toolset.state_labels()
        if state_labels is None:
            raise ValueError("proposals require a named 14-D joint state")
        self._state_labels = state_labels
        self._labels = frozenset(labels)
        self._camera_names = frozenset(camera.name for camera in observation_space.cameras)
        self._candidate_count = candidate_count
        self._max_llm_calls = max_llm_calls
        self._temperature = temperature
        self._effort = effort
        if embodiment_docs is not None and not isinstance(embodiment_docs, str):
            raise ValueError("embodiment_docs must be a string or None")
        self._embodiment_docs = (
            embodiment_docs.strip() if embodiment_docs is not None and embodiment_docs.strip()
            else None
        )
        provider = resolve_provider(
            model or os.environ.get(ENV_MODEL), base_url, api_key_env,
            dict(os.environ) if env is None else env,
            native_wires=frozenset({"chat", "responses"}),
        )
        self.model = provider.model
        self._client = (
            client if client is not None else ResponsesClient(provider, transport=transport)
        )
        self._calls_used = 0
        self._records: list[ProposalResult] = []
        target_description = move["function"]["parameters"]["properties"]["targets"]["description"]
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": "propose_actions",
                    "description": (
                        "Propose absolute joint targets only; no motion will be executed by "
                        "this call. " + move["function"]["description"]
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scene_summary": {"type": "string"},
                            "phase": {"type": "string", "enum": sorted(PHASES)},
                            "visual_relation": {"type": "string", "enum": sorted(VISUAL_RELATIONS)},
                            "object_state": {"type": "string", "enum": sorted(OBJECT_STATES)},
                            "visual_estimates": {
                                "type": "object",
                                "properties": {
                                    name: {
                                        "type": "object",
                                        "properties": {
                                            point: {
                                                "type": "object",
                                                "properties": {
                                                    "visible": {"type": "boolean"},
                                                    "u": {"type": ["number", "null"]},
                                                    "v": {"type": ["number", "null"]},
                                                    "confidence": {"type": "number"},
                                                },
                                                "required": ["visible", "u", "v", "confidence"],
                                            }
                                            for point in (
                                                "target_object", "left_gripper",
                                                "right_gripper", "placement_region",
                                            )
                                        },
                                        "required": [
                                            "target_object", "left_gripper", "right_gripper",
                                            "placement_region",
                                        ],
                                    }
                                    for name in sorted(self._camera_names)
                                },
                                "required": sorted(self._camera_names),
                            },
                            "candidates": {
                                "type": "array", "minItems": 2, "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "targets": {
                                            "type": "object", "description": target_description
                                        },
                                        "note": move["function"]["parameters"]["properties"][
                                            "note"
                                        ],
                                        "intended_effect": {"type": "string"},
                                        "expected_visual_change": {"type": "string"},
                                        "acting_arm": {
                                            "type": "string", "enum": sorted(ACTING_ARMS),
                                        },
                                        "predicted_visual_deltas": {
                                            "type": "object",
                                            "properties": {
                                                name: {
                                                    "type": "object",
                                                    "properties": {
                                                        "du": {"type": ["number", "null"]},
                                                        "dv": {"type": ["number", "null"]},
                                                        "confidence": {"type": "number"},
                                                    },
                                                    "required": ["du", "dv", "confidence"],
                                                }
                                                for name in sorted(self._camera_names)
                                            },
                                            "required": sorted(self._camera_names),
                                        },
                                        "visual_effect": {
                                            "type": "string", "enum": sorted(VISUAL_EFFECTS),
                                        },
                                        "prediction_confidence": {"type": "number"},
                                        "verifiable_result": {"type": "string"},
                                    },
                                    "required": ["id", "targets", "note", "intended_effect",
                                                 "expected_visual_change", "acting_arm",
                                                 "predicted_visual_deltas", "visual_effect",
                                                 "prediction_confidence", "verifiable_result"],
                                },
                            },
                            "preferred_id": {"type": "string"},
                        },
                        "required": ["scene_summary", "phase", "visual_relation",
                                     "object_state", "visual_estimates", "candidates",
                                     "preferred_id"],
                    },
                },
            },
            done,
            give_up,
        ]

    @property
    def records(self) -> tuple[ProposalResult, ...]:
        """Results in call order, including failures; no execution confirmations."""
        return tuple(self._records)

    @property
    def calls_used(self) -> int:
        """Number of model requests made in the current trial."""
        return self._calls_used

    def reset(self) -> None:
        """Begin a new trial with a fresh call budget and empty audit log."""
        self._calls_used = 0
        self._records.clear()

    def bind_embodiment_docs(self, docs: str | None) -> None:
        """Bind the real embodiment notes before the first proposal call."""
        if docs is not None and not isinstance(docs, str):
            raise ValueError("embodiment docs must be a string or None")
        self._embodiment_docs = docs.strip() if docs is not None and docs.strip() else None

    def propose(
        self, task: str, observation: Observation, feedback: ProposalFeedback | None = None
    ) -> ProposalResult:
        """Request exactly one structured tool call for the current observation."""
        _required_text(task, "task")
        if self._calls_used >= self._max_llm_calls:
            result = ProposalFailure(
                "budget_exhausted", "LLM call budget exhausted", self.model, 0.0, None, None
            )
            self._records.append(result)
            return result
        state = observation.state.get(self._state_labels[0])
        if state is None or np.asarray(state).shape != (14,) or not np.isfinite(state).all():
            raise ValueError("observation needs a finite 14-D joint state")
        if set(observation.images) != self._camera_names:
            raise ValueError("observation needs all three declared cameras")
        selected: ProposalCandidate | None = None
        if feedback is not None:
            previous = self._records[-1] if self._records else None
            if isinstance(previous, ProposalBatch):
                selected = next(
                    (candidate for candidate in previous.candidates
                     if candidate.id == feedback.selected_id),
                    None,
                )
            if feedback.selected_id is not None and selected is None:
                raise ValueError("feedback selected_id is not in the preceding batch")
            _required_text(feedback.outcome, "feedback.outcome")
        system = (
            "You inspect a real robot and propose actions only. No proposal is executed "
            "by this call. Use exactly one tool call: propose_actions, done or give_up. "
            f"Request about {self._candidate_count} distinct candidates (2-4 allowed). "
            "Use stable unique IDs within the batch. Targets are absolute named joint "
            "values; omitted joints hold their current state. Explain each intended "
            "effect and the expected visible change after one short executed segment. "
            "For every camera estimate normalized image points u/v in [0,1], visibility "
            "and confidence. If a point cannot be located, mark it invisible and set u/v "
            "to null; never guess coordinates. Predict normalized du/dv for each candidate, "
            "using null for both components when the direction is unknown. "
            "Report phase, gripper-to-object visual relation and object state as "
            "uncertain visual assessments, not measured geometry or contact. "
            "During approach the object normally stays still; assess progress by "
            "gripper-to-object relation and observed joint feedback. Avoid exploratory "
            "joint probes unless their visible task progress is justified. "
            "Base this decision on the current observation."
        )
        if self._embodiment_docs is not None:
            system += "\n\nEmbodiment notes:\n" + self._embodiment_docs
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system,
            },
            {"role": "user", "content": f"Goal: {task}"},
        ]
        if feedback is not None:
            if selected is None:
                choice = "hold (no candidate selected)"
            else:
                choice = (
                    f"{selected.id}, requested absolute targets "
                    f"{json.dumps(selected.targets, sort_keys=True)}"
                )
            messages.append({
                "role": "user",
                "content": (
                    f"Previous choice requested by the caller: {choice}. "
                    f"Caller report: {feedback.outcome}. "
                    "Only the caller can confirm what actually happened; unselected proposals "
                    "were never sent as actions."
                ),
            })
        messages.append({
            "role": "user",
            "content": _observation_content(observation, self._state_labels),
        })
        self._calls_used += 1
        started = time.perf_counter()
        try:
            payload = self._client.complete_raw(
                messages, self._tools, temperature=self._temperature, reasoning_effort=self._effort
            )
        except Exception as exc:
            result = ProposalFailure(
                "request_error", type(exc).__name__, self.model,
                time.perf_counter() - started, None, None,
            )
            self._records.append(result)
            return result
        duration = time.perf_counter() - started
        if not isinstance(payload, dict):
            result = ProposalFailure(
                "invalid_response", "response is not an object", self.model, duration, None, None
            )
            self._records.append(result)
            return result
        model = payload.get("model") if isinstance(payload.get("model"), str) else self.model
        usage: dict[str, int] | None = None
        try:
            usage = _usage(payload)
            result = self._decode(payload, model, duration, usage)
        except (
            _InvalidProposal, AttributeError, KeyError, TypeError, ValueError, OverflowError
        ) as exc:
            code = exc.code if isinstance(exc, _InvalidProposal) else "invalid_response"
            result = ProposalFailure(code, str(exc), model, duration, usage, payload)
        self._records.append(result)
        return result

    def _decode(
        self, payload: dict[str, Any], model: str, duration: float,
        usage: dict[str, int] | None,
    ) -> ProposalResult:
        if payload.get("status") != "completed":
            raise _InvalidProposal("incomplete_response", "Responses status is not completed")
        message, _ = _parse_response(payload)
        if len(message.tool_calls) != 1:
            raise _InvalidProposal("tool_call_count", "expected exactly one tool call")
        call = message.tool_calls[0]
        if call.name not in {"propose_actions", "done", "give_up"}:
            raise _InvalidProposal("unexpected_tool", f"unexpected tool {call.name!r}")
        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise _InvalidProposal("invalid_arguments", str(exc)) from exc
        if not isinstance(args, dict):
            raise _InvalidProposal("invalid_arguments", "tool arguments must be an object")
        if call.name in {"done", "give_up"}:
            status: Literal["done", "give_up"] = (
                "done" if call.name == "done" else "give_up"
            )
            field = "summary" if status == "done" else "reason"
            return ProposalTermination(
                status,
                _required_text(args.get(field), field),
                _required_text(args.get("hindsight"), "hindsight"),
                model, duration, usage, payload,
            )
        summary = _required_text(args.get("scene_summary"), "scene_summary")
        phase = _enum(args.get("phase"), "phase", PHASES)
        visual_relation = _enum(args.get("visual_relation"), "visual_relation", VISUAL_RELATIONS)
        object_state = _enum(args.get("object_state"), "object_state", OBJECT_STATES)
        visual_estimates = (
            _visual_estimates(args["visual_estimates"], self._camera_names)
            if "visual_estimates" in args else {}
        )
        candidates = args.get("candidates")
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 4:
            raise _InvalidProposal("candidate_count", "candidates must contain 2-4 items")
        parsed: list[ProposalCandidate] = []
        seen: set[str] = set()
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise _InvalidProposal("invalid_candidate", f"candidate {index} is not an object")
            candidate_id = _required_text(candidate.get("id"), f"candidates[{index}].id")
            if candidate_id in seen:
                raise _InvalidProposal("duplicate_id", f"duplicate candidate ID {candidate_id!r}")
            seen.add(candidate_id)
            targets = candidate.get("targets")
            if not isinstance(targets, dict) or not targets:
                raise _InvalidProposal("invalid_targets", f"candidate {candidate_id} needs targets")
            parsed_targets: dict[str, float] = {}
            for name, value in targets.items():
                if name not in self._labels:
                    raise _InvalidProposal("unknown_joint", f"unknown joint {name!r}")
                try:
                    valid = (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                    )
                except OverflowError:
                    valid = False
                if not valid:
                    raise _InvalidProposal("non_finite_target", f"invalid target for {name!r}")
                parsed_targets[name] = float(value)
            parsed.append(ProposalCandidate(
                candidate_id, parsed_targets,
                _required_text(candidate.get("note"), f"candidates[{index}].note"),
                _required_text(
                    candidate.get("intended_effect"), f"candidates[{index}].intended_effect"
                ),
                expected_visual_change=_required_text(
                    candidate.get("expected_visual_change"),
                    f"candidates[{index}].expected_visual_change",
                ),
                acting_arm=(
                    _enum(candidate["acting_arm"], f"candidates[{index}].acting_arm", ACTING_ARMS)
                    if "acting_arm" in candidate else "unknown"
                ),
                predicted_visual_deltas=(
                    _visual_deltas(
                        candidate["predicted_visual_deltas"], self._camera_names,
                        f"candidates[{index}].predicted_visual_deltas",
                    ) if "predicted_visual_deltas" in candidate else {}
                ),
                visual_effect=(
                    _enum(candidate["visual_effect"], f"candidates[{index}].visual_effect",
                          VISUAL_EFFECTS)
                    if "visual_effect" in candidate else "unknown"
                ),
                prediction_confidence=(
                    _finite_number(
                        candidate["prediction_confidence"],
                        f"candidates[{index}].prediction_confidence", 0.0, 1.0,
                    ) if "prediction_confidence" in candidate else None
                ),
                verifiable_result=(
                    _required_text(candidate["verifiable_result"],
                                   f"candidates[{index}].verifiable_result")
                    if "verifiable_result" in candidate else "unknown"
                ),
            ))
        preferred = _required_text(args.get("preferred_id"), "preferred_id")
        if preferred not in seen:
            raise _InvalidProposal("invalid_preferred_id", "preferred_id is not a candidate ID")
        return ProposalBatch(summary, tuple(parsed), preferred, model, duration, usage,
                             payload, phase=phase, visual_relation=visual_relation,
                             object_state=object_state, visual_estimates=visual_estimates)

    def close(self) -> None:
        """Release the owned HTTP client, if it has a connection pool."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
