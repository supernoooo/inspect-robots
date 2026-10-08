"""Bounded TypeSafe Choice requests over checked generic or legacy candidates."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .candidates import CandidateSet

DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_JEV_URL = "https://api.typesafe.ai/v1/systemone"
_MODEL = re.compile(r"jev-\d+\.\d+\.\d+\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FORBIDDEN_KEY = re.compile(
    r"(?:rgb|image|pixel|base64|credential|secret|password|api[_-]?key|"
    r"preferred[_-]?id|joint[_-]?pos|qpos|target|coord|xyz|trajectory|pose|position|"
    r"(?:^|[_-])(?:x|y|z)(?:[_-]|$)|pixel|bbox|mask|depth|payload|"
    r"(?:^|[_-])raw(?:[_-]|$))",
    re.IGNORECASE,
)
_FORBIDDEN_TEXT = re.compile(
    r"\b(?:rgb_base64|preferred_id|api_key|password|credential|joint_pos)\b",
    re.IGNORECASE,
)
_ALLOWED_VISUAL_KEYS = frozenset({
    "u", "v", "du", "dv", "visible", "confidence", "source", "camera",
    "target_object", "left_gripper", "right_gripper", "placement_region",
    "normalized_visual_points", "visual_estimates", "predicted_visual_deltas",
    "target_to_placement", "left_to_target", "right_to_target",
    "current_error", "predicted_error", "error_reduction", "distance",
})
_MAX_CONTEXT_CHARS = 4096
_MAX_SUMMARY_BYTES = 4096


class ChoiceError(Exception):
    """Safe error code only; never includes a response body, URL, or credential."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ChoiceOption:
    """Candidate ID and checked text/numeric summary; no raw action data."""

    id: str
    summary: object


@dataclass(frozen=True)
class ChoiceResult:
    selected_id: str
    probabilities: Mapping[str, float] | None
    model: str
    latency_s: float = 0.0
    usage: Mapping[str, int] | None = None


def _safe_summary(value: object, *, depth: int = 0) -> object:
    """Detach generic summaries and reject binary data and unvetted coordinates."""
    if depth > 4:
        raise ChoiceError("invalid_candidates")
    if isinstance(value, str):
        if len(value) > _MAX_CONTEXT_CHARS or _FORBIDDEN_TEXT.search(value):
            raise ChoiceError("unsafe_summary")
        key = os.environ.get("TYPESAFE_API_KEY")
        if key and key in value:
            raise ChoiceError("unsafe_summary")
        return value
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ChoiceError("invalid_candidates")
        return value
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise ChoiceError("invalid_candidates")
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ChoiceError("invalid_candidates")
            if _FORBIDDEN_KEY.search(key) and key not in _ALLOWED_VISUAL_KEYS:
                raise ChoiceError("unsafe_summary")
            result[key] = _safe_summary(item, depth=depth + 1)
        return result
    raise ChoiceError("invalid_candidates")


class JevChoiceClient:
    def __init__(self, *, model: str = DEFAULT_JEV_MODEL, url: str = DEFAULT_JEV_URL,
                 timeout_s: float = 5.0, max_response_bytes: int = 262_144,
                 transport: Callable[..., Any] | None = None) -> None:
        if not isinstance(model, str) or not _MODEL.fullmatch(model):
            raise ValueError("jev_model must be a pinned version such as jev-1.13.0")
        parsed = urllib.parse.urlsplit(url)
        if (parsed.path != "/v1/systemone" or parsed.query or parsed.fragment or
                parsed.username or parsed.password or not parsed.hostname or
                (parsed.scheme != "https" and not
                 (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}))):
            raise ValueError("jev_url must be an HTTPS or local /v1/systemone endpoint")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or max_response_bytes < 1:
            raise ValueError("invalid Jev transport limits")
        self.model = model
        self.url = url
        self.timeout_s = timeout_s
        self.max_response_bytes = max_response_bytes
        self.transport = transport

    def choose_generic(self, *, instruction: str, observation_context: str,
                       candidates: Sequence[ChoiceOption]) -> ChoiceResult:
        """Select one ID from summaries produced after the caller's safety filter.

        Choice failures never imply a fallback candidate. Generic summaries
        cannot carry raw positions, trajectories, images, credentials or an
        Agent preference.
        """
        if not isinstance(observation_context, str) or not observation_context.strip():
            raise ChoiceError("invalid_observation_context")
        key = os.environ.get("TYPESAFE_API_KEY")
        if (not isinstance(instruction, str) or not instruction.strip()):
            raise ChoiceError("invalid_instruction")
        if (len(instruction) > _MAX_CONTEXT_CHARS or _FORBIDDEN_TEXT.search(instruction) or
                (key and key in instruction)):
            raise ChoiceError("unsafe_instruction")
        if (len(observation_context) > _MAX_CONTEXT_CHARS or
                _FORBIDDEN_TEXT.search(observation_context) or
                (key and key in observation_context)):
            raise ChoiceError("unsafe_observation_context")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ChoiceError("invalid_candidates")
        if not candidates or len(candidates) > 255:
            raise ChoiceError("invalid_candidates")
        options: list[ChoiceOption] = []
        for candidate in candidates:
            if not isinstance(candidate, ChoiceOption):
                raise ChoiceError("invalid_candidates")
            summary = _safe_summary(candidate.summary)
            try:
                size = len(json.dumps(summary, allow_nan=False).encode("utf-8"))
            except (TypeError, ValueError, OverflowError):
                raise ChoiceError("invalid_candidates") from None
            if size > _MAX_SUMMARY_BYTES:
                raise ChoiceError("invalid_candidates")
            options.append(ChoiceOption(candidate.id, summary))
        return self._choose(instruction=instruction,
                            state={"observation_context": observation_context}, options=options)

    def choose(self, *, instruction: str, candidates: CandidateSet) -> ChoiceResult:
        """Adapt checked CandidateSet while preserving the legacy wire summary."""
        options = [ChoiceOption(item.id, item.jev_summary()) for item in candidates.available]
        return self._choose(instruction=instruction, state={"stage": candidates.stage.value},
                            options=options)

    def _choose(self, *, instruction: str, state: Mapping[str, str],
                options: Sequence[ChoiceOption]) -> ChoiceResult:
        ids = [item.id for item in options]
        if (not ids or len(ids) > 255 or any(not isinstance(key, str) or not _ID.fullmatch(key)
                                              for key in ids)):
            raise ChoiceError("invalid_candidates")
        if len(ids) != len(set(ids)):
            raise ChoiceError("duplicate_candidate_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ChoiceError("invalid_instruction")
        criteria = {item.id: item.summary for item in options}
        # The request body is built exclusively from the decoded instruction and
        # candidate summaries; raw observations and simulator data stay local.
        body = {"state": {"instruction": instruction, **state},
                "model": self.model,
                "questions": {"action": {"type": "choice",
                                         "instructions": ("Select the candidate that makes justified progress "
                                                          "toward the current phase. Hold only when moving is "
                                                          "unjustified or the evidence is insufficient. "
                                                          "A selected action is still subject to execution gates."),
                                         "criteria": criteria}}}
        try:
            payload = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            raise ChoiceError("invalid_candidates") from None
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key or not key.strip():
            raise ChoiceError("missing_api_key")
        if key.encode("utf-8") in payload:
            raise ChoiceError("unsafe_request")
        opener = self.transport or urllib.request.urlopen
        started = time.monotonic()
        try:
            request = urllib.request.Request(
                self.url, data=payload, method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            with opener(request, timeout=self.timeout_s) as response:
                if response.status != 200:
                    raise ChoiceError("authentication_error" if response.status in (401, 403)
                                      else "service_error")
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ChoiceError("response_too_large")
        except urllib.error.HTTPError as exc:
            raise ChoiceError("authentication_error" if exc.code in (401, 403) else "service_error") from None
        except (TimeoutError, socket.timeout):
            raise ChoiceError("timeout") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ChoiceError("timeout") from None
            raise ChoiceError("service_error") from None
        except ChoiceError:
            raise
        except Exception:
            raise ChoiceError("service_error") from None
        latency_s = time.monotonic() - started
        try:
            value: Any = json.loads(raw)
            if not isinstance(value, dict) or value.get("model") != self.model:
                raise ChoiceError("model_mismatch")
            answers = value.get("answers")
            answer = answers.get("action") if isinstance(answers, dict) else None
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                raise ChoiceError("missing_response")
            selected = answer.get("choice")
            if not isinstance(selected, str) or not selected:
                raise ChoiceError("missing_response")
            if selected not in criteria:
                raise ChoiceError("unknown_candidate_id")
            probabilities = answer.get("probabilities")
            if probabilities is None:
                parsed_probabilities = None
            elif (not isinstance(probabilities, dict) or set(probabilities) != set(ids) or
                  any(type(p) not in (int, float) or p < 0 or p > 1 or not math.isfinite(p)
                      for p in probabilities.values()) or
                  not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3)):
                raise ChoiceError("invalid_probabilities")
            else:
                parsed_probabilities = {key: float(probabilities[key]) for key in ids}
            usage = value.get("usage")
            if usage is not None and (not isinstance(usage, dict) or len(usage) > 32 or
                                      any(not isinstance(k, str) or not k or len(k) > 128 or
                                          type(v) is not int or v < 0 for k, v in usage.items())):
                raise ChoiceError("invalid_usage")
            return ChoiceResult(selected, parsed_probabilities, self.model, latency_s, usage)
        except (TypeError, UnicodeError, json.JSONDecodeError):
            raise ChoiceError("missing_response") from None
