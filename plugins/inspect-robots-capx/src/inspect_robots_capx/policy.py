"""Code-generation policy loop for the CaP-X integration.

Model output is executed in-process with the evaluator's privileges. This is
the policy under evaluation, not a security boundary. Robot actions still pass
through the rollout's approver chain, but untrusted models must run inside an
external container or equivalent isolation.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_agent import (
    ENV_MODEL,
    AssistantMessage,
    ChatClient,
    ResponsesClient,
    png_data_url,
    resolve_provider,
)
from inspect_robots_capx._motion import ArmLayout, MotionQueue
from inspect_robots_capx._sandbox import IK_BACKENDS, CodeSandbox, ExecutionResult
from inspect_robots_capx._servers import CapxServerClients

_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_WIRE_FORMATS = frozenset({"chat", "responses", "claude-code"})
_CLAUDE_CODE_EFFORT = frozenset({"low", "medium", "high", "max"})
_CLAUDE_CODE_TIMEOUT_S = 120.0


# Duplicated from inspect_robots_agent/policy.py; keep the effort contract in sync.
# The CLI -P parser coerces the literal `none` to Python None, so the constructor
# needs this marker to tell "operator asked for none" from "operator said nothing".
class _Unset:
    """Marker type distinguishing an omitted effort from an explicit one."""


_UNSET: Final = _Unset()
_EXECUTION_REPORT_CHAR_LIMIT = 16_000
_REPORT_TRUNCATION_MARKER = "[execution report truncated; tail follows]\n"

# Duplicated in inspect_robots_agent/policy.py; keep both limits in sync.
_PRIOR_LEARNINGS_TEXT_LIMIT = 32 * 1024

_FENCED_CODE = re.compile(r"^```(?:python)?[ \t]*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_FENCED_ANYWHERE = re.compile(r"```(?:python)?[ \t]*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_CONTROL_WORD = re.compile(r"(FINISH|GIVE_UP)[.!]?", re.IGNORECASE)

_HELPER_DOCS = """Helpers available in the persistent namespace:

segment(text: str, camera: str | None = None) -> list[dict]
    Segment the configured camera (or another named camera) with SAM3. Each
    result has a boolean `mask`, `box`, `score`, and `label`. Results include
    many low-confidence candidates: sort by `score` and treat a best score
    below about 0.3 as "not found". Depth and plan_grasp() are aligned to the
    configured camera only.
plan_grasp(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    Return (K, 4, 4) grasp poses in the camera frame and (K,) scores. K may be
    zero. Use `obs["{extrinsics_key}"] @ pose` to transform a pose into the
    robot-base frame. World and robot base are treated as the same frame.
solve_ik(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray
    Return the arm-joint vector expected by move_to_joints. {ik_note}
move_to_joints(joints: np.ndarray) -> None
    Queue a speed-limited arm move while holding the gripper.
open_gripper() / close_gripper() -> None
    Queue a speed-limited gripper ramp while holding the arm.

`obs` is the current turn's dict with `images`, `state`, and every
observation.extra entry. Depth (`{depth_key}`), intrinsics (`{intrinsics_key}`),
and extrinsics (`{extrinsics_key}`) may be provided by the embodiment as
arrays or zero-argument callables; `obs[...]` access and the helper functions
resolve callable values automatically.
"""

_MULTI_ARM_HELPER_DOCS = """Helpers available in the persistent namespace:

segment(text: str, camera: str | None = None) -> list[dict]
    Segment the configured camera (or another named camera) with SAM3. Each
    result has a boolean `mask`, `box`, `score`, and `label`. Results include
    many low-confidence candidates: sort by `score` and treat a best score
    below about 0.3 as "not found". Depth and plan_grasp() are aligned to the
    configured camera only.
plan_grasp(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    Return (K, 4, 4) grasp poses in the camera frame and (K,) scores. K may be
    zero. Use `obs["{extrinsics_key}"] @ pose` to transform a pose into the
    robot-base frame. World and robot base are treated as the same frame.

This embodiment has several arms: {arms}. Every arm and gripper helper names
its arm with `arm=`, one of those names.

solve_ik(position: np.ndarray, quaternion_wxyz: np.ndarray, arm: str) -> np.ndarray
    Return that arm's joint vector expected by move_to_joints. {ik_note}
move_to_joints(joints: np.ndarray, arm: str) -> None
    Queue a speed-limited move of one arm while every other arm and every
    gripper holds.
move_arms(**joints_by_arm: np.ndarray) -> None
    Queue one synchronized move of several arms, e.g. move_arms({arm_example}),
    that starts and ends together; arms you do not name hold.
open_gripper(arm: str) / close_gripper(arm: str) -> None
    Queue a speed-limited ramp of that arm's gripper while everything else holds.

Queued moves run one after another in call order; use move_arms when arms
must move at the same time.

`obs` is the current turn's dict with `images`, `state`, and every
observation.extra entry. Depth (`{depth_key}`), intrinsics (`{intrinsics_key}`),
and extrinsics (`{extrinsics_key}`) may be provided by the embodiment as
arrays or zero-argument callables; `obs[...]` access and the helper functions
resolve callable values automatically.
"""

_IK_NOTES = {
    "pyroki": "Solved by the Pyroki server in its URDF base frame.",
    "embodiment": (
        "Solved by the embodiment in the robot-base frame, seeded from the queued "
        "joint endpoint; see the embodiment notes for its frame, units, and limits."
    ),
}

_SYSTEM_TEMPLATE = """You generate Python code to directly solve a robot manipulation task.
Write raw Python with no Markdown fences. Helpers are already bound. Import
numpy explicitly when you need it. Variables persist across turns in this
trial. Perception within a turn uses the initial observation; queued motions
execute after your code returns, and you verify their effect next turn.

After each execution you receive the code, stdout, and stderr. Respond with
FINISH when the goal is complete, GIVE_UP when it cannot be completed, or
REGENERATE followed by corrected Python. You may also return raw corrected
Python directly. You have {budget} LLM calls for the whole trial.

Embodiment: {name}
Action profile: {action_summary}

{helper_docs}"""


def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an image-free deep copy suitable for persistence and visualization."""
    sanitized = copy.deepcopy(messages)
    for message in sanitized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                content[index] = {
                    "type": "text",
                    "text": "[image omitted: streamed camera frame]",
                }
    return sanitized


_ARM_GRIPPER_SUFFIX = "_gripper"


def _arm_layouts(labels: tuple[str, ...] | None, prefix: str) -> list[ArmLayout]:
    """Return the multi-arm layout ``dim_labels`` declare, or ``[]`` for one arm.

    Labels declare several arms with one ``<arm>_gripper`` entry per arm (two
    or more) and no bare ``gripper``; every other label must start with
    exactly one ``<arm>_`` prefix. Arms keep the order of their grippers.
    """
    if labels is None or "gripper" in labels:
        return []
    grippers = [
        (index, label.removesuffix(_ARM_GRIPPER_SUFFIX))
        for index, label in enumerate(labels)
        if label.endswith(_ARM_GRIPPER_SUFFIX) and label != _ARM_GRIPPER_SUFFIX
    ]
    if len(grippers) < 2:
        return []
    names = [name for _, name in grippers]
    if len(set(names)) != len(names):
        raise ValueError(f"{prefix} one '<arm>_gripper' dim_labels entry per arm; got {names}")
    joints: dict[str, list[int]] = {name: [] for name in names}
    gripper_indices = {index for index, _ in grippers}
    for index, label in enumerate(labels):
        if index in gripper_indices:
            continue
        owners = [name for name in names if label.startswith(f"{name}_")]
        if len(owners) != 1:
            raise ValueError(
                f"{prefix} every multi-arm dim_labels entry to start with exactly one arm "
                f"prefix {[f'{name}_' for name in names]}; {label!r} matches {owners}"
            )
        joints[owners[0]].append(index)
    for name, indices in joints.items():
        if not indices:
            raise ValueError(f"{prefix} at least one joint for arm {name!r}")
    return [
        ArmLayout(name=name, joint_indices=tuple(joints[name]), gripper_index=index)
        for index, name in grippers
    ]


@dataclass(frozen=True)
class CapxPolicyConfig(PolicyConfig):
    """Inference, server, motion, and transcript settings persisted in eval logs."""

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    wire: str = "chat"
    #: Resolved effort level; ``None`` means the field is omitted and the
    #: provider default applies. The recorded value is the normalized one, so
    #: an operator ``none`` reads back as ``"none"`` rather than a bare omission.
    effort: str | None = None
    sam3_url: str = "http://127.0.0.1:8114"
    graspnet_url: str = "http://127.0.0.1:8115"
    pyroki_url: str = "http://127.0.0.1:8116"
    camera: str | None = None
    depth_key: str = "depth"
    intrinsics_key: str = "intrinsics"
    extrinsics_key: str = "extrinsics"
    #: ``pyroki`` (CaP-X server) or ``embodiment`` (``Observation.extra[ik_key]``).
    ik: str = "pyroki"
    ik_key: str = "ik_solver"
    #: Resolved Claude Code executable when ``wire == "claude-code"``.
    claude_command: str | None = None
    claude_timeout_s: float | None = None
    max_llm_calls: int = 100
    max_code_failures: int = 3
    #: Keep camera images from only the newest N observations (None keeps all).
    image_horizon: int | None = None
    max_speed_frac: float = 0.1
    #: Gripper ramp speed (range fraction per second); None reuses max_speed_frac.
    gripper_speed_frac: float | None = None
    request_timeout_s: float = 120.0
    gripper_open_is_high: bool = True
    transcript_echo: bool = False
    #: Resolved absolute path to the injected prior-learnings file.
    prior_learnings: str | None = None
    #: SHA-256 hexdigest of the injected prior-learnings text.
    prior_learnings_sha256: str | None = None


class CapxPolicy(PolicyBase):
    """Runs a persistent CaP-X-style codegen conversation over a bound joint arm.

    ``prior_learnings`` optionally loads a UTF-8 notes file once at construction
    and appends its text to every trial's system prompt.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        wire: str = "chat",
        max_llm_calls: int = 100,
        max_code_failures: int = 3,
        temperature: float | None = None,
        effort: str | None | _Unset = _UNSET,
        sam3_url: str = "http://127.0.0.1:8114",
        graspnet_url: str = "http://127.0.0.1:8115",
        pyroki_url: str = "http://127.0.0.1:8116",
        camera: str | None = None,
        depth_key: str = "depth",
        intrinsics_key: str = "intrinsics",
        extrinsics_key: str = "extrinsics",
        ik: str = "pyroki",
        ik_key: str = "ik_solver",
        max_speed_frac: float = 0.1,
        gripper_speed_frac: float | None = None,
        request_timeout_s: float = 120.0,
        gripper_open_is_high: bool = True,
        transcript_echo: bool = False,
        prior_learnings: str | None = None,
        image_horizon: int | None = None,
        claude_command: str | None = None,
        claude_timeout_s: float = _CLAUDE_CODE_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        prior_learnings_path: str | None = None
        prior_learnings_text: str | None = None
        prior_learnings_sha256: str | None = None
        if prior_learnings is not None:
            if not isinstance(prior_learnings, str) or not prior_learnings:
                raise ConfigError(
                    "prior_learnings must be a non-empty filesystem path string, "
                    f"got {prior_learnings!r}.\n"
                    "fix: the -P parser coerces unquoted values; pass "
                    "-P 'prior_learnings=\"path/to/learnings.md\"'"
                )
            path = Path(prior_learnings)
            try:
                prior_learnings_text = path.read_text(encoding="utf-8")
                prior_learnings_path = str(path.resolve())
            except (OSError, UnicodeDecodeError) as exc:
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} could not be read as UTF-8 "
                    f"({exc}).\n"
                    "fix: pass the path to a readable UTF-8 learnings file"
                ) from exc
            if not prior_learnings_text.strip():
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} is empty or whitespace-only.\n"
                    "fix: add concise notes to the file or omit -P prior_learnings="
                )
            if len(prior_learnings_text) > _PRIOR_LEARNINGS_TEXT_LIMIT:
                raise ConfigError(
                    f"prior_learnings file {prior_learnings!r} has "
                    f"{len(prior_learnings_text)} characters; the limit is "
                    f"{_PRIOR_LEARNINGS_TEXT_LIMIT}.\n"
                    "fix: summarize it first and pass the path to the shorter learnings file"
                )
            prior_learnings_sha256 = hashlib.sha256(
                prior_learnings_text.encode("utf-8")
            ).hexdigest()

        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be >= 1")
        if max_code_failures < 1:
            raise ValueError("max_code_failures must be >= 1")
        if image_horizon is not None and (
            isinstance(image_horizon, bool)
            or not isinstance(image_horizon, int)
            or image_horizon < 1
        ):
            raise ConfigError(
                f"image_horizon must be a positive integer, got {image_horizon!r}.\n"
                "fix: omit -P image_horizon to keep every frame, or pass e.g. -P image_horizon=2"
            )
        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ValueError("max_speed_frac must be finite and > 0")
        if gripper_speed_frac is not None and (
            isinstance(gripper_speed_frac, bool)
            or not isinstance(gripper_speed_frac, (int, float))
            or not np.isfinite(gripper_speed_frac)
            or gripper_speed_frac <= 0
        ):
            raise ConfigError(
                f"gripper_speed_frac must be a positive number, got {gripper_speed_frac!r}.\n"
                "fix: omit it to reuse max_speed_frac, or pass e.g. -P gripper_speed_frac=0.5"
            )
        if not np.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and > 0")
        # Unset leaves effort omitted so the provider default applies. A supplied
        # value normalizes the CLI-coerced None to the level "none" and is then
        # validated; ConfigError (not ValueError) so the CLI renders a guided
        # message instead of a traceback (#168).
        resolved_effort: str | None = None
        if not isinstance(effort, _Unset):
            resolved_effort = "none" if effort is None else effort
            if resolved_effort not in _EFFORT_LEVELS:
                raise ConfigError(
                    f"effort must be one of {sorted(_EFFORT_LEVELS)}, got {resolved_effort!r}.\n"
                    "fix: omit -P effort= to use the provider default"
                )
        if wire not in _WIRE_FORMATS:
            raise ValueError(f"wire must be one of {sorted(_WIRE_FORMATS)}, got {wire!r}")
        if ik not in IK_BACKENDS:
            raise ConfigError(
                f"ik must be one of {sorted(IK_BACKENDS)}, got {ik!r}.\n"
                "fix: use -P ik=embodiment only with an embodiment that publishes "
                "an IK solver in Observation.extra"
            )
        if not isinstance(ik_key, str) or not ik_key:
            raise ConfigError(f"ik_key must be a non-empty string, got {ik_key!r}")

        environ = dict(os.environ) if env is None else env
        self._client: Any
        if wire == "claude-code":
            # Subscription login through the official CLI; the transport (if
            # any) still serves the perception servers.
            if base_url is not None or api_key_env is not None:
                raise ConfigError(
                    "wire='claude-code' uses the official CLI login, not an API endpoint.\n"
                    "fix: drop -P base_url and -P api_key_env"
                )
            if temperature is not None:
                raise ConfigError("temperature is not supported on wire='claude-code'")
            if resolved_effort not in {None, *_CLAUDE_CODE_EFFORT}:
                raise ConfigError("effort on wire='claude-code' must be low, medium, high, or max")
            cli_model = (model or environ.get(ENV_MODEL) or "").removeprefix("anthropic/")
            if not cli_model:
                raise ConfigError(
                    "wire='claude-code' needs a model.\n"
                    "fix: pass -P model=anthropic/<model id> (or a CLI alias such as opus)"
                )
            try:
                # Optional: only some inspect-robots-agent builds ship the
                # subscription-login CLI client.
                from inspect_robots_agent._claude_code import ClaudeCodeClient
            except ImportError as exc:
                raise ConfigError(
                    "wire='claude-code' needs an inspect-robots-agent build that provides "
                    "the Claude Code CLI client (inspect_robots_agent._claude_code).\n"
                    "fix: use an API key instead, e.g. -P model=anthropic/<model id> with "
                    "ANTHROPIC_API_KEY set"
                ) from exc
            self._client = ClaudeCodeClient(
                cli_model, command=claude_command, timeout_s=claude_timeout_s, env=environ
            )
            resolved_model: str | None = cli_model
            resolved_base_url: str | None = "claude-code://local"
        else:
            if claude_command is not None or claude_timeout_s != _CLAUDE_CODE_TIMEOUT_S:
                raise ConfigError("claude_command/claude_timeout_s require -P wire=claude-code")
            provider = resolve_provider(
                model=model or environ.get(ENV_MODEL),
                base_url=base_url,
                api_key_env=api_key_env,
                env=environ,
            )
            if wire == "responses":
                self._client = ResponsesClient(provider, transport=transport)
            else:
                self._client = ChatClient(provider, transport=transport)
            resolved_model, resolved_base_url = provider.model, provider.base_url
        self._servers = CapxServerClients(
            sam3_url=sam3_url,
            graspnet_url=graspnet_url,
            pyroki_url=pyroki_url,
            request_timeout_s=request_timeout_s,
            transport=transport,
        )
        self._camera_config = camera
        self._depth_key = depth_key
        self._intrinsics_key = intrinsics_key
        self._extrinsics_key = extrinsics_key
        self._ik = ik
        self._ik_key = ik_key
        self._max_llm_calls = max_llm_calls
        self._max_code_failures = max_code_failures
        self._image_horizon = image_horizon
        self._temperature = temperature
        self._effort = resolved_effort
        self._max_speed_frac = max_speed_frac
        self._gripper_speed_frac = gripper_speed_frac
        self._gripper_open_is_high = gripper_open_is_high
        self._transcript_echo = transcript_echo
        self._prior_learnings_text = prior_learnings_text
        self.config = CapxPolicyConfig(
            temperature=temperature,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key_env=api_key_env,
            wire=wire,
            effort=resolved_effort,
            sam3_url=sam3_url,
            graspnet_url=graspnet_url,
            pyroki_url=pyroki_url,
            camera=camera,
            depth_key=depth_key,
            intrinsics_key=intrinsics_key,
            extrinsics_key=extrinsics_key,
            ik=ik,
            ik_key=ik_key,
            claude_command=self._client.command if wire == "claude-code" else None,
            claude_timeout_s=claude_timeout_s if wire == "claude-code" else None,
            max_llm_calls=max_llm_calls,
            max_code_failures=max_code_failures,
            image_horizon=image_horizon,
            max_speed_frac=max_speed_frac,
            gripper_speed_frac=gripper_speed_frac,
            request_timeout_s=request_timeout_s,
            gripper_open_is_high=gripper_open_is_high,
            transcript_echo=transcript_echo,
            prior_learnings=prior_learnings_path,
            prior_learnings_sha256=prior_learnings_sha256,
        )
        self.info = PolicyInfo(name="capx", action_space=Box(shape=(1,)))
        self._motion: MotionQueue | None = None
        self._sandbox: CodeSandbox | None = None
        self._state_key: str | None = None
        self._state_labels: tuple[str, tuple[str, ...]] | None = None
        self._embodiment_name = "(unbound)"
        self._embodiment_docs: str | None = None
        self._arms: tuple[str, ...] = ()
        self._action_summary = "unbound; call bind() before act()"
        self._messages: list[dict[str, Any]] = []
        self._delta_cursor = 0
        self._calls_used = 0
        self._consecutive_failures = 0
        self._closed = False
        atexit.register(self.close)

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt an embodiment that satisfies the plan-0021 single-arm profile.

        Several arms are accepted too when ``dim_labels`` name one
        ``<arm>_gripper`` per arm and every other label starts with exactly
        one ``<arm>_`` prefix (for example ``left_j0 .. left_gripper,
        right_j0 .. right_gripper``). That profile needs ``ik="embodiment"``.
        """
        space = embodiment_info.action_space
        prefix = "plan 0021 CaP-X v1 profile requires"
        if len(space.shape) != 1 or space.dim < 2:
            raise ValueError(f"{prefix} a 1-D Box action space with at least 2 dimensions")
        semantics = space.semantics
        if semantics is None or semantics.control_mode != "joint_pos":
            mode = None if semantics is None else semantics.control_mode
            raise ValueError(f"{prefix} control_mode='joint_pos', got {mode!r}")
        if semantics.gripper == "none":
            raise ValueError(f"{prefix} ActionSemantics.gripper to be continuous or binary")
        labels = semantics.dim_labels
        arms = _arm_layouts(labels, prefix)
        gripper_index: int | None = None
        if arms:
            if self._ik != "embodiment":
                raise ValueError(
                    f"{prefix} ik='embodiment' for a multi-arm embodiment "
                    f"({', '.join(arm.name for arm in arms)}); Pyroki solves one arm"
                )
        elif labels is None:
            gripper_index = space.dim - 1
        else:
            matches = [index for index, label in enumerate(labels) if label == "gripper"]
            if len(matches) != 1:
                raise ValueError(
                    f"{prefix} exactly one dim_labels entry named 'gripper'; found {matches}"
                )
            gripper_index = matches[0]
        if embodiment_info.control_hz is None:
            raise ValueError(f"{prefix} a declared finite control_hz > 0")

        state_spec = embodiment_info.observation_space.state
        if state_spec is None:
            raise ValueError(f"{prefix} an observation StateSpec for full joint state")
        matching = [field.key for field in state_spec.fields if field.shape == (space.dim,)]
        if len(matching) != 1:
            raise ValueError(
                f"{prefix} exactly one state field with shape ({space.dim},); "
                f"found {matching or 'none'}"
            )
        state_key = matching[0]

        camera_names = embodiment_info.observation_space.camera_names
        if self._camera_config is None:
            if len(camera_names) != 1:
                raise ValueError(
                    f"{prefix} one camera when camera=None; declared {sorted(camera_names)}"
                )
            camera = next(iter(camera_names))
        else:
            camera = self._camera_config
            if camera not in camera_names:
                raise ValueError(
                    f"{prefix} the configured camera to be declared; {camera!r} is absent, "
                    "available: "
                    f"{sorted(camera_names)}"
                )

        try:
            motion = MotionQueue(
                space,
                control_hz=embodiment_info.control_hz,
                max_speed_frac=self._max_speed_frac,
                gripper_index=gripper_index,
                gripper_open_is_high=self._gripper_open_is_high,
                gripper_speed_frac=self._gripper_speed_frac,
                arms=arms or None,
            )
        except ValueError as exc:
            raise ValueError(f"{prefix} valid bounded motion arithmetic: {exc}") from exc
        self._motion = motion
        self._sandbox = CodeSandbox(
            servers=self._servers,
            motion=motion,
            camera=camera,
            state_key=state_key,
            depth_key=self._depth_key,
            intrinsics_key=self._intrinsics_key,
            extrinsics_key=self._extrinsics_key,
            ik=self._ik,
            ik_key=self._ik_key,
        )
        self._state_key = state_key
        self._state_labels = (state_key, labels) if labels is not None else None
        self._embodiment_name = embodiment_info.name
        self._embodiment_docs = getattr(embodiment_info, "docs", None)
        self._arms = tuple(arm.name for arm in arms)
        polarity = (
            "high=open, low=closed" if self._gripper_open_is_high else "low=open, high=closed"
        )
        if arms:
            assert labels is not None
            described = "; ".join(
                f"arm {arm.name!r}: {len(arm.joint_indices)} joints "
                f"{[labels[index] for index in arm.joint_indices]}, gripper dim "
                f"{arm.gripper_index}"
                for arm in arms
            )
            self._action_summary = (
                f"joint_pos with {len(arms)} arms ({described}); grippers {polarity}, "
                f"control_hz={embodiment_info.control_hz:g}"
            )
        else:
            arm_labels = [
                labels[index] if labels is not None else str(index)
                for index in range(space.dim)
                if index != gripper_index
            ]
            self._action_summary = (
                f"joint_pos with {motion.arm_dof} arm joints {arm_labels}, gripper dim "
                f"{gripper_index} ({polarity}), control_hz={embodiment_info.control_hz:g}"
            )
        self.info = PolicyInfo(
            name="capx",
            action_space=space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        """Start a trial with fresh code, transcript, failure, and IK state."""
        keys = {
            "depth_key": self._depth_key,
            "intrinsics_key": self._intrinsics_key,
            "extrinsics_key": self._extrinsics_key,
            "ik_note": _IK_NOTES[self._ik],
        }
        if self._arms:
            helper_docs = _MULTI_ARM_HELPER_DOCS.format(
                arms=", ".join(repr(arm) for arm in self._arms),
                arm_example=", ".join(f"{arm}=..." for arm in self._arms),
                **keys,
            )
        else:
            helper_docs = _HELPER_DOCS.format(**keys)
        system = _SYSTEM_TEMPLATE.format(
            budget=self._max_llm_calls,
            name=self._embodiment_name,
            action_summary=self._action_summary,
            helper_docs=helper_docs,
        )
        if self._embodiment_docs is not None and self._embodiment_docs.strip():
            system += "\n\nEmbodiment notes:\n" + self._embodiment_docs.strip()
        if self._prior_learnings_text is not None:
            system = (
                system
                + "\n\nNotes from a previous attempt at tasks like this one. They may "
                + "be wrong or stale; the current observation always wins:\n"
                + self._prior_learnings_text
            )
        self._messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {scene.instruction}"},
        ]
        self._delta_cursor = 0
        self._calls_used = 0
        self._consecutive_failures = 0
        if self._sandbox is not None:
            self._sandbox.reset()
        self._servers.pyroki.reset()
        self._echo(f"[capx] goal: {scene.instruction}")

    def transcript(self) -> list[dict[str, Any]] | None:
        """Return an image-free deep copy of the current trial conversation."""
        return _sanitize(self._messages) if self._messages else None

    def transcript_delta(self) -> list[dict[str, Any]] | None:
        """Return only sanitized messages appended since the previous live-stream read."""
        new = self._messages[self._delta_cursor :]
        self._delta_cursor = len(self._messages)
        return _sanitize(new) if new else None

    def act(self, observation: Observation) -> ActionChunk:
        """Run codegen turns until code queues motion or the model requests a stop."""
        motion = self._motion
        sandbox = self._sandbox
        state_key = self._state_key
        if motion is None or sandbox is None or state_key is None:
            raise RuntimeError(
                "CapxPolicy.act() before bind(); run it through eval() or call "
                "policy.bind(embodiment.info) first"
            )
        sandbox.set_observation(observation)
        state = observation.state[state_key]
        self._messages.append(
            {
                "role": "user",
                "content": _observation_content(observation, self._state_labels),
            }
        )
        if self._image_horizon is not None:
            _evict_images(self._messages, self._image_horizon)
        llm_latency = 0.0
        while True:
            if self._calls_used >= self._max_llm_calls:
                self._echo("[capx] -- LLM call budget exhausted; forcing GIVE_UP")
                return motion.hold_chunk(
                    state,
                    stop_reason="GIVE_UP",
                    inference_latency_s=llm_latency,
                )
            started = time.monotonic()
            message = self._client.complete(
                self._messages,
                [],
                temperature=self._temperature,
                reasoning_effort=self._effort,
            )
            llm_latency += time.monotonic() - started
            self._calls_used += 1
            self._messages.append(message.raw())
            reply = (message.content or "").strip()
            if reply:
                self._echo(f"[capx] << {reply}")
            # A stop word on the first line ends the trial even when the model
            # appends a summary; that prose is not code and must not execute.
            control = _CONTROL_WORD.fullmatch(reply.partition("\n")[0].strip())
            if control is not None:
                return motion.hold_chunk(
                    state,
                    stop_reason=control.group(1).upper(),
                    inference_latency_s=llm_latency,
                )

            code = _extract_code(message)
            result = sandbox.execute(code)
            self._messages.append({"role": "user", "content": _execution_report(code, result)})
            self._echo_execution(result)
            if result.raised:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

            if motion.has_actions():
                return motion.take_chunk(inference_latency_s=llm_latency, code=code)
            if result.raised and self._consecutive_failures >= self._max_code_failures:
                raise RuntimeError(
                    "model code failed in "
                    f"{self._consecutive_failures} consecutive turns; see transcript stderr"
                )

    def close(self) -> None:
        """Idempotently close LLM and shared CaP-X HTTP connection pools."""
        if self._closed:
            return
        self._closed = True
        self._client.close()
        self._servers.close()
        with contextlib.suppress(Exception):  # defensive during interpreter teardown
            atexit.unregister(self.close)

    def _echo(self, text: str) -> None:
        if self._transcript_echo:
            print(text, file=sys.stderr, flush=True)

    def _echo_execution(self, result: ExecutionResult) -> None:
        if not self._transcript_echo:
            return
        if result.stdout:
            self._echo(f"[capx] stdout:\n{result.stdout.rstrip()}")
        if result.stderr:
            self._echo(f"[capx] stderr:\n{result.stderr.rstrip()}")


_EVICTED_IMAGE = {"type": "text", "text": "[image omitted: older camera frame]"}


def _evict_images(messages: list[dict[str, Any]], horizon: int) -> None:
    """Replace camera images outside the newest ``horizon`` image-bearing turns.

    Older frames are dropped from the history for good, which bounds request
    size (the Claude Code CLI rejects inputs over 10 MB) at the cost of the
    model no longer seeing them.
    """
    kept = 0
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list) or not any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            continue
        kept += 1
        if kept > horizon:
            message["content"] = [
                dict(_EVICTED_IMAGE) if part.get("type") == "image_url" else part
                for part in content
            ]


def _extract_code(message: AssistantMessage) -> str:
    """Normalize raw, fenced, prose-wrapped, or ``REGENERATE``-marked code.

    Text up to the last line that is exactly ``REGENERATE`` is the model's
    commentary and is dropped, so "explanation, REGENERATE, code" replies run
    only the code. A reply that is exactly one fenced block (or a bare
    snippet) is used as is; otherwise the first fenced block wins, so
    surrounding prose does not burn a failure turn on a ``SyntaxError``.
    """
    code = (message.content or "").strip()
    lines = code.splitlines()
    markers = [index for index, line in enumerate(lines) if line.strip() == "REGENERATE"]
    if markers:
        code = "\n".join(lines[markers[-1] + 1 :]).strip()
    fenced = _FENCED_CODE.fullmatch(code)
    if fenced is not None:
        return fenced.group(1)
    embedded = _FENCED_ANYWHERE.search(code)
    if embedded is not None:
        return embedded.group(1)
    return _strip_leading_prose(code)


def _compiles(code: str) -> bool:
    try:
        compile(code, "<capx>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _strip_leading_prose(code: str) -> str:
    """Drop leading commentary lines when only that prose stops the reply compiling.

    Some models open with a sentence before unfenced code (and some wires glue
    it onto the first code line). Only a reply that fails to compile is
    touched, and only by removing whole leading lines; if no suffix compiles,
    the reply is returned unchanged so the model sees its own SyntaxError.
    """
    if _compiles(code):
        return code
    lines = code.splitlines()
    for start in range(1, len(lines)):
        candidate = "\n".join(lines[start:]).strip()
        if candidate and _compiles(candidate):
            return candidate
    return code


def _execution_report(code: str, result: ExecutionResult) -> str:
    """Build a bounded tail-first CaP-X execution feedback message."""
    report = (
        "Executed code:\n"
        f"```python\n{code}\n```\n"
        "stdout:\n"
        f"```text\n{result.stdout}\n```\n"
        "stderr:\n"
        f"```text\n{result.stderr}\n```\n"
        "Respond with FINISH, GIVE_UP, or REGENERATE followed by Python."
    )
    if len(report) <= _EXECUTION_REPORT_CHAR_LIMIT:
        return report
    tail_size = _EXECUTION_REPORT_CHAR_LIMIT - len(_REPORT_TRUNCATION_MARKER)
    return _REPORT_TRUNCATION_MARKER + report[-tail_size:]


def _state_lines(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None,
) -> list[str]:
    """Render observation state with action labels when the bound vector aligns."""
    lines: list[str] = []
    for key, value in observation.state.items():
        array = np.asarray(value, dtype=np.float64)
        rounded = np.round(array, 4).tolist()
        if state_labels is not None and key == state_labels[0]:
            labels = state_labels[1]
            if array.shape == (len(labels),):
                labeled = " ".join(
                    f"{label}={item}" for label, item in zip(labels, rounded, strict=True)
                )
                lines.append(f"state[{key}]: {labeled}")
                continue
        lines.append(f"state[{key}]: {rounded}")
    return lines


def _observation_content(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None,
) -> list[dict[str, Any]]:
    """Encode fresh state text and camera PNGs without repeating execution feedback."""
    lines = ["Current observation."]
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    lines.extend(_state_lines(observation, state_labels))
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    for name, image in observation.images.items():
        parts.append({"type": "text", "text": f"camera {name!r}:"})
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
    return parts


def capx_policy(**kwargs: Any) -> CapxPolicy:
    """Build the registry policy while forwarding CLI ``-P`` keyword arguments."""
    return CapxPolicy(**kwargs)
