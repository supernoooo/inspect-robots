"""Joint-space motion synthesis for model-generated policy code.

The per-step ceiling reproduces ``DeltaLimitApprover``'s declared limit or
native-dtype ``0.05 * (high - low)`` arithmetic. The speed fraction may make
the ceiling tighter, but never looser, so actions within one returned chunk
are not silently rewritten by the CLI's default delta backstop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk

_BACKSTOP_STEP_FRAC = 0.05
_DEFAULT_SPEED_FRAC = 0.1
_RELATIVE_HEADROOM = 1e-6


@dataclass(frozen=True)
class ArmLayout:
    """Where one named arm's joints and gripper sit in a multi-arm action vector."""

    name: str
    joint_indices: tuple[int, ...]
    gripper_index: int


class MotionQueue:
    """Queue interpolated full-config targets around a turn-local state cursor.

    A single-arm queue takes ``gripper_index``; every other dimension is an
    arm joint and the helpers take no ``arm`` argument. A multi-arm queue
    takes ``arms`` instead, and every arm and gripper helper names its arm.
    """

    def __init__(
        self,
        action_space: Box,
        *,
        control_hz: float,
        max_speed_frac: float,
        gripper_index: int | None = None,
        gripper_open_is_high: bool = True,
        gripper_speed_frac: float | None = None,
        arms: Sequence[ArmLayout] | None = None,
    ) -> None:
        if len(action_space.shape) != 1:
            raise ValueError(f"motion needs a 1-D action box, got {action_space.shape}")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ValueError("max_speed_frac must be finite and > 0")
        if gripper_speed_frac is not None and (
            not np.isfinite(gripper_speed_frac) or gripper_speed_frac <= 0
        ):
            raise ValueError("gripper_speed_frac must be finite and > 0")
        layouts = _layouts(action_space.dim, gripper_index, arms)
        gripper_indices = [layout.gripper_index for layout in layouts.values()]
        low = action_space.low
        high = action_space.high
        if low is None or high is None:
            raise ValueError("motion needs finite low and high action bounds")
        if not bool(np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("motion needs finite low and high action bounds")

        low64 = np.asarray(low, dtype=np.float64)
        high64 = np.asarray(high, dtype=np.float64)
        with np.errstate(over="ignore"):
            native_range = np.asarray(high - low, dtype=np.float64)
            native_backstop = np.asarray(_BACKSTOP_STEP_FRAC * (high - low), dtype=np.float64)
        float64_range = high64 - low64
        if not bool(np.all(np.isfinite(native_range)) and np.all(np.isfinite(float64_range))):
            raise ValueError("action-space range (high - low) overflows; bounds are too large")
        movable = high64 > low64
        semantics = action_space.semantics
        declared = semantics.max_step if semantics is not None else None
        max_step = declared or (None,) * action_space.dim
        declared_mask = np.fromiter(
            (entry is not None for entry in max_step),
            dtype=np.bool_,
            count=action_space.dim,
        )
        effective_budget = native_backstop.copy()
        for index, entry in enumerate(max_step):
            if entry is not None:
                effective_budget[index] = entry
        # Interpolants snap to the float grid at the bounds' magnitude; if that
        # grid is coarse relative to the effective budget (offset boxes like
        # [1e16, 1e16 + 2]), emitted steps can exceed the approver's budget.
        spacing = np.spacing(np.maximum(np.abs(low64), np.abs(high64)))
        if bool(np.any(movable & (spacing > 5e-7 * effective_budget))):
            raise ValueError(
                "bounds are too coarse at this magnitude for speed-limited "
                "interpolation (float spacing exceeds the per-step budget)"
            )
        # The gripper may ramp at its own speed (open/close is a short, low-risk
        # travel); both speeds stay under the same per-step backstop.
        speeds = np.full(action_space.dim, max_speed_frac, dtype=np.float64)
        if gripper_speed_frac is not None:
            speeds[gripper_indices] = gripper_speed_frac
        step_fracs = np.minimum(speeds / control_hz, _BACKSTOP_STEP_FRAC)
        step_limits = np.minimum(step_fracs * float64_range, native_backstop)
        declared_scales = np.minimum(speeds / _DEFAULT_SPEED_FRAC, 1.0)
        step_limits[declared_mask] = (effective_budget * declared_scales)[declared_mask]
        if bool(np.any(movable & (step_limits <= 0))):
            raise ValueError("speed fraction underflows the per-step limit for a movable dimension")

        self.control_hz = control_hz
        self._dim = action_space.dim
        self._layouts = layouts
        self._low = low64
        self._high = high64
        self._step_limits = step_limits
        self._gripper_values: dict[int, tuple[float, float]] = {}
        for index in gripper_indices:
            high_value = float(high64[index])
            low_value = float(low64[index])
            self._gripper_values[index] = (
                (high_value, low_value) if gripper_open_is_high else (low_value, high_value)
            )
        # Single-arm compatibility: the open/closed values of the first gripper.
        self.gripper_open_value, self.gripper_closed_value = self._gripper_values[
            gripper_indices[0]
        ]
        self._cursor: npt.NDArray[np.float64] | None = None
        self._actions: list[Action] = []

    @property
    def arms(self) -> tuple[str, ...]:
        """Return the arm names of a multi-arm queue, or ``()`` for a single arm."""
        return tuple(name for name in self._layouts if name is not None)

    def joint_count(self, arm: str | None = None) -> int:
        """Return the joint count ``move_to_joints`` expects for ``arm``."""
        return len(self._layout(arm).joint_indices)

    def arm_joints(self, arm: str | None = None) -> npt.NDArray[np.float64]:
        """Return the queued endpoint's joints of ``arm``; raises before a turn is seeded."""
        return self._require_cursor()[list(self._layout(arm).joint_indices)].copy()

    @property
    def arm_dof(self) -> int:
        """Return the joint count expected by ``move_to_joints`` and Pyroki (single arm)."""
        return self.joint_count()

    @property
    def cursor(self) -> npt.NDArray[np.float64] | None:
        """Return a defensive copy of the current queued endpoint, if seeded."""
        return None if self._cursor is None else self._cursor.copy()

    @property
    def arm_cursor(self) -> npt.NDArray[np.float64]:
        """Return the queued endpoint's arm joints (single arm); raises before a seed."""
        return self.arm_joints()

    def reset(self) -> None:
        """Clear queued actions and the observation-derived cursor."""
        self._cursor = None
        self._actions.clear()

    def begin_turn(self, state: npt.NDArray[Any]) -> None:
        """Discard old queue state and seed this turn from observed proprioception."""
        cursor = np.asarray(state, dtype=np.float64).reshape(-1)
        if cursor.shape != (self._dim,):
            raise ValueError(
                f"proprioceptive reference has shape {cursor.shape}, expected ({self._dim},)"
            )
        if not bool(np.all(np.isfinite(cursor))):
            raise ValueError("proprioceptive reference contains a non-finite value")
        self._cursor = cursor.copy()
        self._actions.clear()

    def move_to_joints(self, joints: npt.NDArray[Any], arm: str | None = None) -> None:
        """Queue an arm target while preserving every gripper and any other arm."""
        target = self._require_cursor().copy()
        self._place_joints(target, joints, arm, "move_to_joints")
        self._queue_target(target)

    def move_arms(self, targets: Mapping[str, npt.NDArray[Any]]) -> None:
        """Queue one synchronized target for several named arms; unnamed arms hold."""
        if not self.arms:
            raise ValueError("move_arms needs a multi-arm embodiment; use move_to_joints(joints)")
        if not targets:
            raise ValueError(f"move_arms needs at least one arm target, e.g. {self.arms[0]}=...")
        target = self._require_cursor().copy()
        for arm, joints in targets.items():
            self._place_joints(target, joints, arm, "move_arms")
        self._queue_target(target)

    def open_gripper(self, arm: str | None = None) -> None:
        """Queue a speed-limited ramp to the configured open box bound."""
        index = self._layout(arm).gripper_index
        self._queue_gripper(index, self._gripper_values[index][0])

    def close_gripper(self, arm: str | None = None) -> None:
        """Queue a speed-limited ramp to the configured closed box bound."""
        index = self._layout(arm).gripper_index
        self._queue_gripper(index, self._gripper_values[index][1])

    def has_actions(self) -> bool:
        """Report whether model code queued any targets in the current turn."""
        return bool(self._actions)

    def take_chunk(
        self,
        *,
        inference_latency_s: float | None = None,
        code: str | None = None,
    ) -> ActionChunk:
        """Drain the non-empty queue into a chunk and annotate its first action."""
        if not self._actions:
            raise ValueError("motion queue is empty")
        actions = self._actions
        self._actions = []
        if code is not None:
            actions[0] = replace(actions[0], meta={**dict(actions[0].meta), "code": code})
        return ActionChunk(
            actions=actions,
            control_hz=self.control_hz,
            inference_latency_s=inference_latency_s,
        )

    def hold_chunk(
        self,
        state: npt.NDArray[Any],
        *,
        stop_reason: str,
        inference_latency_s: float | None = None,
    ) -> ActionChunk:
        """Return the required one-action stop chunk at the observed full config."""
        value = np.asarray(state, dtype=np.float64).reshape(-1)
        if value.shape != (self._dim,):
            raise ValueError(f"hold state has shape {value.shape}, expected ({self._dim},)")
        action = Action(
            data=value.copy(),
            meta={"request_stop": True, "stop_reason": stop_reason},
        )
        return ActionChunk(
            actions=[action],
            control_hz=self.control_hz,
            inference_latency_s=inference_latency_s,
        )

    def _require_cursor(self) -> npt.NDArray[np.float64]:
        if self._cursor is None:
            raise RuntimeError("motion cursor is unset; provide an observation before moving")
        return self._cursor

    def _layout(self, arm: str | None) -> ArmLayout:
        if not self.arms:
            if arm is not None:
                raise ValueError(
                    f"this embodiment has a single arm; call without arm (got arm={arm!r})"
                )
            return self._layouts[None]
        if arm is None:
            raise ValueError(f"name the arm: arm must be one of {list(self.arms)}")
        layout = self._layouts.get(arm)
        if layout is None:
            raise ValueError(f"unknown arm {arm!r}; arm must be one of {list(self.arms)}")
        return layout

    def _place_joints(
        self,
        target: npt.NDArray[np.float64],
        joints: npt.NDArray[Any],
        arm: str | None,
        helper: str,
    ) -> None:
        layout = self._layout(arm)
        values = np.asarray(joints, dtype=np.float64).reshape(-1)
        count = len(layout.joint_indices)
        if values.shape != (count,):
            owner = "arm" if arm is None else f"{arm!r} arm"
            raise ValueError(f"{helper} expects {count} {owner} joints, got shape {values.shape}")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"{helper} received a non-finite joint target")
        target[list(layout.joint_indices)] = values

    def _queue_gripper(self, index: int, value: float) -> None:
        target = self._require_cursor().copy()
        target[index] = value
        self._queue_target(target)

    def _queue_target(self, target: npt.NDArray[np.float64]) -> None:
        current = self._require_cursor()
        delta = target - current
        ratios: list[np.float64] = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index, distance in enumerate(np.abs(delta)):
                limit = self._step_limits[index]
                if distance == 0:
                    continue
                if limit <= 0:
                    raise ValueError(f"action dimension {index} is fixed and cannot move")
                ratios.append(np.divide(distance, limit))
            ratio = max(ratios, default=np.float64(0.0))
            headed_ratio = np.divide(ratio, 1.0 - _RELATIVE_HEADROOM)
        if not np.isfinite(headed_ratio):
            raise ValueError("motion distance is too large to interpolate safely")
        steps = max(1, math.ceil(float(headed_ratio)))
        for fraction in np.linspace(1.0 / steps, 1.0, steps):
            self._actions.append(Action(data=current + delta * fraction))
        self._actions[-1] = Action(data=target.copy())
        self._cursor = target.copy()


def _layouts(
    dim: int,
    gripper_index: int | None,
    arms: Sequence[ArmLayout] | None,
) -> dict[str | None, ArmLayout]:
    """Validate a single-arm gripper index or a complete, disjoint multi-arm layout."""
    if (gripper_index is None) == (arms is None):
        raise ValueError("give either gripper_index (single arm) or arms (multi-arm), not both")
    if gripper_index is not None:
        if not 0 <= gripper_index < dim:
            raise ValueError(f"gripper index {gripper_index} is outside {dim}-D box")
        joints = tuple(index for index in range(dim) if index != gripper_index)
        return {None: ArmLayout(name="", joint_indices=joints, gripper_index=gripper_index)}
    assert arms is not None
    if not arms:
        raise ValueError("arms must name at least one arm")
    layouts: dict[str | None, ArmLayout] = {}
    claimed: list[int] = []
    for layout in arms:
        if not isinstance(layout.name, str) or not layout.name:
            raise ValueError("every arm needs a non-empty name")
        if layout.name in layouts:
            raise ValueError(f"arm {layout.name!r} is declared twice")
        if not layout.joint_indices:
            raise ValueError(f"arm {layout.name!r} has no joints")
        layouts[layout.name] = layout
        claimed.extend((*layout.joint_indices, layout.gripper_index))
    if sorted(claimed) != list(range(dim)):
        raise ValueError(f"arm layouts must cover each of the {dim} action dimensions exactly once")
    return layouts
