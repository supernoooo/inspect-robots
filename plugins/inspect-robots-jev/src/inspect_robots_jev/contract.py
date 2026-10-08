"""The only Observation decoder and safe hold action for Jev policies."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots import Action, ActionChunk, Observation
from inspect_robots_jev.yam_contract import ACTION_DIM, CAMERA_NAMES, STATE_KEY, action_space


class InputError(ValueError):
    """A bad policy input; ``joint_pos`` is set only when a safe hold is possible."""

    def __init__(
        self, code: str, message: str, joint_pos: npt.NDArray[np.float64] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.joint_pos = joint_pos


@dataclass(frozen=True, eq=False)
class DecodedInput:
    """Whitelisted, copied inputs for future vision, geometry, and policy modules."""

    images: Mapping[str, npt.NDArray[np.uint8]]
    joint_pos: npt.NDArray[np.float64]
    instruction: str
    image_times: Mapping[str, float]
    state_time: float


def camera_frame_ids(observation: Observation) -> dict[str, tuple[int, int]]:
    """Read YAM publication identifiers without treating capture time as an age gate."""
    raw = observation.extra.get("camera_frame_ids")
    if not isinstance(raw, Mapping) or set(raw) != set(CAMERA_NAMES):
        raise InputError("missing_frame_ids", "all three YAM camera frame IDs are required")
    result: dict[str, tuple[int, int]] = {}
    for name in CAMERA_NAMES:
        pair = raw[name]
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                any(type(value) is not int or value < 0 for value in pair)):
            raise InputError("invalid_frame_ids", f"invalid YAM frame ID for {name}")
        result[name] = (pair[0], pair[1])
    return result


def frames_advanced(current: Mapping[str, tuple[int, int]],
                    previous: Mapping[str, tuple[int, int]]) -> bool:
    """Require a new publication from the same open cycle on every camera."""
    return all(current[name][0] == previous[name][0] and
               current[name][1] > previous[name][1] for name in CAMERA_NAMES)


def _joint_pos(value: Any) -> npt.NDArray[np.float64]:
    try:
        raw = np.asarray(value)
        if not np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(
            raw.dtype, np.floating
        ):
            raise ValueError("joint_pos must be a numeric array")
        position = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError("invalid_joint_pos", "joint_pos must be a numeric array") from exc
    if position.shape != (ACTION_DIM,) or not np.isfinite(position).all():
        raise InputError("invalid_joint_pos", "joint_pos must be a finite 14-D vector")
    space = action_space()
    assert space.low is not None and space.high is not None
    if np.any(position < space.low) or np.any(position > space.high):
        raise InputError("invalid_joint_pos", "joint_pos exceeds YAM action limits")
    return position


def hold(joint_pos: npt.ArrayLike, reason: str) -> ActionChunk:
    """Emit one absolute target at the current pose with a JSON-safe reason."""
    position = _joint_pos(joint_pos)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("hold reason must be a nonempty string")
    return ActionChunk(
        actions=[Action(data=position)],
        meta={"kind": "hold", "reason": reason},
    )


def _timestamp(value: Any, name: str, *, now: float,
               max_image_age_s: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InputError("invalid_time", f"{name} must be a finite monotonic timestamp")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise InputError("invalid_time", f"{name} must be a finite monotonic timestamp")
    if max_image_age_s is not None and (
        timestamp > now or now - timestamp > max_image_age_s
    ):
        raise InputError("stale_time", f"{name} is outside the permitted age window")
    return timestamp


def decode_observation(
    observation: Observation,
    *,
    height: int,
    width: int,
    max_image_age_s: float | None,
    max_skew_s: float | None = None,
    now: float | None = None,
) -> DecodedInput:
    """Validate and copy only YAM RGB, joint, instruction, and capture times.

    InputError carries a validated joint state for recoverable faults so callers
    can hold. No access is made to Observation.extra or other state fields.
    """
    if height < 1 or width < 1:
        raise ValueError("height and width must be >= 1")
    if max_image_age_s is not None and (
        not math.isfinite(max_image_age_s) or max_image_age_s <= 0
    ):
        raise ValueError("max_image_age_s must be finite and > 0")
    if max_skew_s is None:
        max_skew_s = max_image_age_s
    if max_skew_s is not None and (not math.isfinite(max_skew_s) or max_skew_s <= 0):
        raise ValueError("max_skew_s must be finite and > 0")
    if STATE_KEY not in observation.state:
        raise InputError("missing_joint_pos", "observation is missing joint_pos")
    position = _joint_pos(observation.state[STATE_KEY])
    if now is None:
        now = time.monotonic()
    if not math.isfinite(now):
        raise ValueError("now must be finite")

    try:
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for camera in CAMERA_NAMES:
            if camera not in observation.images:
                raise InputError("missing_camera", f"observation is missing {camera}")
            image = np.asarray(observation.images[camera])
            if image.dtype != np.uint8 or image.shape != (height, width, 3):
                raise InputError("invalid_rgb", f"{camera} must be uint8 RGB ({height}, {width}, 3)")
            images[camera] = image.copy()

        instruction = observation.instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise InputError("invalid_instruction", "instruction must be a nonempty string")

        image_times: dict[str, float] = {}
        for camera in CAMERA_NAMES:
            if camera not in observation.image_times:
                raise InputError("missing_time", f"observation is missing image time for {camera}")
            image_times[camera] = _timestamp(
                observation.image_times[camera],
                f"image_times[{camera}]",
                now=now,
                max_image_age_s=max_image_age_s,
            )
        state_time = _timestamp(
            observation.state_time, "state_time", now=now, max_image_age_s=max_image_age_s
        )
        stamps = (state_time, *image_times.values())
        if max_skew_s is not None and max(stamps) - min(stamps) > max_skew_s:
            raise InputError("time_skew", "camera/state capture times exceed max_skew_s")
    except InputError as exc:
        exc.joint_pos = position
        raise

    return DecodedInput(
        images=images,
        joint_pos=position,
        instruction=instruction,
        image_times=image_times,
        state_time=state_time,
    )
