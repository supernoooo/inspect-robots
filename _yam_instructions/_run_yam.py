"""Rig configuration for the CaP-X pipeline on the official robocurve/inspect-robots-yam plugin.

The ``yam_capx_configured`` CLI entry wraps the upstream ``yam_arms``
embodiment: it drives one arm (``arm=left`` or ``arm=right``) through the
single-arm ``capx`` profile, holds the other at its start pose, and publishes
depth, intrinsics, tool state, and an in-process IK solver. Hardware connects
only when an evaluation starts, never during doctor/preflight.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_yam.config import DEFAULT_CAMERAS
from inspect_robots_yam.embodiment import YAMEmbodiment

from inspect_robots.errors import ConfigError
from inspect_robots.registry import embodiment

_DEPTH_SLOTS = {name: name.removesuffix("_cam") for name in DEFAULT_CAMERAS}


@embodiment("yam_capx_configured")
def configured_yam_capx(**overrides: Any) -> Any:
    """One YAM arm with aligned depth, intrinsics, extrinsics, and IK for ``capx``.

    Besides every ``YamConfig`` field, accepts ``arm`` (``left``/``right``),
    ``depth_camera`` (a ``*_depth_serial`` slot; default ``top_cam``),
    ``camera_extrinsics`` (4x4 camera-to-base JSON/.npy in the driven arm's
    base frame), ``ik_position_tolerance_m``, ``ik_orientation_weight``, and
    ``model_path`` (an MJCF overriding i2rt's bundled arm+gripper model).
    """
    from _yam_capx import (
        DEFAULT_IK_ORIENTATION_WEIGHT,
        DEFAULT_IK_POSITION_TOLERANCE_M,
        SIDES,
        YamCapxEmbodiment,
        YamKinematics,
        arm_slice,
        load_extrinsics,
    )

    side = overrides.pop("arm", "left")
    if side not in SIDES:
        raise ConfigError(f"arm must be one of {list(SIDES)}, got {side!r}.")
    depth_camera = overrides.pop("depth_camera", "top_cam")
    if depth_camera not in _DEPTH_SLOTS:
        raise ConfigError(
            f"depth_camera must be one of {list(DEFAULT_CAMERAS)}, got {depth_camera!r}."
        )
    extrinsics_path = overrides.pop("camera_extrinsics", None)
    ik_position_tolerance_m = overrides.pop(
        "ik_position_tolerance_m", DEFAULT_IK_POSITION_TOLERANCE_M
    )
    ik_orientation_weight = overrides.pop("ik_orientation_weight", DEFAULT_IK_ORIENTATION_WEIGHT)
    model_path = overrides.pop("model_path", None)
    if overrides.get("control_interface", "joints") != "joints":
        raise ConfigError(
            "yam_capx_configured drives joints directly; remove control_interface "
            "(the model solves IK through solve_ik)."
        )
    if overrides.get("joints_are_delta", False):
        raise ConfigError(
            "yam_capx_configured needs absolute joint targets (joints_are_delta=false)."
        )
    try:
        extrinsics = None if extrinsics_path in (None, "") else load_extrinsics(extrinsics_path)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Invalid camera_extrinsics: {exc}") from exc
    try:
        arm = YAMEmbodiment(**overrides)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid YAM configuration: {exc}") from exc
    config = arm._cfg
    if getattr(config, f"{_DEPTH_SLOTS[depth_camera]}_depth_serial") is None:
        raise ConfigError(
            f"depth_camera {depth_camera!r} needs {_DEPTH_SLOTS[depth_camera]}_depth_serial "
            "(a RealSense serving colour and depth); a *_cam_device slot has no depth."
        )
    active = arm_slice(side)
    try:
        kinematics = YamKinematics(
            config.gripper_type,
            joint_low=config.low[active],
            joint_high=config.high[active],
            model_path=model_path,
        )
        return YamCapxEmbodiment(
            arm,
            kinematics,
            side=side,
            depth_camera=depth_camera,
            extrinsics=extrinsics,
            ik_position_tolerance_m=ik_position_tolerance_m,
            ik_orientation_weight=ik_orientation_weight,
        )
    except ValueError as exc:
        raise ConfigError(f"Invalid YAM CaP-X configuration: {exc}") from exc


# The setup wizard interviews the same CAN and camera slots as yam_arms; the
# RealSense depth serials are outside its scope and come from the config file.
configured_yam_capx.DEVICE_SLOTS = YAMEmbodiment.DEVICE_SLOTS  # type: ignore[attr-defined]
configured_yam_capx.OPTION_SLOTS = tuple(  # type: ignore[attr-defined]
    slot for slot in YAMEmbodiment.OPTION_SLOTS if slot.arg != "eef_orientation"
)
configured_yam_capx.NUMBER_SLOTS = YAMEmbodiment.NUMBER_SLOTS  # type: ignore[attr-defined]
configured_yam_capx.RUNTIME_REQUIREMENTS = {  # type: ignore[attr-defined]
    **YAMEmbodiment.RUNTIME_REQUIREMENTS,
    "mink": "uv pip install mink",
    "mujoco": "uv pip install 'mujoco>=3.3.1'",
}


if __name__ == "__main__":
    from inspect_robots.cli import main

    raise SystemExit(main())
