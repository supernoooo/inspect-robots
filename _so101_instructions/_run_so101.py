"""Rig configuration for the official robocurve/inspect-robots-so101 plugin.

The so101_configured CLI entry configures the official upstream embodiment.
It adds camera objects, joint labels and a leader or measured-home lifecycle.
Hardware connects only when an evaluation starts, never during doctor/preflight.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from inspect_robots.conformance import DeviceSlot
from inspect_robots.errors import ConfigError
from inspect_robots.registry import embodiment

DEFAULT_CAMERA = (
    "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
)
DEFAULT_HEAD_SERIAL = "262522073381"


def _joint_vector(field: str, value: Any) -> tuple[float, ...]:
    message = (
        f"{field} must be a JSON array of six finite numbers. "
        "Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. "
        "The gripper uses 0..100."
    )
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(message) from exc
    if (not isinstance(value, (list, tuple)) or len(value) != 6
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in value)):
        raise ConfigError(message)
    return tuple(value)


def _calibration_path(config: Any) -> Path | None:
    """Select the same saved file as the official LeRobot follower, without connecting."""
    from lerobot.robots.so_follower import SOFollower
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    if not config.robot_id:
        return None
    directory = (Path(config.calibration_dir).expanduser() if config.calibration_dir
                 else HF_LEROBOT_CALIBRATION / ROBOTS / SOFollower.name)
    return directory / f"{config.robot_id}.json"


def _calibrated_config(config: Any) -> Any:
    """Expose existing encoder limits in the same native units as motor actions."""
    from inspect_robots_so101 import MOTORS
    from lerobot.motors.feetech.tables import MODEL_RESOLUTION

    path = _calibration_path(config)
    if path is None or not path.is_file():
        # Preflight remains available before calibration. The official driver
        # still requires calibration matching the arm before an eval starts.
        return config
    try:
        records = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read SO-101 calibration {path}: {exc}") from exc
    if not isinstance(records, dict):
        raise ConfigError(f"SO-101 calibration {path} must contain motor records.")
    lows, highs = [], []
    max_res = MODEL_RESOLUTION["sts3215"] - 1
    for index, motor in enumerate(MOTORS):
        record = records.get(motor)
        if not isinstance(record, dict):
            raise ConfigError(f"SO-101 calibration {path} is missing motor {motor}.")
        lo, hi = record.get("range_min"), record.get("range_max")
        if (isinstance(lo, bool) or isinstance(hi, bool)
                or not isinstance(lo, int) or not isinstance(hi, int) or lo >= hi):
            raise ConfigError(
                f"SO-101 calibration {path}: {motor} requires integer range_min < range_max."
            )
        if index == len(MOTORS) - 1:
            low, high = 0.0, 100.0  # LeRobot RANGE_0_100 gripper.
        elif config.use_degrees:
            # MotorsBus._normalize: degrees = (encoder - range midpoint) * 360 / max_res.
            extent = (hi - lo) * 180.0 / max_res
            low, high = -extent, extent
        else:
            low, high = -100.0, 100.0  # LeRobot RANGE_M100_100 body.
        lows.append(max(config.joint_low[index], low))
        highs.append(min(config.joint_high[index], high))
    if any(lo >= hi for lo, hi in zip(lows, highs, strict=True)):
        raise ConfigError(f"Requested joint bounds do not overlap the saved calibration {path}.")
    return replace(config, joint_low=tuple(lows), joint_high=tuple(highs))


def rig_config(**overrides: Any) -> tuple[Any, dict[str, Any]]:
    """Build the upstream config and separate launcher lifecycle options."""
    from inspect_robots_so101 import SOArmConfig

    camera_name = overrides.pop("camera_name", os.environ.get("SO101_CAMERA_NAME", "wrist"))
    camera_device = overrides.pop("camera_device", os.environ.get("SO101_CAMERA", DEFAULT_CAMERA))
    head_name = overrides.pop("head_camera_name", os.environ.get("SO101_HEAD_CAMERA_NAME", "front"))
    head_serial = overrides.pop(
        "head_camera_serial", os.environ.get("SO101_HEAD_CAMERA_SERIAL", DEFAULT_HEAD_SERIAL)
    )
    if not isinstance(head_serial, (str, int)) or not str(head_serial).strip():
        raise ConfigError("head_camera_serial must be the RealSense SDK serial number.")
    if not camera_name or not head_name or camera_name == head_name:
        raise ConfigError("camera_name and head_camera_name must be distinct, nonempty names.")
    camera_mode = overrides.pop("camera_mode", "both")
    if camera_mode not in ("both", "front", "wrist", "head"):
        raise ConfigError(
            "camera_mode must be 'both', 'front', or 'wrist' "
            "('head' is an alias for 'front')."
        )
    selected_cameras = (
        (head_name, camera_name) if camera_mode == "both"
        else (camera_name,) if camera_mode == "wrist"
        else (head_name,)
    )
    capture_home = overrides.pop("capture_home", True)
    return_home = overrides.pop("return_home", True)
    if not isinstance(capture_home, bool) or not isinstance(return_home, bool):
        raise ConfigError("capture_home and return_home must be True or False.")
    home_pose = overrides.pop("home_pose", None)
    if home_pose is not None:
        home_pose = _joint_vector("home_pose", home_pose)
    leader_port = overrides.pop("leader_port", None)
    leader_id = overrides.pop("leader_id", "leader_arm")
    leader_calibration_dir = overrides.pop("leader_calibration_dir", None)
    if leader_port is not None:
        if not isinstance(leader_port, str) or not leader_port.strip():
            raise ConfigError("leader_port must be a nonempty serial port path.")
        if home_pose is not None:
            raise ConfigError("Set either leader_port or home_pose, not both.")
        if not isinstance(leader_id, str) or not leader_id.strip():
            raise ConfigError("leader_id must be nonempty when leader_port is set.")
        if leader_calibration_dir is not None and (
            not isinstance(leader_calibration_dir, str) or not leader_calibration_dir.strip()
        ):
            raise ConfigError("leader_calibration_dir must be a directory path.")
    elif leader_calibration_dir is not None:
        raise ConfigError("leader_calibration_dir requires leader_port.")
    home_limit = overrides.pop("home_max_relative_target", 5.0)
    if (isinstance(home_limit, bool) or not isinstance(home_limit, (int, float))
            or not math.isfinite(home_limit) or home_limit <= 0):
        raise ConfigError("home_max_relative_target must be positive and finite.")
    home_ramp_secs = overrides.pop("home_ramp_secs", 3.0)
    if (isinstance(home_ramp_secs, bool) or not isinstance(home_ramp_secs, (int, float))
            or not math.isfinite(home_ramp_secs) or home_ramp_secs < 0):
        raise ConfigError("home_ramp_secs must be nonnegative and finite.")
    home_tolerance = overrides.pop("home_tolerance", 1.0)
    if (isinstance(home_tolerance, bool) or not isinstance(home_tolerance, (int, float))
            or not math.isfinite(home_tolerance) or home_tolerance <= 0):
        raise ConfigError("home_tolerance must be positive and finite.")
    arm_position_p_coefficient = overrides.pop("arm_position_p_coefficient", None)
    if (arm_position_p_coefficient is not None
            and (isinstance(arm_position_p_coefficient, bool)
                 or not isinstance(arm_position_p_coefficient, int)
                 or not 1 <= arm_position_p_coefficient <= 32)):
        raise ConfigError("arm_position_p_coefficient must be an integer from 1 to 32.")
    home_options = {
        "capture_home": capture_home,
        "return_home": return_home,
        "home_pose": home_pose,
        "home_max_relative_target": float(home_limit),
        "home_ramp_secs": float(home_ramp_secs),
        "home_tolerance": float(home_tolerance),
        "arm_position_p_coefficient": arm_position_p_coefficient,
        "leader_port": leader_port,
        "leader_id": leader_id,
        "leader_calibration_dir": leader_calibration_dir,
    }
    settings: dict[str, Any] = {
        "port": os.environ.get("SO101_PORT", "/dev/ttyACM0"),
        "robot_type": "so101_follower",
        "robot_id": os.environ.get("SO101_ROBOT_ID", "my_so101"),
        "calibration_dir": os.environ.get("SO101_CALIBRATION_DIR") or None,
        "cameras": selected_cameras,
        "cam_width": 640,
        "cam_height": 480,
        "control_hz": 10.0,
        "use_degrees": True,
        "settle_tolerance": 0.2,
        "home_pose": None,
    }
    # Explicit bounds may narrow calibration limits; stale SO101_JOINT_* env vars are ignored.
    for field in ("joint_low", "joint_high"):
        if field in overrides:
            value = overrides.pop(field)
            settings[field] = _joint_vector(field, value)
    settings.update(overrides)
    if camera_mode != "both" and "camera_configs" in settings:
        raise ConfigError("camera_mode cannot be combined with custom camera_configs.")
    if leader_port is not None:
        follower_port = settings["port"]
        if leader_port == follower_port or (
            isinstance(follower_port, str)
            and Path(leader_port).exists() and Path(follower_port).exists()
            and Path(leader_port).samefile(follower_port)
        ):
            raise ConfigError("leader_port must point to a different arm than the follower port.")

    if "camera_configs" not in settings:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense import RealSenseCameraConfig

        camera_device = str(camera_device)
        index_or_path = int(camera_device) if camera_device.isdecimal() else camera_device
        camera_configs: dict[str, Any] = {}
        if camera_name in selected_cameras:
            camera_configs[camera_name] = OpenCVCameraConfig(
                index_or_path=index_or_path, width=settings["cam_width"],
                height=settings["cam_height"], fps=30,
            )
        if head_name in selected_cameras:
            camera_configs[head_name] = RealSenseCameraConfig(
                serial_number_or_name=str(head_serial), width=settings["cam_width"],
                height=settings["cam_height"], fps=30, use_depth=False,
            )
        settings["camera_configs"] = camera_configs
    elif settings["camera_configs"] and "cameras" not in overrides:
        settings["cameras"] = tuple(settings["camera_configs"])

    try:
        if settings["calibration_dir"]:
            settings["calibration_dir"] = str(Path(settings["calibration_dir"]).expanduser())
        config = _calibrated_config(SOArmConfig(**settings))
        if home_pose is not None:
            # Validate home with the upstream contract without enabling a cap in eval.
            replace(config, home_pose=home_pose, max_relative_target=float(home_limit))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid SO-101 configuration: {exc}") from exc
    if any(low >= high for low, high in zip(config.joint_low, config.joint_high, strict=True)):
        raise ConfigError("Every joint_low value must be below the matching joint_high value.")
    if not 0 <= config.joint_low[5] < config.joint_high[5] <= 100:
        raise ConfigError("The gripper limits (sixth values) must be within 0..100.")
    return config, home_options


def describe_arm(arm: Any) -> Any:
    """Add joint labels and camera descriptions without changing motor behavior."""
    from inspect_robots_so101 import MOTORS

    config = arm._cfg
    space = arm.info.action_space
    semantics = replace(space.semantics, dim_labels=MOTORS)
    units = "degrees" if config.use_degrees else "LeRobot normalized joint units (-100 to 100)"
    target_kind = "joint deltas" if config.joints_are_delta else "absolute joint positions"
    path = _calibration_path(config)
    calibration_docs = (
        f" Joint bounds were converted from the existing calibration file {path.name} "
        "in the driver's native units; these are physical position ranges, not per-step limits. "
        "Calibration does not provide Cartesian reach directions or camera extrinsics."
        if path is not None and path.is_file() else ""
    )
    home_pose = arm._home_session.home_pose
    camera_docs = (
        "Available camera streams: " + ", ".join(config.cameras) + ". "
        "The front stream is the RealSense overview; the wrist stream is arm-mounted."
    )
    home_docs = (
        f" Configured home pose in action order is {list(home_pose)} in the same native units. "
        "The startup ramp and normal shutdown use this deployment pose."
        if home_pose is not None else ""
    )
    if arm._home_session.leader_port is not None:
        home_docs = (
            " At each connection, the current calibrated leader arm joint positions are "
            "captured as the startup and shutdown home target."
        )
    arm.info = replace(
        arm.info,
        action_space=replace(space, semantics=semantics),
        docs=(
            "SO-101 follower using the official robocurve SO-101 driver. Action order: "
            + ", ".join(MOTORS)
            + f". The five arm joints use {units}; the gripper uses 0 to 100. "
            + f"Targets are {target_kind}. Read the measured joint_pos before choosing "
            "a target; omitted joints should hold their position. "
            + camera_docs + " "
            "The standard gripper has one moving jaw and one fixed jaw, not two independently "
            "actuated fingers; 0 is calibrated closed and 100 is calibrated open."
            + calibration_docs + home_docs
        ),
    )
    return arm


@embodiment("so101_configured")
def configured_so101(**overrides: Any) -> Any:
    """CLI rig entry; the upstream driver executes a local feedback home ramp."""
    from _so101_motion import CLIArm

    config, home_options = rig_config(**overrides)
    return describe_arm(CLIArm(config, **home_options))


@embodiment("so101_eef_configured")
def configured_so101_eef(**overrides: Any) -> Any:
    """Expose EEF deltas while retaining the calibrated official joint driver."""
    from _so101_eef import SO101EEFEmbodiment, load_kinematics

    urdf_path = overrides.pop("urdf_path", None)
    tracking_options = {
        key: overrides.pop(key)
        for key in ("tracking_position_tolerance_m", "tracking_orientation_tolerance_deg",
                    "tracking_joint_tolerance_deg")
        if key in overrides
    }
    if urdf_path is None:
        urdf_path = (
            Path(__file__).resolve().parent.parent
            / ".so101/models/SO101/so101_new_calib.urdf"
        )
    # Constructing CLIArm does not connect hardware. Its calibrated limits must
    # also be given to placo; the official URDF has narrower generic limits.
    arm = configured_so101(**overrides)
    kinematics = load_kinematics(
        urdf_path, joint_low=arm._cfg.joint_low, joint_high=arm._cfg.joint_high
    )
    try:
        return SO101EEFEmbodiment(arm, kinematics, **tracking_options)
    except ValueError as exc:
        raise ConfigError(f"Invalid SO-101 EEF configuration: {exc}") from exc


# RealSense is selected by SDK serial, not its V4L device path.
configured_so101.DEVICE_SLOTS = (
    DeviceSlot(arg="port", kind="serial", label="SO-101 follower serial port"),
    DeviceSlot(arg="camera_device", kind="v4l2", label="wrist camera (Sonix CAM1)"),
)
configured_so101_eef.DEVICE_SLOTS = configured_so101.DEVICE_SLOTS


if __name__ == "__main__":
    from inspect_robots.cli import main

    raise SystemExit(main())
