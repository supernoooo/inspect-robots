"""Run the installed Inspect Robots CLI with a camera-configured SO-101 arm.

See _instructions_so101.sh for installation, calibration, and evaluation commands.
Hardware is connected by the SO-101 plugin only when an evaluation starts.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

from inspect_robots.cli import main
from inspect_robots.conformance import DeviceSlot
from inspect_robots.registry import embodiment

DEFAULT_CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"


@embodiment("so101_configured")
def configured_so101(**overrides: Any) -> Any:
    """Build a single-camera arm and describe its joints to the agent policy."""
    from inspect_robots_so101 import MOTORS, SOArmConfig, SOArmEmbodiment

    camera_name = os.environ.get("SO101_CAMERA_NAME", "front")
    camera_device = overrides.pop("camera_device", os.environ.get("SO101_CAMERA", DEFAULT_CAMERA))
    settings: dict[str, Any] = {
        "port": os.environ.get("SO101_PORT", "/dev/ttyACM0"),
        "robot_type": "so101_follower",
        "robot_id": os.environ.get("SO101_ROBOT_ID", "my_so101"),
        "calibration_dir": os.environ.get("SO101_CALIBRATION_DIR") or None,
        "cameras": (camera_name,),
        "cam_width": 640,
        "cam_height": 480,
        "control_hz": 10.0,
        "use_degrees": True,
        "max_relative_target": 1.0,
        "home_pose": None,
    }
    for field in ("joint_low", "joint_high"):
        raw = os.environ.get(f"SO101_{field.upper()}")
        if raw is not None:
            settings[field] = tuple(json.loads(raw))
    settings.update(overrides)

    if "camera_configs" not in settings:
        from lerobot.cameras.opencv import OpenCVCameraConfig

        camera_device = str(camera_device)
        index_or_path = int(camera_device) if camera_device.isdecimal() else camera_device
        settings["camera_configs"] = {
            camera_name: OpenCVCameraConfig(
                index_or_path=index_or_path,
                width=settings["cam_width"],
                height=settings["cam_height"],
                fps=30,
            )
        }

    config = SOArmConfig(**settings)
    arm = SOArmEmbodiment(config)
    space = arm.info.action_space
    # The upstream plugin omits joint labels. Supply names and native units so
    # the LLM can address shoulder_pan, elbow_flex, gripper, etc. explicitly.
    semantics = replace(space.semantics, dim_labels=MOTORS, max_step=(1.0,) * len(MOTORS))
    units = "degrees" if config.use_degrees else "LeRobot normalized joint units (-100 to 100)"
    arm.info = replace(
        arm.info,
        action_space=replace(space, semantics=semantics),
        docs=(
            "SO-101 follower. Action order: "
            + ", ".join(MOTORS)
            + f". The five arm joints use {units}; the gripper uses 0 to 100. "
            "Targets are absolute joint positions. Read the measured joint_pos "
            "before choosing a target; omitted joints should hold their position."
        ),
    )
    return arm


# Tell the setup wizard this rig has one follower serial port and one camera.
setattr(
    configured_so101,
    "DEVICE_SLOTS",
    (
        DeviceSlot(arg="port", kind="serial", label="SO-101 follower serial port"),
        DeviceSlot(arg="camera_device", kind="v4l2", label="front camera (Sonix CAM1)"),
    ),
)


if __name__ == "__main__":
    raise SystemExit(main())
