"""A separately selectable Cartesian EEF interface for the YAM embodiment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from inspect_robots_yam import YamConfig
from inspect_robots_yam.config import DEFAULT_JOINT_HOME_POSE
from inspect_robots_yam.embodiment import YAMEmbodiment

__all__ = ["YamEEFEmbodiment"]

_EEF_GRASP_NOTES = """EEF grasp planning on the real robot:
- The grasp_site tool +z axis is the approach direction, pointing from the wrist toward the fingertips. Tool +y is the jaw-opening and closing axis. A top-down approach needs tool +z to point toward the table. Yaw turns the approach direction within the horizontal plane; pitch and roll can change its tilt.
- Infer the object's location and accessible faces from live camera images. No object pose, demonstration, or scene geometry is supplied. Plan the grasp approach direction as well as x/y/z. If an angled approach is needed, include an explicit orientation target within the move_to bounds rather than leaving the current tilt unchanged.
- Camera image up/down is not the arm's base-frame forward/back. The wrist cameras move and tilt with the gripper, and an object at the bottom of a wrist frame may already be at or behind the fingertips. Check the top view and compare images before/after a small, clear-height motion before deciding which base-frame direction to move. If the target is disappearing under the near image edge, reconsider the direction rather than repeatedly advancing.
- Select a jaw-closing direction from the visible object shape and include yaw when the jaws need to rotate around the vertical axis to span it. Pitch changes the approach tilt; it does not align the jaws with an object rotated on the tabletop.
- Orientation commands and eef_state yaw/pitch/roll are relative to the orientation captured at reset. Compare the requested pose with the next measured eef_state and camera views before closing the gripper. An IK command is not proof that the requested pose was reached. If tracking is poor, try a safer approach position or height while preserving the contact direction needed for the grasp; do not tilt solely to make IK pass.
- Try orientation changes at a clear height before descending. Keep the wrist, camera housing, and fingertips clear of the table throughout the path."""

_DEFAULT_HOME_ORIENTATION_NOTE = (
    "At the default zero-joint home, tool +z points along base +x, so zero "
    "relative pitch/roll is a forward, horizontal approach. Negative pitch "
    "tilts that tool axis toward base -z; yaw or roll alone cannot point it down. "
    "The available pitch bounds may permit only a shallow tilt, not a "
    "top-down grasp."
)


class YamEEFEmbodiment(YAMEmbodiment):
    """Use YAM's native EEF -> IK -> joint command path under ``yam_eef``.

    All YAM constructor options, camera sources, operator hooks, and hardware
    guardrails are inherited. ``control_interface`` is fixed to ``eef_pos`` so
    selecting this registered embodiment cannot silently select joint control.
    Without an explicit home or named start pose, use the same zero-joint,
    open-gripper home as the original YAM joint-control embodiment.
    """

    def __init__(self, config: YamConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            if config.control_interface != "eef_pos":
                raise ValueError("yam_eef requires config.control_interface='eef_pos'")
            if "control_interface" in kwargs:
                raise ValueError("pass control_interface in config or keyword arguments, not both")
            if config.home_pose is None and config.start_pose is None:
                config = replace(config, home_pose=DEFAULT_JOINT_HOME_POSE)
        else:
            interface = kwargs.pop("control_interface", "eef_pos")
            if interface != "eef_pos":
                raise ValueError("yam_eef requires control_interface='eef_pos'")
            if kwargs.get("home_pose") is None and kwargs.get("start_pose") is None:
                kwargs["home_pose"] = DEFAULT_JOINT_HOME_POSE
            kwargs["control_interface"] = "eef_pos"
        super().__init__(config=config, **kwargs)
        docs = self.info.docs + "\n\n" + _EEF_GRASP_NOTES
        if (
            self._cfg.start_pose is None
            and self._cfg.home_pose is not None
            and tuple(self._cfg.home_pose) == DEFAULT_JOINT_HOME_POSE
        ):
            docs += "\n" + _DEFAULT_HOME_ORIENTATION_NOTE
        self.info = replace(self.info, name="yam_eef", docs=docs)
