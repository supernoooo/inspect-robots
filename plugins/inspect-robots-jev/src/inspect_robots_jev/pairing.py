"""Hardware-free, fail-closed pairing check for JEV and YAM 0.36.0."""

from __future__ import annotations

import numpy as np

from inspect_robots.compat import CompatIssue, CompatibilityReport, check_compatibility
from inspect_robots_yam.config import YamConfig

from inspect_robots_jev.yam_contract import ACTION_DIM, CAMERA_NAMES, DIM_LABELS, STATE_KEY


def strict_yam_preflight(policy: object, embodiment: object) -> CompatibilityReport:
    """Check the declared spaces and YAM config without reset or device access.

    The core check omits bounds, dimension labels, image sizes and gripper
    encoding. YAM 0.36.0 stores its frozen config on ``_cfg``; absent or
    unexpected configuration is a hard failure, never a guessed default.
    """
    report = check_compatibility(policy, embodiment)  # type: ignore[arg-type]

    def error(code: str, message: str) -> None:
        report.issues.append(CompatIssue("error", code, message))

    info = embodiment.info  # type: ignore[attr-defined]
    pinfo = policy.info  # type: ignore[attr-defined]
    if info.name != "yam_arms" or info.is_simulated:
        error("embodiment", "strict YAM pairing requires real yam_arms")
    cfg = getattr(embodiment, "_cfg", None)
    if not isinstance(cfg, YamConfig):
        error("yam_config", "YAM 0.36.0 frozen configuration is unavailable")
        return report
    if cfg.control_interface != "joints" or cfg.joints_are_delta:
        error("absolute_joints", "YAM must use control_interface=joints and joints_are_delta=false")
    if cfg.gripper_closed != 0.0 or cfg.gripper_open != 1.0:
        error("gripper_encoding", "YAM gripper endpoints must be closed=0 and open=1")

    pbox, ebox = pinfo.action_space, info.action_space
    for name, box in (("policy", pbox), ("embodiment", ebox)):
        semantics = box.semantics
        if box.shape != (ACTION_DIM,) or semantics is None or (
            semantics.control_mode != "joint_pos" or
            semantics.rotation_repr != "none" or
            semantics.frame != "base" or
            semantics.gripper != "continuous"
        ):
            error("action_semantics", f"{name} must declare 14-D absolute joint positions")
        if semantics is None or semantics.dim_labels != DIM_LABELS:
            error("action_order", f"{name} dimension labels differ from YAM wire order")
    if pbox.low is None or pbox.high is None or ebox.low is None or ebox.high is None:
        error("action_bounds", "both action spaces must declare lower and upper bounds")
    elif all(np.asarray(bound).shape == (ACTION_DIM,) and np.isfinite(bound).all()
             for bound in (pbox.low, pbox.high, ebox.low, ebox.high)):
        if np.any(ebox.low > pbox.low) or np.any(ebox.high < pbox.high):
            error("action_bounds", "YAM limits do not contain all policy targets")
        if not np.array_equal(ebox.low, cfg.low) or not np.array_equal(ebox.high, cfg.high):
            error("action_bounds", "YAM declared limits differ from its active config")
        if not np.array_equal(ebox.low[[6, 13]], [0.0, 0.0]) or not np.array_equal(
            ebox.high[[6, 13]], [1.0, 1.0]
        ):
            error("gripper_bounds", "normalized gripper limits must be [0, 1]")
    else:
        error("action_bounds", "action bounds must be finite 14-D vectors")

    for name, obs in (("policy", pinfo.observation_space), ("embodiment", info.observation_space)):
        if tuple(camera.name for camera in obs.cameras) != CAMERA_NAMES:
            error("camera_order", f"{name} must declare top, left, right RGB cameras")
        if obs.state is None or obs.state_keys != frozenset({STATE_KEY}) or (
            len(obs.state.fields) != 1 or
            obs.state.fields[0].key != STATE_KEY or
            obs.state.fields[0].shape != (ACTION_DIM,)
        ):
            error("joint_state", f"{name} must declare 14-D joint_pos state")
    pcams, ecams = pinfo.observation_space.cameras, info.observation_space.cameras
    if len(pcams) != 3 or len(ecams) != 3 or any(
        (p.name, p.height, p.width, p.channels) != (e.name, e.height, e.width, e.channels)
        or e.channels != 3 or (e.height, e.width) != (cfg.cam_height, cfg.cam_width)
        for p, e in zip(pcams, ecams)
    ):
        error("camera_size", "policy RGB dimensions must equal active rig output dimensions")
    return report
