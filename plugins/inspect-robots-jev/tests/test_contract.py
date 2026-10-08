from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import Observation, Policy, Scene
from inspect_robots.compat import check_compatibility
from inspect_robots.compat import assert_compatible
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import CompatibilityError
from inspect_robots_isaacsim_2yam import IsaacSim2YamEmbodiment
from inspect_robots_isaacsim_2yam.contract import (
    ACTION_DIM, CAMERA_NAMES, GRIPPER_OPEN_POSITION, action_space,
)
from inspect_robots_jev import InputError, decode_observation, hold
from inspect_robots_jev import contract as contract_module
from inspect_robots_jev.kinematics import GRIPPER_OPEN_POSITION as JEV_MJCF_GRIPPER_OPEN
from inspect_robots_jev.pairing import strict_yam_preflight
from inspect_robots_yam.config import YamConfig, action_box, observation_space as yam_observation_space

FIXTURES = Path(__file__).parent / "fixtures"
POLICY_PATHS = {"calibration_path": FIXTURES / "synthetic_calibration_v1.json",
                "mjcf_path": FIXTURES / "tiny_yam.xml"}


class PoisonExtra(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError("Observation.extra was read")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("Observation.extra was read")

    def __len__(self) -> int:
        raise AssertionError("Observation.extra was read")


def observation(*, now: float = 100.0) -> Observation:
    return Observation(
        images={name: np.zeros((2, 3, 3), dtype=np.uint8) for name in CAMERA_NAMES},
        state={"joint_pos": np.r_[np.zeros(6), 0.5, np.zeros(6), 1.0], "truth": [1]},
        instruction="place objects in box",
        image_times={name: now - 0.1 for name in CAMERA_NAMES},
        state_time=now - 0.1,
        extra=PoisonExtra(),
    )


def policy_observation() -> Observation:
    source = observation()
    source.images.update({name: np.zeros((101, 101, 3), dtype=np.uint8)
                          for name in CAMERA_NAMES})
    return source


def make_policy(name: str, monkeypatch: pytest.MonkeyPatch) -> Policy:
    monkeypatch.setattr(contract_module.time, "monotonic", lambda: 100.0)
    matches = {ep.name: ep for ep in entry_points(group="inspect_robots.policies")}
    return matches[name].load()(cam_height=101, cam_width=101,
                                max_image_age_s=0.5, **POLICY_PATHS)


@pytest.mark.parametrize("name", ["jev-direct", "jev-hybrid"])
def test_entry_points_compatible_and_hold(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy(name, monkeypatch)
    assert isinstance(policy, Policy)
    assert policy.info.name == name
    assert check_compatibility(policy, IsaacSim2YamEmbodiment(cam_height=101, cam_width=101)).ok
    assert policy.info.action_space.semantics == action_space().semantics
    assert policy.config.action_horizon == 6 and policy.config.replan_interval is None
    policy.reset(Scene(id="one", instruction="place objects in box"))
    source = policy_observation()
    source.image_times["top_cam"] = 99.0
    result = policy.act(source)
    assert len(result) == 1
    np.testing.assert_array_equal(result.actions[0].data, source.state["joint_pos"])
    assert result.meta == {"kind": "hold", "reason": "stale_time"}
    assert json.loads(json.dumps(result.meta)) == result.meta
    assert policy.episode_calls == 1
    assert policy.diagnostics[-1]["code"] == "stale_time"


def test_whitelist_copies_only_declared_fields() -> None:
    assert JEV_MJCF_GRIPPER_OPEN == GRIPPER_OPEN_POSITION
    source = observation()
    decoded = decode_observation(source, height=2, width=3, max_image_age_s=0.5, now=100.0)
    assert set(decoded.images) == set(CAMERA_NAMES)
    assert set(decoded.image_times) == set(CAMERA_NAMES)
    assert not hasattr(decoded, "extra")
    assert not hasattr(decoded, "truth")
    assert decoded.instruction == source.instruction
    assert decoded.state_time == source.state_time
    assert decoded.images["top_cam"] is not source.images["top_cam"]
    assert decoded.joint_pos is not source.state["joint_pos"]
    assert decoded.joint_pos.shape == (ACTION_DIM,)


@pytest.mark.parametrize("name", ["jev-direct", "jev-hybrid"])
@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda obs: obs.image_times.pop("left_cam"), "missing_time"),
        (lambda obs: obs.image_times.update(top_cam=float("nan")), "invalid_time"),
        (lambda obs: obs.image_times.update(top_cam=99.0), "stale_time"),
        (lambda obs: obs.image_times.update(top_cam=101.0), "stale_time"),
        (lambda obs: replace(obs, state_time=float("inf")), "invalid_time"),
        (lambda obs: replace(obs, state_time=99.0), "stale_time"),
        (lambda obs: obs.images.update(top_cam=np.zeros((2, 3, 4), dtype=np.uint8)), "invalid_rgb"),
        (lambda obs: obs.images.update(left_cam=np.zeros((2, 3, 3), dtype=np.float32)), "invalid_rgb"),
        (lambda obs: obs.images.pop("right_cam"), "missing_camera"),
        (lambda obs: replace(obs, instruction=" "), "invalid_instruction"),
    ],
)
def test_recoverable_inputs_hold(
    name: str, change: Any, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = make_policy(name, monkeypatch)
    source = policy_observation()
    changed = change(source)
    if isinstance(changed, Observation):
        source = changed
    result = policy.act(source)
    assert len(result) == 1
    np.testing.assert_array_equal(result.actions[0].data, source.state["joint_pos"])
    assert result.meta["reason"] == expected
    assert policy.diagnostics[-1]["code"] == expected
    json.dumps(policy.diagnostics)
    policy.reset(Scene(id="two", instruction="another"))
    assert policy.episode_calls == 0
    assert policy.diagnostics == ()


@pytest.mark.parametrize("name", ["jev-direct", "jev-hybrid"])
@pytest.mark.parametrize(
    ("state", "code"),
    [
        ({}, "missing_joint_pos"),
        ({"joint_pos": np.zeros(13)}, "invalid_joint_pos"),
        ({"joint_pos": np.full(14, np.nan)}, "invalid_joint_pos"),
        ({"joint_pos": np.r_[np.zeros(6), 2.0, np.zeros(7)]}, "invalid_joint_pos"),
    ],
)
def test_bad_joint_state_is_explicit_error(
    name: str, state: dict[str, Any], code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = make_policy(name, monkeypatch)
    source = policy_observation()
    source.state.clear()
    source.state.update(state)
    with pytest.raises(InputError) as caught:
        policy.act(source)
    assert caught.value.code == code
    assert caught.value.joint_pos is None


def test_hold_preserves_absolute_wire_order_and_checks_limits() -> None:
    current = np.r_[np.linspace(-1, 1, 6), 0.25, np.linspace(1, -1, 6), 0.75]
    chunk = hold(current, "camera_stale")
    assert len(chunk.actions) == 1
    np.testing.assert_array_equal(chunk.actions[0].data, current)
    assert chunk.actions[0].data is not current
    json.dumps(chunk.meta)
    with pytest.raises(InputError):
        hold(np.full(14, np.inf), "bad")
    with pytest.raises(InputError):
        hold(np.r_[np.zeros(6), -0.1, np.zeros(7)], "bad")
    with pytest.raises(ValueError, match="reason"):
        hold(current, "")


@pytest.mark.parametrize("bad_age", [0.0, -1.0, float("nan"), float("inf")])
def test_max_age_must_be_positive_finite(bad_age: float, monkeypatch: pytest.MonkeyPatch) -> None:
    matches = {ep.name: ep for ep in entry_points(group="inspect_robots.policies")}
    with pytest.raises(ValueError, match="max_image_age_s"):
        matches["jev-direct"].load()(max_image_age_s=bad_age, **POLICY_PATHS)


class FakeYam:
    """Declared YAM spaces with a driver seam that fails if touched."""

    def __init__(self, cfg: YamConfig) -> None:
        self._cfg = cfg
        self.driver_factory = lambda: pytest.fail("driver must not be opened")
        if cfg.control_interface == "eef_pos":
            box = action_box(cfg.eef_low_array, cfg.eef_high_array, control_interface="eef_pos")
        elif cfg.joints_are_delta:
            box = action_box(cfg.delta_low, cfg.delta_high, joints_are_delta=True)
        else:
            box = action_box(cfg.low, cfg.high)
        self.info = EmbodimentInfo(
            name="yam_arms", action_space=box,
            observation_space=yam_observation_space(
                cfg.cam_height, cfg.cam_width, ("top_cam", "left_cam", "right_cam"),
                control_interface=cfg.control_interface),
            is_simulated=False,
        )


def test_real_yam_core_and_strict_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("jev-direct", monkeypatch)
    yam = FakeYam(YamConfig(cam_height=101, cam_width=101))
    assert check_compatibility(policy, yam).ok
    assert strict_yam_preflight(policy, yam).ok
    assert_compatible(policy, yam)  # the normal rollout gate invokes strict preflight


@pytest.mark.parametrize(("mutation", "code"), [
    ("delta", "absolute_joints"),
    ("eef", "absolute_joints"),
    ("camera", "camera_size"),
    ("order", "action_order"),
    ("gripper", "gripper_encoding"),
    ("bounds", "action_bounds"),
])
def test_real_yam_strict_pairing_rejects(
    mutation: str, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = make_policy("jev-direct", monkeypatch)
    options: dict[str, Any] = {"cam_height": 101, "cam_width": 101}
    if mutation == "delta":
        options["joints_are_delta"] = True
    elif mutation == "eef":
        options["control_interface"] = "eef_pos"
    elif mutation == "gripper":
        options["gripper_open"] = 0.5
    elif mutation == "bounds":
        options["joint_low"] = (0.0,) + YamConfig().joint_low[1:]
    yam = FakeYam(YamConfig(**options))
    if mutation == "camera":
        yam.info = replace(yam.info, observation_space=yam_observation_space(
            102, 101, ("top_cam", "left_cam", "right_cam")))
    elif mutation == "order":
        original = yam.info.action_space
        yam.info = replace(yam.info, action_space=replace(
            original, semantics=replace(
                original.semantics,
                dim_labels=tuple(reversed(original.semantics.dim_labels)),
            ),
        ))
    report = strict_yam_preflight(policy, yam)
    assert code in {issue.code for issue in report.errors}
    with pytest.raises(CompatibilityError):
        assert_compatible(policy, yam)
