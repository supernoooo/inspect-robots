"""Batch 02 hardware-free image/state time contract and hold behavior."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inspect_robots import Observation
from inspect_robots_jev import contract
from inspect_robots_jev.contract import InputError, decode_observation
from inspect_robots_jev.policy import _HoldPolicy
from inspect_robots_jev.yam_contract import CAMERA_NAMES


def observed(*, stamp: float = 99.95) -> Observation:
    """Create a complete current YAM observation with legal joint state."""
    return Observation(
        images={name: np.zeros((4, 4, 3), np.uint8) for name in CAMERA_NAMES},
        state={"joint_pos": np.zeros(14)}, instruction="place red block in tray",
        image_times={name: stamp for name in CAMERA_NAMES}, state_time=stamp,
    )


@pytest.mark.parametrize("change,code", [
    (lambda o: o.image_times.pop("left_cam"), "missing_time"),
    (lambda o: o.image_times.update(top_cam=float("nan")), "invalid_time"),
    (lambda o: o.image_times.update(top_cam=float("inf")), "invalid_time"),
    (lambda o: o.image_times.update(top_cam=100.001), "stale_time"),
    (lambda o: o.image_times.update(top_cam=98.999), "stale_time"),
    (lambda o: o.images.pop("right_cam"), "missing_camera"),
    (lambda o: o.image_times.update(top_cam=99.8), "time_skew"),
])
def test_faults_hold_with_valid_current_state(change, code: str,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """All bad image/time inputs produce one current-pose hold, not planning."""
    monkeypatch.setattr(contract.time, "monotonic", lambda: 100.0)
    obs = observed()
    change(obs)
    policy = _HoldPolicy("test", cam_height=4, cam_width=4,
                         max_image_age_s=1.0, max_skew_s=0.1)
    chunk = policy.act(obs)
    assert chunk.meta == {"kind": "hold", "reason": code}
    np.testing.assert_array_equal(chunk.actions[0].data, obs.state["joint_pos"])


def test_age_and_skew_boundaries_are_inclusive() -> None:
    """Fake clock pins exact age and synchronization limits."""
    obs = observed(stamp=99.0)
    obs.image_times["top_cam"] = 99.1
    decoded = decode_observation(obs, height=4, width=4,
                                 max_image_age_s=1.0, max_skew_s=0.1, now=100.0)
    assert decoded.image_times["top_cam"] == 99.1
    assert decoded.state_time == 99.0


def test_state_camera_skew_is_reported() -> None:
    obs = replace(observed(), state_time=99.8)
    with pytest.raises(InputError) as caught:
        decode_observation(obs, height=4, width=4, max_image_age_s=1.0,
                           max_skew_s=0.1, now=100.0)
    assert caught.value.code == "time_skew"
    assert caught.value.joint_pos is not None


def test_legacy_image_map_cannot_be_treated_as_recent() -> None:
    """An unstamped image remains an input fault even with current joints."""
    obs = observed()
    obs.image_times.clear()
    with pytest.raises(InputError, match="missing image time") as caught:
        decode_observation(obs, height=4, width=4, max_image_age_s=1.0, now=100.0)
    assert caught.value.code == "missing_time"
    assert caught.value.joint_pos is not None


def test_invalid_state_cannot_construct_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or non-finite encoder vector stops the policy with a fault."""
    monkeypatch.setattr(contract.time, "monotonic", lambda: 100.0)
    policy = _HoldPolicy("test", cam_height=4, cam_width=4)
    obs = observed()
    obs.state["joint_pos"][0] = float("nan")
    with pytest.raises(InputError) as caught:
        policy.act(obs)
    assert caught.value.code == "invalid_joint_pos"
    assert caught.value.joint_pos is None


def test_policy_requires_a_new_observation_after_action() -> None:
    """The direct policy's pre-vision gate rejects repeated frame/state stamps."""
    from pathlib import Path

    from inspect_robots_jev.policy import JevDirectPolicy
    from inspect_robots_jev.vision_client import VisionOutcome

    fixtures = Path(__file__).parent / "fixtures"
    policy = JevDirectPolicy(calibration_path=fixtures / "synthetic_calibration_v1.json",
                             mjcf_path=fixtures / "tiny_yam.xml", cam_height=101,
                             cam_width=101, max_image_age_s=1.0)
    calls = []
    policy.vision.observe = lambda image, camera, stamp: (
        calls.append((camera, stamp)) or VisionOutcome("reobserve_hold", "empty_detection"))
    stamp = contract.time.monotonic() - 0.1
    first = replace(observed(stamp=stamp),
                    images={name: np.zeros((101, 101, 3), np.uint8) for name in CAMERA_NAMES})
    policy.act(first)
    repeated = policy.act(first)
    assert repeated.meta == {"kind": "hold", "reason": "observation_not_new"}
    assert len(calls) == 3
    newer = replace(observed(stamp=stamp + 0.01),
                    images={name: np.zeros((101, 101, 3), np.uint8) for name in CAMERA_NAMES})
    newer.image_times["right_cam"] = stamp
    still_old = policy.act(newer)
    assert still_old.meta["reason"] == "observation_not_new"
    assert len(calls) == 3
