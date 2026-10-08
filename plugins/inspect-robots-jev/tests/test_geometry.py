"""Offline Batch 03 checks using a tiny committed MJCF and synthetic pixels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import Observation
from inspect_robots_isaacsim_2yam.contract import CAMERA_NAMES
from inspect_robots_jev.contract import decode_observation
from inspect_robots_jev.geometry import Calibration, localize_targets
from inspect_robots_jev.kinematics import YamKinematics
from inspect_robots_jev.vision_client import VisionOutcome
from inspect_robots_jev.vision_protocol import Detection

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_yam.xml"
CALIBRATION_FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_calibration_v1.json"


def calibration_data() -> dict[str, Any]:
    return json.loads(CALIBRATION_FIXTURE.read_text(encoding="utf-8"))


def decoded(q: np.ndarray | None = None):
    state = np.r_[np.zeros(6), 0.5, np.zeros(6), 0.5] if q is None else q
    observation = Observation(
        images={name: np.zeros((101, 101, 3), dtype=np.uint8) for name in CAMERA_NAMES},
        state={"joint_pos": state, "object_true_pose": [99, 99, 99]},
        instruction="place objects in box", image_times={name: 99.9 for name in CAMERA_NAMES},
        state_time=99.9, extra=PoisonExtra(),
    )
    return decode_observation(observation, height=101, width=101, max_image_age_s=1, now=100)


class PoisonExtra(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError("truth bait was read")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("truth bait was read")

    def __len__(self) -> int:
        raise AssertionError("truth bait was read")


def detection(camera: str, category: str = "red_block", *, u: int = 50, v: int = 50,
              occluded: bool = False, score: float = 0.9, captured_at: float = 99.9) -> Detection:
    mask = np.zeros((101, 101), dtype=np.bool_)
    mask[v - 2:v + 3, u - 2:u + 3] = True
    return Detection(category, (u - 2, v - 2, u + 3, v + 3), mask, score, score,
                     occluded, camera, "request", captured_at, {"detector_id": "fake"})


def outcomes(*items: Detection) -> dict[str, VisionOutcome]:
    return {name: VisionOutcome("ready", None, tuple(item for item in items if item.camera == name))
            for name in CAMERA_NAMES}


def test_calibration_strict_version_and_units() -> None:
    data = calibration_data()
    assert Calibration.from_json(data).calibration_id == "synthetic-v1"
    assert Calibration.load(CALIBRATION_FIXTURE).calibration_id == "synthetic-v1"
    for change in ({"version": 2}, {"units": "millimetres"}, {"object_true_pose": [1, 2, 3]}):
        with pytest.raises(ValueError):
            Calibration.from_json(dict(data, **change))
    broken = calibration_data()
    broken["cameras"]["left_cam"]["parent"] = "base"
    with pytest.raises(ValueError):
        Calibration.from_json(broken)


def test_three_camera_projection_and_axis_directions() -> None:
    model = YamKinematics(FIXTURE)
    config = Calibration.from_json(calibration_data())
    observation = decoded()
    for camera, expected_xy in (("top_cam", [0, 0]), ("left_cam", [0.69, 0.2]), ("right_cam", [0.69, -0.2])):
        center = localize_targets(observation, outcomes(detection(camera)), config, model)["red_block"]
        assert center.usable and center.reason is None
        np.testing.assert_allclose(center.position_base_m, [*expected_xy, 0], atol=1e-8)
        assert center.sources == (camera,)
        assert center.calibration_id == "synthetic-v1"
        assert center.mjcf_sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        assert center.transform_chains[0][-1] == camera + "_optical"
    shifted = localize_targets(observation, outcomes(detection("top_cam", u=60, v=40)), config, model)["red_block"]
    np.testing.assert_allclose(shifted.position_base_m, [0.1, 0.1, 0], atol=1e-8)
    for camera, center in (("left_cam", [0.69, 0.2]), ("right_cam", [0.69, -0.2])):
        offset = localize_targets(observation, outcomes(detection(camera, u=60, v=40)), config, model)["red_block"]
        np.testing.assert_allclose(offset.position_base_m, [center[0] + 0.07, center[1] + 0.07, 0], atol=1e-8)
    moved_q = observation.joint_pos.copy()
    moved_q[0] = 0.25
    moving = localize_targets(decoded(moved_q), outcomes(detection("left_cam")), config, model)["red_block"]
    assert moving.usable and np.linalg.norm(moving.position_base_m[:2] - [0.69, 0.2]) > 0.1


def test_opening_height_margin_and_quality_failures() -> None:
    config = Calibration.from_json(calibration_data())
    observation = decoded()
    target = localize_targets(observation, outcomes(detection("top_cam", "box")), config, None)["box"]
    assert target.usable and "opening_center_from_mask" in target.flags
    np.testing.assert_allclose(target.position_base_m, [0, 0, 0.14], atol=1e-8)
    assert target.uncertainty_m < config.box_lateral_margin_m
    assert json.loads(json.dumps(target.diagnostic()))["position_base_m"] == [0, 0, 0.14]
    np.testing.assert_allclose(target.opening_half_extents_m,
                               config.box_opening_size_m / 2 - config.box_lateral_margin_m - target.uncertainty_m)
    for item, reason in ((detection("top_cam", occluded=True), "occluded"),
                         (detection("top_cam", score=0.2), "low_confidence"),
                         (detection("top_cam", captured_at=98), "timestamp_mismatch")):
        failed = localize_targets(observation, outcomes(item), config, None)["red_block"]
        assert not failed.usable and failed.reason == reason and failed.candidate_position() is None
    missing = localize_targets(observation, outcomes(detection("top_cam")), None, None)["red_block"]
    assert missing.reason == "missing_calibration" and missing.candidate_position() is None
    disagreement = localize_targets(observation, outcomes(detection("top_cam"), detection("left_cam")), config,
                                    YamKinematics(FIXTURE))["red_block"]
    assert disagreement.reason == "multiview_disagreement" and disagreement.candidate_position() is None


def test_ray_parallel_and_wrist_requires_fk() -> None:
    data = calibration_data()
    data["cameras"]["top_cam"]["transform"]["rotation_wxyz"] = [0.7071067811865476, 0, 0.7071067811865476, 0]
    config = Calibration.from_json(data)
    failed = localize_targets(decoded(), outcomes(detection("top_cam")), config, None)["red_block"]
    assert failed.reason == "ray_does_not_intersect_table_or_degenerate"
    wrist = localize_targets(decoded(), outcomes(detection("left_cam")), config, None)["red_block"]
    assert wrist.reason == "missing_mjcf"


def test_fk_known_pose_and_jacobian_finite_difference() -> None:
    model = YamKinematics(FIXTURE)
    q = np.r_[np.zeros(6), 0.5, np.zeros(6), 0.5]
    np.testing.assert_allclose(model.forward("left", q).position_base_m, [0.69, 0.2, 0.2], atol=1e-10)
    np.testing.assert_allclose(model.forward("right", q).position_base_m, [0.69, -0.2, 0.2], atol=1e-10)
    q[:6] = [0.15, -0.3, 0.4, 0.2, -0.2, 0.1]
    q[7:13] = [-0.2, 0.25, -0.35, 0.2, 0.3, -0.1]
    data = model._data(q)
    for side, start, gripper in (("left", 0, 6), ("right", 7, 13)):
        np.testing.assert_allclose(data.qpos[list(model.arm_qpos[side])], q[start:start + 6])
        np.testing.assert_allclose(data.qpos[list(model.finger_qpos[side])], -0.0475 * q[gripper])
    for side, start in (("left", 0), ("right", 7)):
        jac = model.jacobian(side, q)
        assert jac.shape == (6, 6)
        base = model.forward(side, q)
        for index in range(6):
            altered = q.copy()
            altered[start + index] += 1e-6
            pose = model.forward(side, altered)
            linear = (pose.position_base_m - base.position_base_m) / 1e-6
            delta = pose.rotation_base_eef @ base.rotation_base_eef.T
            angular = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]) / 2e-6
            np.testing.assert_allclose(jac[:, index], np.r_[linear, angular], atol=2e-6)


def test_ik_round_trip_and_unreachable() -> None:
    model = YamKinematics(FIXTURE)
    target_q = np.r_[[0.2, -0.4, 0.5, 0.3, -0.25, 0.15], 0.7,
                     [-0.15, 0.35, -0.45, -0.2, 0.2, 0.25], 0.3]
    seed = target_q.copy()
    seed[:6] += [0.05, -0.06, 0.05, -0.04, 0.03, -0.02]
    seed[7:13] += [-0.04, 0.05, -0.03, 0.04, -0.02, 0.02]
    for side in ("left", "right"):
        pose = model.forward(side, target_q)
        result = model.inverse(side, pose.position_base_m, pose.rotation_base_eef, seed)
        assert result.success, result.reason
        assert result.joint_pos is not None
        actual = model.forward(side, result.joint_pos)
        np.testing.assert_allclose(actual.position_base_m, pose.position_base_m, atol=1e-4)
        np.testing.assert_allclose(actual.rotation_base_eef, pose.rotation_base_eef, atol=2e-3)
        assert result.joint_pos[6] == seed[6] and result.joint_pos[13] == seed[13]
    impossible = model.inverse("left", [10, 0, 0], np.eye(3), seed, max_iterations=12)
    assert not impossible.success and impossible.joint_pos is None
    assert impossible.reason == "unreachable"
    assert json.loads(json.dumps(impossible.diagnostic()))["reason"] == impossible.reason
    zero = np.r_[np.zeros(6), 0.5, np.zeros(6), 0.5]
    at_zero = model.forward("left", zero)
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    singular = model.inverse("left", at_zero.position_base_m, quarter_turn, zero, max_iterations=1)
    assert not singular.success and singular.joint_pos is None and singular.reason == "singular"
