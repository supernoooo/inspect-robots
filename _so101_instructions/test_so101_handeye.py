"""Offline hand-eye calibration tests: synthetic geometry, marker images, and files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from _so101_capx import load_extrinsics, load_kinematics
from _so101_handeye import (
    MIN_SAMPLES,
    WEAK_ROTATION,
    deproject,
    detect,
    finish,
    kabsch,
    load_samples,
    marker_point,
    project,
    report,
    result_document,
    save_samples,
    solve,
    write_marker,
)
from lerobot.utils.rotation import Rotation

MODEL = Path(__file__).resolve().parent.parent / ".so101/models/SO101/so101_new_calib.urdf"
K = np.array([[605.0, 0.0, 322.0], [0.0, 604.0, 251.0], [0.0, 0.0, 1.0]])
# Camera ~0.6 m in front of and above the base, looking back and down.
CAMERA_TO_BASE = np.eye(4)
CAMERA_TO_BASE[:3, :3] = (
    Rotation.from_rotvec([0.0, 0.0, 1.57]).as_matrix()
    @ Rotation.from_rotvec([-2.4, 0.0, 0.0]).as_matrix()
)
CAMERA_TO_BASE[:3, 3] = [0.55, 0.05, 0.45]
OFFSET = np.array([0.012, -0.008, 0.02])


def _poses(count: int, rng: np.random.Generator, *, rotate: bool = True) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    for pose in poses:
        if rotate:
            pose[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.7, 3)).as_matrix()
        pose[:3, 3] = rng.uniform([0.15, -0.15, 0.02], [0.35, 0.15, 0.25])
    return poses


def _camera_points(poses: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    markers = np.einsum("nij,j->ni", poses[:, :3, :3], OFFSET) + poses[:, :3, 3]
    base_to_camera = np.linalg.inv(CAMERA_TO_BASE)
    points = markers @ base_to_camera[:3, :3].T + base_to_camera[:3, 3]
    return points + rng.normal(0, noise, points.shape)


def test_deproject_and_project_round_trip():
    point = deproject((400.0, 120.0), 0.7, K)
    assert point[2] == pytest.approx(0.7)
    assert project(point, K) == pytest.approx((400.0, 120.0))


def test_kabsch_recovers_a_rigid_transform():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(10, 3))
    target = source @ CAMERA_TO_BASE[:3, :3].T + CAMERA_TO_BASE[:3, 3]
    assert np.allclose(kabsch(source, target), CAMERA_TO_BASE, atol=1e-9)


def test_solve_recovers_extrinsics_and_marker_offset_under_noise():
    rng = np.random.default_rng(2)
    poses = _poses(12, rng)
    solution = solve(_camera_points(poses, 0.001, rng), poses)

    assert solution.rms < 0.002
    assert np.allclose(solution.camera_to_base[:3, 3], CAMERA_TO_BASE[:3, 3], atol=0.004)
    assert np.allclose(solution.marker_offset, OFFSET, atol=0.004)
    assert solution.rotation_spread >= WEAK_ROTATION
    assert "quality: good" in report(solution, [])


def test_constant_orientation_is_flagged_as_weak():
    rng = np.random.default_rng(3)
    poses = _poses(10, rng, rotate=False)
    solution = solve(_camera_points(poses, 0.0005, rng), poses)

    assert solution.rotation_spread < WEAK_ROTATION
    assert "WARNING: wrist orientation barely varied" in report(solution, [3])
    assert "dropped [3]" in report(solution, [3])


def test_report_grades_usable_and_poor_fits():
    rng = np.random.default_rng(4)
    poses = _poses(10, rng)
    usable = solve(_camera_points(poses, 0.005, rng), poses)
    poor = solve(_camera_points(poses, 0.03, rng), poses)
    assert "quality: usable" in report(usable, [])
    assert "quality: poor" in report(poor, [])


def test_solve_rejects_too_few_or_mismatched_samples():
    with pytest.raises(ValueError, match="at least 4"):
        solve(np.zeros((3, 3)), np.repeat(np.eye(4)[None], 3, axis=0))
    with pytest.raises(ValueError, match="at least 4"):
        solve(np.zeros((5, 3)), np.repeat(np.eye(4)[None], 4, axis=0))


def test_marker_point_uses_median_depth_of_the_marker_center():
    depth = np.full((480, 640), 0.6, dtype=np.float32)
    depth[200:210, 300:310] = 0.0  # a hole inside the marker
    corners = np.array([[290, 190], [350, 190], [350, 250], [290, 250]], dtype=np.float64)

    point, fraction = marker_point(corners, depth, K)

    assert point == pytest.approx(deproject((320.0, 220.0), 0.6, K))
    assert 0.3 < fraction < 1.0


def test_marker_point_rejects_missing_depth_and_tiny_markers():
    corners = np.array([[290, 190], [350, 190], [350, 250], [290, 250]], dtype=np.float64)
    with pytest.raises(ValueError, match="has depth"):
        marker_point(corners, np.zeros((480, 640), dtype=np.float32), K)
    with pytest.raises(ValueError, match="too small"):
        marker_point(np.full((4, 2), -50.0), np.ones((480, 640), dtype=np.float32), K)


def test_written_marker_is_detected_by_id(tmp_path):
    path = tmp_path / "marker.png"
    write_marker(path, marker_id=3, pixels=120)
    marker = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    canvas = np.full((480, 640), 128, dtype=np.uint8)
    canvas[100 : 100 + marker.shape[0], 200 : 200 + marker.shape[1]] = marker
    rgb = np.repeat(canvas[:, :, None], 3, axis=2)

    corners = detect(rgb, 3)

    assert corners is not None and corners.shape == (4, 2)
    assert corners.mean(axis=0) == pytest.approx([200 + 90 - 0.5, 100 + 90 - 0.5], abs=1.5)
    assert detect(rgb, 4) is None
    assert detect(np.zeros_like(rgb), 3) is None
    with pytest.raises(OSError):
        write_marker(tmp_path / "missing-dir" / "m.png")


def test_samples_round_trip_and_missing_file(tmp_path):
    path = tmp_path / "nested" / "samples.json"
    assert load_samples(path) == []
    save_samples(path, [{"joints_deg": [0] * 6, "camera_point_m": [0, 0, 1]}], camera_serial="1")
    assert load_samples(path)[0]["camera_point_m"] == [0, 0, 1]
    assert json.loads(path.read_text())["camera_serial"] == "1"


def test_result_document_is_readable_as_extrinsics(tmp_path):
    rng = np.random.default_rng(5)
    poses = _poses(10, rng)
    document = result_document(solve(_camera_points(poses, 0.0, rng), poses), camera_serial="x")
    path = tmp_path / "extrinsics.json"
    path.write_text(json.dumps(document))
    assert np.allclose(load_extrinsics(path), CAMERA_TO_BASE, atol=1e-6)
    assert document["n_samples"] == 10 and document["camera_serial"] == "x"


@pytest.mark.skipif(not MODEL.is_file(), reason="SO-101 model cache missing")
def test_finish_solves_fk_samples_drops_indices_and_writes_extrinsics(tmp_path, capsys):
    kinematics = load_kinematics(MODEL)
    rng = np.random.default_rng(6)
    joints = [
        np.r_[rng.uniform([-40, -60, -10, -30, -90], [40, 20, 70, 80, 90]), 30.0] for _ in range(12)
    ]
    poses = np.array([kinematics.forward_kinematics(q[:5]) for q in joints])
    points = _camera_points(poses, 0.0005, rng)
    points[4] += 0.05  # one bad capture
    samples = [
        {"joints_deg": q.tolist(), "camera_point_m": p.tolist()}
        for q, p in zip(joints, points, strict=True)
    ]
    out = tmp_path / "front_camera_to_base.json"
    args = argparse.Namespace(drop="4", samples=str(tmp_path / "s.json"), out=str(out))

    assert finish(samples, kinematics, args, "TEST-SERIAL", str(MODEL), K) == 0

    assert np.allclose(load_extrinsics(out)[:3, 3], CAMERA_TO_BASE[:3, 3], atol=0.005)
    document = json.loads(out.read_text())
    assert document["dropped"] == [4] and document["n_samples"] == 11
    assert "quality: good" in capsys.readouterr().out

    args.drop = ""
    assert finish(samples, kinematics, args, None, str(MODEL), None) in {0, 2}
    args.drop = ",".join(str(index) for index in range(12 - MIN_SAMPLES + 1))
    assert finish(samples, kinematics, args, None, str(MODEL), None) == 1
