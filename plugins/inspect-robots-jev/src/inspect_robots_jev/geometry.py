"""Versioned optical calibration and conservative tabletop localization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from inspect_robots_jev.yam_contract import CAMERA_NAMES

from .contract import DecodedInput
from .kinematics import YamKinematics
from .vision_client import VisionOutcome
from .vision_protocol import CLASSES, Detection

CALIBRATION_VERSION = 1
BASE_FRAME = "base: x_forward,y_left,z_up; metres"
OPTICAL_FRAME = "optical: x_right,y_down,z_forward; metres"


def _fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must have exactly {sorted(expected)}")
    return value


def _vector(value: Any, count: int, label: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, list) or len(value) != count or any(type(x) not in (int, float) for x in value):
        raise ValueError(f"{label} must be a {count}-element numeric list")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str, *, zero_allowed: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero_allowed and value == 0):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


@dataclass(frozen=True, eq=False)
class Transform:
    """Translation and wxyz quaternion for child coordinates in parent frame."""

    translation_m: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]

    @classmethod
    def from_json(cls, value: Any) -> Transform:
        row = _fields(value, {"translation_m", "rotation_wxyz"}, "transform")
        translation = _vector(row["translation_m"], 3, "translation_m")
        quat = _vector(row["rotation_wxyz"], 4, "rotation_wxyz")
        if abs(np.linalg.norm(quat) - 1) > 1e-5:
            raise ValueError("rotation_wxyz must be a unit quaternion")
        w, x, y, z = quat / np.linalg.norm(quat)
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])
        return cls(translation, rotation)


@dataclass(frozen=True, eq=False)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    parent: str
    transform: Transform

    @classmethod
    def from_json(cls, value: Any, name: str) -> CameraCalibration:
        row = _fields(value, {"width", "height", "fx", "fy", "cx", "cy", "parent", "transform"}, name)
        if type(row["width"]) is not int or type(row["height"]) is not int or min(row["width"], row["height"]) < 1:
            raise ValueError("camera dimensions must be positive integers")
        if row["parent"] != ("base" if name == "top_cam" else name.replace("_cam", "_eef")):
            raise ValueError(f"invalid parent for {name}")
        fx, fy = (_positive(row[key], key) for key in ("fx", "fy"))
        cx, cy = (_positive(row[key], key, zero_allowed=True) for key in ("cx", "cy"))
        if cx >= row["width"] or cy >= row["height"]:
            raise ValueError("principal point must lie within image")
        return cls(row["width"], row["height"], fx, fy, cx, cy, row["parent"], Transform.from_json(row["transform"]))


@dataclass(frozen=True, eq=False)
class Calibration:
    calibration_id: str
    cameras: Mapping[str, CameraCalibration]
    table_normal_base: npt.NDArray[np.float64]
    table_offset_m: float
    box_opening_size_m: npt.NDArray[np.float64]
    box_rim_height_m: float
    box_clearance_m: float
    box_lateral_margin_m: float

    @classmethod
    def from_json(cls, value: Any) -> Calibration:
        row = _fields(value, {"version", "calibration_id", "units", "base_frame", "optical_frame", "cameras", "table_plane", "box_opening"}, "calibration")
        if type(row["version"]) is not int or row["version"] != CALIBRATION_VERSION:
            raise ValueError("unsupported calibration version")
        if not isinstance(row["calibration_id"], str) or not row["calibration_id"].strip():
            raise ValueError("calibration_id must be nonempty")
        if row["units"] != "metres+radians" or row["base_frame"] != BASE_FRAME or row["optical_frame"] != OPTICAL_FRAME:
            raise ValueError("coordinate convention or units mismatch")
        cameras = _fields(row["cameras"], set(CAMERA_NAMES), "cameras")
        parsed = {name: CameraCalibration.from_json(cameras[name], name) for name in CAMERA_NAMES}
        plane = _fields(row["table_plane"], {"normal_base", "offset_m"}, "table_plane")
        normal = _vector(plane["normal_base"], 3, "normal_base")
        if abs(np.linalg.norm(normal) - 1) > 1e-5 or normal[2] < 0.5:
            raise ValueError("table normal must be unit length and point upward")
        offset = _positive(plane["offset_m"], "offset_m", zero_allowed=True)
        box = _fields(row["box_opening"], {"size_m", "rim_height_m", "clearance_m", "lateral_margin_m"}, "box_opening")
        size = _vector(box["size_m"], 2, "size_m")
        if np.any(size <= 0):
            raise ValueError("box opening size must be positive")
        rim = _positive(box["rim_height_m"], "rim_height_m")
        clearance = _positive(box["clearance_m"], "clearance_m")
        margin = _positive(box["lateral_margin_m"], "lateral_margin_m")
        if np.any(size / 2 <= margin):
            raise ValueError("box margin must leave a nonempty opening")
        return cls(row["calibration_id"], parsed, normal, offset, size, rim, clearance, margin)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, eq=False)
class Localization:
    category: str
    usable: bool
    position_base_m: npt.NDArray[np.float64] | None
    reason: str | None
    uncertainty_m: float | None
    sources: tuple[str, ...]
    pixels_xy: tuple[tuple[float, float], ...]
    transform_chains: tuple[tuple[str, ...], ...]
    calibration_id: str | None
    calibration_version: int | None
    mjcf_sha256: str | None
    flags: tuple[str, ...] = ()
    opening_half_extents_m: npt.NDArray[np.float64] | None = None

    def candidate_position(self) -> npt.NDArray[np.float64] | None:
        """Return a detached position only when all quality gates passed."""
        return self.position_base_m.copy() if self.usable and self.position_base_m is not None else None

    def diagnostic(self) -> dict[str, object]:
        """JSON-safe quality and provenance record for downstream diagnostics."""
        return {"category": self.category, "usable": self.usable,
                "position_base_m": self.position_base_m.tolist() if self.position_base_m is not None else None,
                "reason": self.reason, "uncertainty_m": self.uncertainty_m,
                "sources": list(self.sources), "pixels_xy": [list(pixel) for pixel in self.pixels_xy],
                "transform_chains": [list(chain) for chain in self.transform_chains],
                "calibration_id": self.calibration_id, "calibration_version": self.calibration_version,
                "mjcf_sha256": self.mjcf_sha256, "flags": list(self.flags),
                "opening_half_extents_m": self.opening_half_extents_m.tolist() if self.opening_half_extents_m is not None else None}


def _failure(category: str, reason: str, calibration: Calibration | None,
             kinematics: YamKinematics | None, *, flags: tuple[str, ...] = ()) -> Localization:
    return Localization(category, False, None, reason, None, (), (), (),
                        calibration.calibration_id if calibration else None,
                        CALIBRATION_VERSION if calibration else None,
                        kinematics.mjcf_sha256 if kinematics else None, flags)


def _camera_pose(name: str, calibration: Calibration, kinematics: YamKinematics | None,
                 joint_pos: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[str, ...]]:
    camera = calibration.cameras[name]
    if camera.parent == "base":
        return camera.transform.translation_m, camera.transform.rotation, ("base", name + "_optical")
    if kinematics is None:
        raise ValueError("missing_mjcf")
    side = "left" if name == "left_cam" else "right"
    eef = kinematics.forward(side, joint_pos)
    rotation = eef.rotation_base_eef @ camera.transform.rotation
    translation = eef.position_base_m + eef.rotation_base_eef @ camera.transform.translation_m
    return translation, rotation, ("wire_joint_pos", "MJCF_base", side + "_eef", name + "_optical")


def _intersect(u: float, v: float, camera: CameraCalibration, origin: npt.NDArray[np.float64],
               rotation: npt.NDArray[np.float64], calibration: Calibration) -> tuple[npt.NDArray[np.float64], float] | None:
    direction = rotation @ np.array([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0])
    direction /= np.linalg.norm(direction)
    normal = calibration.table_normal_base
    incidence = float(abs(normal @ direction))
    denominator = float(normal @ direction)
    if incidence < 0.15 or abs(denominator) < 1e-10:
        return None
    distance = (calibration.table_offset_m - float(normal @ origin)) / denominator
    if distance <= 0:
        return None
    return origin + distance * direction, incidence


def _one(detection: Detection, decoded: DecodedInput, calibration: Calibration,
         kinematics: YamKinematics | None, max_uncertainty_m: float) -> Localization:
    name = detection.camera
    camera = calibration.cameras[name]
    if detection.mask.shape != (camera.height, camera.width) or decoded.images[name].shape[:2] != detection.mask.shape:
        return _failure(detection.category, "dimension_mismatch", calibration, kinematics)
    if detection.captured_at != decoded.image_times[name]:
        return _failure(detection.category, "timestamp_mismatch", calibration, kinematics)
    if detection.occluded:
        return _failure(detection.category, "occluded", calibration, kinematics)
    if min(detection.detection_score, detection.segmentation_score) < 0.5:
        return _failure(detection.category, "low_confidence", calibration, kinematics)
    ys, xs = np.nonzero(detection.mask)
    if not len(xs):
        return _failure(detection.category, "empty_mask", calibration, kinematics)
    u, v = float(np.median(xs)), float(np.median(ys))
    try:
        origin, rotation, chain = _camera_pose(name, calibration, kinematics, decoded.joint_pos)
    except ValueError as exc:
        return _failure(detection.category, str(exc), calibration, kinematics)
    hit = _intersect(u, v, camera, origin, rotation, calibration)
    if hit is None:
        return _failure(detection.category, "ray_does_not_intersect_table_or_degenerate", calibration, kinematics)
    position, incidence = hit
    # Approximate segmentation edge and pixel quantization error in the table
    # plane, without claiming a calibrated probabilistic covariance.
    pixel_error = max(1.5, 0.1 * max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)))
    probes = [_intersect(u + du, v + dv, camera, origin, rotation, calibration)
              for du, dv in ((pixel_error, 0), (0, pixel_error))]
    if any(probe is None for probe in probes):
        return _failure(detection.category, "uncertainty_projection_failed", calibration, kinematics)
    uncertainty = float(max(np.linalg.norm(probe[0] - position) for probe in probes if probe is not None))
    flags = ("oblique_view",) if incidence < 0.3 else ()
    if uncertainty > max_uncertainty_m:
        return _failure(detection.category, "high_uncertainty", calibration, kinematics, flags=flags)
    opening_half_extents = None
    if detection.category == "box":
        # The mask gives a table footprint, while known geometry supplies the
        # opening height. The XY center is valid only if its uncertainty fits
        # inside the safety margin on both axes.
        if uncertainty >= calibration.box_lateral_margin_m:
            return _failure(detection.category, "box_margin_exceeded", calibration, kinematics)
        opening_half_extents = calibration.box_opening_size_m / 2 - calibration.box_lateral_margin_m - uncertainty
        if np.any(opening_half_extents <= 0):
            return _failure(detection.category, "box_opening_too_small", calibration, kinematics)
        position = position + calibration.table_normal_base * (calibration.box_rim_height_m + calibration.box_clearance_m)
        flags += ("opening_center_from_mask",)
    return Localization(detection.category, True, position, None, uncertainty, (name,), ((u, v),), (chain,),
                        calibration.calibration_id, CALIBRATION_VERSION,
                        kinematics.mjcf_sha256 if kinematics else None, flags, opening_half_extents)


def localize_targets(
    decoded: DecodedInput, outcomes: Mapping[str, VisionOutcome], calibration: Calibration | None,
    kinematics: YamKinematics | None, *, max_uncertainty_m: float = 0.04,
    max_multiview_disagreement_m: float = 0.05,
) -> dict[str, Localization]:
    """Fuse singleton class detections; failures never expose a candidate pose.

    Accepts only the Batch 01 decoded input and Batch 02 ready outcomes. Extra
    fields from raw observations or service payloads cannot reach this layer.
    """
    if max_uncertainty_m <= 0 or max_multiview_disagreement_m <= 0:
        raise ValueError("quality thresholds must be positive")
    if calibration is None:
        return {name: _failure(name, "missing_calibration", None, kinematics) for name in sorted(CLASSES)}
    grouped: dict[str, list[Detection]] = {name: [] for name in CLASSES}
    for camera in CAMERA_NAMES:
        outcome = outcomes.get(camera)
        if outcome is None or outcome.state != "ready":
            continue
        for detection in outcome.detections:
            if detection.camera != camera or detection.category not in CLASSES:
                continue
            grouped[detection.category].append(detection)
    result: dict[str, Localization] = {}
    for category in sorted(CLASSES):
        detections = grouped[category]
        if not detections:
            result[category] = _failure(category, "missing_detection", calibration, kinematics)
            continue
        if len({d.camera for d in detections}) != len(detections):
            result[category] = _failure(category, "ambiguous_instances", calibration, kinematics)
            continue
        estimates = [_one(d, decoded, calibration, kinematics, max_uncertainty_m) for d in detections]
        failed = next((item for item in estimates if not item.usable), None)
        if failed is not None:
            result[category] = failed
            continue
        positions = np.stack([item.position_base_m for item in estimates])
        if any(np.linalg.norm(a - b) > max_multiview_disagreement_m
               for a in positions for b in positions):
            result[category] = _failure(category, "multiview_disagreement", calibration, kinematics)
            continue
        weights = np.array([1 / max(item.uncertainty_m or 0, 1e-6)**2 for item in estimates])
        position = np.average(positions, axis=0, weights=weights)
        uncertainty = float(max(item.uncertainty_m or 0 for item in estimates))
        result[category] = Localization(category, True, position, None, uncertainty,
                                        tuple(item.sources[0] for item in estimates),
                                        tuple(item.pixels_xy[0] for item in estimates),
                                        tuple(item.transform_chains[0] for item in estimates),
                                        calibration.calibration_id, CALIBRATION_VERSION,
                                        kinematics.mjcf_sha256 if kinematics else None,
                                        tuple(flag for item in estimates for flag in item.flags),
                                        np.min(np.stack([item.opening_half_extents_m for item in estimates]), axis=0)
                                        if category == "box" else None)
    return result
