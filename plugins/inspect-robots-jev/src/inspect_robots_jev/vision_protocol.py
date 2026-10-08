"""Strict RGB vision wire formats; v1 remains the implicit simulation profile."""

from __future__ import annotations

import base64
import binascii
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots_jev.yam_contract import CAMERA_NAMES

VERSION = 1
PROFILE_VERSION = 2
SIM_PROFILE = "sim_put_everything_in_box"
TRAY_PROFILE = "right_red_block_tray"
PROFILES = {SIM_PROFILE: frozenset({"red_block", "yellow_ball", "box"}),
            TRAY_PROFILE: frozenset({"red_block", "tray"})}
CLASSES = frozenset({"red_block", "yellow_ball", "box"})
PROMPT_VERSIONS = {SIM_PROFILE: "sim-box-v1", TRAY_PROFILE: "right-red-block-tray-v1"}
SERVICE_VERSION = "0.1.0"
MAX_PIXELS = 2048 * 2048


class VisionProtocolError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _fail(code: str, detail: str) -> None:
    raise VisionProtocolError(code, detail)


def _keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        _fail("invalid_protocol", f"{name} keys must be {sorted(expected)}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("invalid_protocol", f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("invalid_protocol", f"{name} must be a nonempty string <= 256 characters")
    return value


def _time(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        _fail("invalid_time", "captured_at must be a finite client monotonic timestamp")
    return float(value)


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        _fail("invalid_protocol", f"{name} must be finite and in [0, 1]")
    if not 0 <= value <= 1:
        _fail("invalid_protocol", f"{name} must be finite and in [0, 1]")
    return float(value)


def loads(data: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("invalid_protocol", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail("invalid_protocol", f"nonfinite JSON number: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionProtocolError("invalid_protocol", "invalid UTF-8 JSON") from exc


def dumps(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _profile(profile: Any) -> str:
    if not isinstance(profile, str) or profile not in PROFILES:
        _fail("unsupported_profile", "unsupported task_profile")
    return profile


def encode_request(image: npt.NDArray[np.uint8], camera: str, request_id: str, captured_at: float,
                   *, task_profile: str | None = None) -> dict[str, Any]:
    if camera not in CAMERA_NAMES:
        _fail("invalid_camera", "unknown camera")
    _text(request_id, "request_id")
    _time(captured_at)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        _fail("invalid_rgb", "image must be H x W x 3 uint8 RGB")
    height, width = image.shape[:2]
    if height < 1 or width < 1 or height * width > MAX_PIXELS:
        _fail("invalid_rgb", "image dimensions are outside supported bounds")
    body = {"version": VERSION if task_profile is None else PROFILE_VERSION,
            "request_id": request_id, "camera": camera,
            "captured_at": captured_at, "height": height, "width": width,
            "rgb_base64": base64.b64encode(image.tobytes()).decode("ascii")}
    if task_profile is not None:
        body["task_profile"] = _profile(task_profile)
    return body


def decode_request(value: Any) -> tuple[npt.NDArray[np.uint8], str, str, float, int, str]:
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] not in (VERSION, PROFILE_VERSION):
        _fail("unsupported_version", "unsupported request version")
    version = value["version"]
    keys = {"version", "request_id", "camera", "captured_at", "height", "width", "rgb_base64"}
    if version == PROFILE_VERSION:
        keys.add("task_profile")
    body = _keys(value, keys, "request")
    profile = SIM_PROFILE if version == VERSION else _profile(body["task_profile"])
    request_id = _text(body["request_id"], "request_id")
    camera = body["camera"]
    if camera not in CAMERA_NAMES:
        _fail("invalid_camera", "unknown camera")
    captured_at = _time(body["captured_at"])
    height = _integer(body["height"], "height", minimum=1)
    width = _integer(body["width"], "width", minimum=1)
    if height * width > MAX_PIXELS:
        _fail("size_mismatch", "image dimensions exceed limit")
    encoded = body["rgb_base64"]
    if not isinstance(encoded, str):
        _fail("invalid_rgb", "rgb_base64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise VisionProtocolError("invalid_rgb", "bad base64 RGB") from exc
    if len(raw) != height * width * 3:
        _fail("size_mismatch", "RGB byte count differs from dimensions")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy(), camera, request_id, captured_at, version, profile


def encode_mask(mask: npt.NDArray[np.bool_]) -> dict[str, Any]:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_:
        _fail("invalid_mask", "mask must be a 2-D bool array")
    height, width = mask.shape
    if not height or not width or height * width > MAX_PIXELS:
        _fail("invalid_mask", "mask dimensions outside bounds")
    counts: list[int] = []
    current = False
    run = 0
    for pixel in mask.flat:
        if bool(pixel) != current:
            counts.append(run)
            current = not current
            run = 0
        run += 1
    counts.append(run)
    return {"encoding": "rle-row-major-v1", "height": height, "width": width, "counts": counts}


def decode_mask(value: Any, height: int, width: int) -> npt.NDArray[np.bool_]:
    if not isinstance(value, dict) or set(value) != {"encoding", "height", "width", "counts"}:
        _fail("invalid_mask", "invalid mask object")
    if value["encoding"] != "rle-row-major-v1" or type(value["height"]) is not int or type(value["width"]) is not int or value["height"] != height or value["width"] != width:
        _fail("invalid_mask", "mask encoding or dimensions mismatch")
    counts = value["counts"]
    if not isinstance(counts, list) or not counts or len(counts) > height * width + 1:
        _fail("invalid_mask", "invalid RLE counts")
    flat = np.zeros(height * width, dtype=np.bool_)
    offset = 0
    for index, count in enumerate(counts):
        if type(count) is not int or count < (0 if index == 0 else 1) or offset + count > flat.size:
            _fail("invalid_mask", "invalid RLE run")
        if index % 2:
            flat[offset:offset + count] = True
        offset += count
    if offset != flat.size or not flat.any():
        _fail("invalid_mask", "mask is incomplete or empty")
    return flat.reshape(height, width)


@dataclass(frozen=True, eq=False)
class Detection:
    category: str
    box: tuple[int, int, int, int]
    mask: npt.NDArray[np.bool_]
    detection_score: float
    segmentation_score: float
    occluded: bool
    camera: str
    request_id: str
    captured_at: float
    model_version: dict[str, str]


def model_version(value: Any, *, version: int = VERSION) -> dict[str, str]:
    keys = {"detector_id", "detector_revision", "segmenter_id", "segmenter_revision"}
    if version == PROFILE_VERSION:
        keys |= {"prompt_version", "service_version"}
    model = _keys(value, keys, "model_version")
    return {key: _text(item, key) for key, item in model.items()}


def decode_response(value: Any, *, request_id: str, camera: str, captured_at: float, height: int, width: int,
                    task_profile: str | None = None,
                    expected_model_version: dict[str, str] | None = None) -> tuple[Detection, ...]:
    version = VERSION if task_profile is None else PROFILE_VERSION
    profile = SIM_PROFILE if task_profile is None else _profile(task_profile)
    keys = {"version", "request_id", "camera", "captured_at", "height", "width", "model_version", "detections"}
    if version == PROFILE_VERSION:
        keys.add("task_profile")
    body = _keys(value, keys, "response")
    if type(body["version"]) is not int or body["version"] != version:
        _fail("unsupported_version", "unsupported response version")
    if version == PROFILE_VERSION and body["task_profile"] != profile:
        _fail("profile_mismatch", "response task_profile differs")
    if body["request_id"] != request_id:
        _fail("request_mismatch", "response request_id differs")
    if body["camera"] != camera:
        _fail("camera_mismatch", "response camera differs")
    if type(body["height"]) is not int or type(body["width"]) is not int or body["height"] != height or body["width"] != width:
        _fail("size_mismatch", "response dimensions differ")
    if _time(body["captured_at"]) != captured_at:
        _fail("timestamp_mismatch", "response capture time differs")
    model = model_version(body["model_version"], version=version)
    if version == PROFILE_VERSION and model["prompt_version"] != PROMPT_VERSIONS[profile]:
        _fail("model_version_mismatch", "prompt version differs")
    if expected_model_version is not None and model != expected_model_version:
        _fail("model_version_mismatch", "model fingerprint differs")
    rows = body["detections"]
    if not isinstance(rows, list) or len(rows) > 100:
        _fail("invalid_protocol", "detections must be a list of <= 100 items")
    detections: list[Detection] = []
    for row in rows:
        item = _keys(row, {"category", "box", "mask", "detection_score", "segmentation_score", "occluded", "camera", "request_id", "captured_at", "model_version"}, "detection")
        if not isinstance(item["category"], str) or item["category"] not in PROFILES[profile]:
            _fail("invalid_protocol" if version == VERSION else "category_mismatch",
                  "category outside task profile")
        if item["request_id"] != request_id:
            _fail("request_mismatch", "detection request_id differs")
        if item["camera"] != camera:
            _fail("camera_mismatch", "detection camera differs")
        if _time(item["captured_at"]) != captured_at:
            _fail("timestamp_mismatch", "detection capture time differs")
        if model_version(item["model_version"], version=version) != model:
            _fail("model_version_mismatch", "detection model version differs")
        box = item["box"]
        if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
            _fail("invalid_protocol", "box must be four integer xyxy coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            _fail("size_mismatch", "box outside image")
        mask = decode_mask(item["mask"], height, width)
        if mask[:y1].any() or mask[y2:].any() or mask[y1:y2, :x1].any() or mask[y1:y2, x2:].any():
            _fail("invalid_mask", "mask extends beyond box")
        if type(item["occluded"]) is not bool:
            _fail("invalid_protocol", "occluded must be boolean")
        detections.append(Detection(item["category"], (x1, y1, x2, y2), mask,
                                    _score(item["detection_score"], "detection_score"),
                                    _score(item["segmentation_score"], "segmentation_score"),
                                    item["occluded"], camera, request_id, captured_at, model.copy()))
    return tuple(detections)
