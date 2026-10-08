"""Strict local-clock vision client; failures never expose grasp detections."""

from __future__ import annotations

import math
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .vision_protocol import (
    CLASSES,
    PROFILES,
    TRAY_PROFILE,
    Detection,
    VisionProtocolError,
    decode_response,
    dumps,
    encode_request,
    loads,
    model_version,
)


@dataclass(frozen=True, eq=False)
class VisionOutcome:
    state: str  # ready or reobserve_hold
    failure: str | None
    detections: tuple[Detection, ...] = ()
    model_version: dict[str, str] | None = None
    request_id: str | None = None
    review_detections: tuple[Detection, ...] = ()


class VisionClient:
    def __init__(self, url: str, *, timeout_s: float = 5.0, max_response_bytes: int = 4_000_000,
                 max_image_age_s: float = 1.0, min_detection_score: float = 0.4,
                 min_segmentation_score: float = 0.5,
                 required_classes: frozenset[str] | None = None,
                 task_profile: str | None = None,
                 expected_model_version: dict[str, str] | None = None) -> None:
        if not url.startswith(("http://127.0.0.1:", "http://localhost:")) or not url.endswith("/v1/detect"):
            raise ValueError("url must be a local /v1/detect endpoint")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or max_response_bytes < 1:
            raise ValueError("invalid transport limits")
        if not math.isfinite(max_image_age_s) or max_image_age_s <= 0:
            raise ValueError("max_image_age_s must be positive")
        if not 0 <= min_detection_score <= 1 or not 0 <= min_segmentation_score <= 1:
            raise ValueError("confidence thresholds must be in [0, 1]")
        if task_profile is not None and task_profile not in PROFILES:
            raise ValueError("unsupported task_profile")
        allowed = CLASSES if task_profile is None else PROFILES[task_profile]
        if required_classes is None:
            required_classes = allowed
        if not required_classes <= allowed:
            raise ValueError("unknown required class")
        if task_profile == TRAY_PROFILE and required_classes != allowed:
            raise ValueError("tray profile requires red_block and tray")
        if task_profile == TRAY_PROFILE and expected_model_version is None:
            raise ValueError("tray profile requires a pinned model fingerprint")
        if expected_model_version is not None:
            expected_model_version = model_version(expected_model_version,
                                                   version=2 if task_profile else 1)
        self.url = url
        self.timeout_s = timeout_s
        self.max_response_bytes = max_response_bytes
        self.max_image_age_s = max_image_age_s
        self.min_detection_score = min_detection_score
        self.min_segmentation_score = min_segmentation_score
        self.required_classes = required_classes
        self.task_profile = task_profile
        self.expected_model_version = expected_model_version

    def observe(self, image: npt.NDArray[np.uint8], camera: str, captured_at: float,
                *, enforce_age: bool = True) -> VisionOutcome:
        try:
            request_id = uuid.uuid4().hex
            body = encode_request(image, camera, request_id, captured_at,
                                  task_profile=self.task_profile)
            if enforce_age and (captured_at > time.monotonic() or time.monotonic() - captured_at > self.max_image_age_s):
                return VisionOutcome("reobserve_hold", "stale_image")
            request = urllib.request.Request(self.url, data=dumps(body),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if response.status != 200:
                    return VisionOutcome("reobserve_hold", "service_error")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) > self.max_response_bytes):
                    return VisionOutcome("reobserve_hold", "response_too_large")
                payload = response.read(self.max_response_bytes + 1)
            if len(payload) > self.max_response_bytes:
                return VisionOutcome("reobserve_hold", "response_too_large")
            response_body = loads(payload)
            detections = decode_response(response_body, request_id=request_id, camera=camera,
                                         captured_at=captured_at, height=image.shape[0], width=image.shape[1],
                                         task_profile=self.task_profile,
                                         expected_model_version=self.expected_model_version)
            fingerprint = model_version(response_body["model_version"],
                                        version=2 if self.task_profile else 1)
            if enforce_age and (captured_at > time.monotonic() or time.monotonic() - captured_at > self.max_image_age_s):
                return VisionOutcome("reobserve_hold", "stale_image")
            if not detections:
                return VisionOutcome("reobserve_hold", "empty_detection", model_version=fingerprint, request_id=request_id)
            if any(d.detection_score < self.min_detection_score or d.segmentation_score < self.min_segmentation_score for d in detections):
                return VisionOutcome("reobserve_hold", "low_confidence", model_version=fingerprint,
                                     request_id=request_id, review_detections=detections)
            if self.task_profile == TRAY_PROFILE and any(
                    d.box[0] == 0 or d.box[1] == 0 or d.box[2] == image.shape[1] or d.box[3] == image.shape[0]
                    for d in detections if d.category in self.required_classes):
                return VisionOutcome("reobserve_hold", "cropped_target", model_version=fingerprint,
                                     request_id=request_id, review_detections=detections)
            if any(d.occluded for d in detections if d.category in self.required_classes):
                return VisionOutcome("reobserve_hold", "occluded", model_version=fingerprint,
                                     request_id=request_id, review_detections=detections)
            if not self.required_classes <= {d.category for d in detections}:
                return VisionOutcome("reobserve_hold", "missing_target", model_version=fingerprint,
                                     request_id=request_id, review_detections=detections)
            if self.task_profile == TRAY_PROFILE and any(sum(d.category == category for d in detections) != 1
                                                         for category in self.required_classes):
                return VisionOutcome("reobserve_hold", "ambiguous_instances", model_version=fingerprint,
                                     request_id=request_id, review_detections=detections)
            return VisionOutcome("ready", None, detections, fingerprint, request_id)
        except VisionProtocolError as exc:
            return VisionOutcome("reobserve_hold", exc.code)
        except (TimeoutError, socket.timeout):
            return VisionOutcome("reobserve_hold", "timeout")
        except urllib.error.HTTPError as exc:
            try:
                error_body = loads(exc.read(min(self.max_response_bytes + 1, 4096)))
                code = error_body.get("error") if isinstance(error_body, dict) else None
            except (VisionProtocolError, OSError):
                code = None
            known = {"invalid_mask", "size_mismatch", "category_mismatch",
                     "unsupported_version", "unsupported_profile", "invalid_protocol",
                     "invalid_camera", "invalid_rgb", "invalid_time"}
            return VisionOutcome("reobserve_hold", code if code in known else "service_error")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return VisionOutcome("reobserve_hold", "timeout")
            return VisionOutcome("reobserve_hold", "service_error")
