from __future__ import annotations

import io
import time
import urllib.request

import numpy as np
import pytest

from inspect_robots_jev.vision_client import VisionClient
from inspect_robots_jev.vision_protocol import (
    PROFILE_VERSION, PROMPT_VERSIONS, SERVICE_VERSION, TRAY_PROFILE,
    VERSION, dumps, encode_mask, loads,
)


MODEL = {"detector_id": "fake-dino", "detector_revision": "fixed-a",
         "segmenter_id": "fake-sam", "segmenter_revision": "fixed-b"}
TRAY_MODEL = {**MODEL, "prompt_version": PROMPT_VERSIONS[TRAY_PROFILE],
              "service_version": SERVICE_VERSION}


def response_for(request: dict) -> dict:
    height, width = request["height"], request["width"]
    tray = request["version"] == PROFILE_VERSION and request["task_profile"] == TRAY_PROFILE
    categories = ("red_block", "tray") if tray else ("red_block", "yellow_ball", "box")
    model = (TRAY_MODEL if request["version"] == PROFILE_VERSION else MODEL).copy()
    rows = []
    for index, category in enumerate(categories):
        x1, x2 = (index * 4 + 1, index * 4 + 3) if tray else (index * 4, (index + 1) * 4)
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[1:3, x1:x2] = True
        rows.append({"category": category, "box": [x1, 1, x2, 3], "mask": encode_mask(mask),
                     "detection_score": 0.9, "segmentation_score": 0.9, "occluded": False,
                     "camera": request["camera"], "request_id": request["request_id"],
                     "captured_at": request["captured_at"], "model_version": model.copy()})
    response = {"version": request["version"], "request_id": request["request_id"], "camera": request["camera"],
            "captured_at": request["captured_at"], "height": height, "width": width,
            "model_version": model, "detections": rows}
    if request["version"] == PROFILE_VERSION:
        response["task_profile"] = request["task_profile"]
    return response


@pytest.fixture
def serve(monkeypatch):
    active = {}
    def start(mutate=lambda response: None, *, delay=0.0, status=200, raw=None):
        active.update(mutate=mutate, delay=delay, status=status, raw=raw)
        return "http://127.0.0.1:8765/v1/detect"

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        if active["delay"]:
            raise TimeoutError("fake service timeout")
        sent = loads(request.data)
        response = response_for(sent)
        active["mutate"](response)
        payload = active["raw"] if active["raw"] is not None else dumps(response)
        stream = io.BytesIO(payload)
        stream.status = active["status"]
        stream.headers = {"Content-Length": str(len(payload))}
        return stream

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    yield start


def observe(url: str, **kwargs):
    return VisionClient(url, **kwargs).observe(np.zeros((4, 12, 3), dtype=np.uint8),
                                               "top_cam", time.monotonic() - 0.01)


def test_ready_has_metadata_and_masks(serve) -> None:
    result = observe(serve())
    assert result.state == "ready" and result.failure is None
    assert {d.category for d in result.detections} == {"red_block", "yellow_ball", "box"}
    assert all(d.mask.sum() == 8 and d.model_version == MODEL for d in result.detections)


@pytest.mark.parametrize(("mutate", "reason"), [
    (lambda r: r.update(detections=[]), "empty_detection"),
    (lambda r: r["detections"][0].update(detection_score=0.1), "low_confidence"),
    (lambda r: r["detections"][0].update(segmentation_score=0.1), "low_confidence"),
    (lambda r: r["detections"][0].update(occluded=True), "occluded"),
    (lambda r: r["detections"].pop(), "missing_target"),
    (lambda r: r["detections"][0]["mask"].update(counts=[0, 100]), "invalid_mask"),
    (lambda r: r["detections"][0]["mask"].update(width=11), "invalid_mask"),
    (lambda r: r.update(request_id="wrong"), "request_mismatch"),
    (lambda r: r.update(captured_at=r["captured_at"] + 1), "timestamp_mismatch"),
    (lambda r: r.update(width=11), "size_mismatch"),
    (lambda r: r["detections"][0].update(box=[0, 1, 3, 3]), "invalid_mask"),
    (lambda r: r["detections"][0].update(category=[]), "invalid_protocol"),
])
def test_failures_never_expose_detections(serve, mutate, reason: str) -> None:
    result = observe(serve(mutate))
    assert result.state == "reobserve_hold"
    assert result.failure == reason
    assert result.detections == ()
    if reason in {"low_confidence", "cropped_target", "occluded", "missing_target", "ambiguous_instances"}:
        assert result.review_detections


def test_transport_failures_and_invalid_image(serve) -> None:
    assert observe(serve(delay=0.1), timeout_s=0.01).failure == "timeout"
    assert observe(serve(status=500)).failure == "service_error"
    assert observe(serve(), max_response_bytes=20).failure == "response_too_large"
    assert observe(serve(raw=b'{"a":1,"a":2}')).failure == "invalid_protocol"
    url = serve()
    client = VisionClient(url, max_image_age_s=0.1)
    stale = client.observe(np.zeros((4, 12, 3), dtype=np.uint8), "top_cam", time.monotonic() - 1)
    assert stale.failure == "stale_image" and stale.detections == ()
    invalid = client.observe(np.zeros((4, 12, 3), dtype=np.float32), "top_cam", time.monotonic())
    assert invalid.failure == "invalid_rgb" and invalid.detections == ()


def test_strict_json_and_version(serve) -> None:
    assert observe(serve(lambda r: r.update(version=2))).failure == "unsupported_version"
    assert observe(serve(raw=b'{"bad":NaN}')).failure == "invalid_protocol"


def tray_observe(serve, mutate=lambda response: None, **kwargs):
    client = VisionClient(serve(mutate), task_profile=TRAY_PROFILE,
                          expected_model_version=TRAY_MODEL, **kwargs)
    stamp = time.monotonic() - 0.01
    result = client.observe(np.zeros((4, 12, 3), dtype=np.uint8), "right_cam", stamp)
    return result, stamp


def test_tray_profile_exact_metadata_and_classes(serve) -> None:
    result, stamp = tray_observe(serve)
    assert result.state == "ready" and result.failure is None
    assert result.model_version == TRAY_MODEL
    assert {d.category for d in result.detections} == {"red_block", "tray"}
    assert all(d.captured_at == stamp and d.camera == "right_cam" and
               d.request_id == result.request_id and d.model_version == TRAY_MODEL
               for d in result.detections)
    with pytest.raises(ValueError, match="pinned"):
        VisionClient(serve(), task_profile=TRAY_PROFILE)


@pytest.mark.parametrize(("mutate", "reason"), [
    (lambda r: r.update(detections=[]), "empty_detection"),
    (lambda r: r["detections"][0].update(detection_score=0.1), "low_confidence"),
    (lambda r: r["detections"][0].update(segmentation_score=0.1), "low_confidence"),
    (lambda r: r["detections"][0].update(occluded=True), "occluded"),
    (lambda r: r["detections"].pop(), "missing_target"),
    (lambda r: r["detections"][0].update(category="box"), "category_mismatch"),
    (lambda r: r["detections"][0].update(box=[0, 1, 3, 3]), "cropped_target"),
    (lambda r: r["detections"].append(r["detections"][0].copy()), "ambiguous_instances"),
    (lambda r: r["detections"][0]["mask"].update(counts=[48]), "invalid_mask"),
    (lambda r: r["detections"][0]["mask"].update(width=11), "invalid_mask"),
    (lambda r: r.update(captured_at=r["captured_at"] + 1), "timestamp_mismatch"),
    (lambda r: r.update(width=11), "size_mismatch"),
    (lambda r: r.update(version=1), "unsupported_version"),
    (lambda r: r.update(task_profile="sim_put_everything_in_box"), "profile_mismatch"),
    (lambda r: r["model_version"].update(detector_revision="different"), "model_version_mismatch"),
    (lambda r: r["model_version"].update(prompt_version="wrong"), "model_version_mismatch"),
])
def test_tray_failures_hide_detections(serve, mutate, reason: str) -> None:
    result, _ = tray_observe(serve, mutate)
    assert result.state == "reobserve_hold"
    assert result.failure == reason
    assert result.detections == ()
    if reason in {"low_confidence", "cropped_target", "occluded", "missing_target", "ambiguous_instances"}:
        assert result.review_detections


def test_tray_timeout_and_offline_original_time(serve) -> None:
    assert tray_observe(lambda mutate: serve(mutate, delay=0.1))[0].failure == "timeout"
    client = VisionClient(serve(), task_profile=TRAY_PROFILE,
                          expected_model_version=TRAY_MODEL)
    saved_stamp = 123.25
    image = np.zeros((4, 12, 3), dtype=np.uint8)
    assert client.observe(image, "top_cam", saved_stamp).failure == "stale_image"
    result = client.observe(image, "top_cam", saved_stamp, enforce_age=False)
    assert result.state == "ready"
    assert all(d.captured_at == saved_stamp for d in result.detections)
