from __future__ import annotations

import io
import time

import numpy as np
import pytest

from inspect_robots_jev.vision_protocol import (
    PROMPT_VERSIONS, SERVICE_VERSION, SIM_PROFILE, TRAY_PROFILE,
    decode_response, dumps, encode_request, loads,
)
from inspect_robots_jev_vision.backend import Region
from inspect_robots_jev_vision.server import make_handler


class FakeBackend:
    model_version = {"detector_id": "fake-dino", "detector_revision": "sha-detector",
                     "segmenter_id": "fake-sam", "segmenter_revision": "sha-segmenter"}

    def infer(self, image: np.ndarray, task_profile: str = "sim_put_everything_in_box") -> list[Region]:
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        mask[1:3, 1:3] = True
        regions = [Region("red_block", (1, 1, 3, 3), mask, 0.91, 0.92, False)]
        if task_profile == TRAY_PROFILE:
            tray_mask = np.zeros(image.shape[:2], dtype=np.bool_)
            tray_mask[1:3, 3:5] = True
            regions.append(Region("tray", (3, 1, 5, 3), tray_mask, 0.88, 0.87, False))
        return regions


def post(value: dict[str, object], backend=None) -> tuple[int, dict[str, object]]:
    """Exercise the real HTTP handler in memory; no sandbox socket needed."""
    handler_type = make_handler(backend or FakeBackend())
    handler = handler_type.__new__(handler_type)
    payload = dumps(value)
    handler.path = "/v1/detect"
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)
    handler.wfile = io.BytesIO()
    status = []
    handler.send_response = lambda code: status.append(code)
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    handler.do_POST()
    return status[0], loads(handler.wfile.getvalue())


def test_local_service_stable_fields_and_exact_timestamp() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[1:3, 1:3] = (255, 10, 10)
    stamp = time.monotonic() - 0.125
    request = encode_request(image, "top_cam", "stable-id", stamp)
    status, body = post(request)
    assert status == 200
    assert body["captured_at"] == stamp
    assert body["request_id"] == "stable-id"
    assert body["camera"] == "top_cam"
    assert body["model_version"] == FakeBackend.model_version
    detections = decode_response(body, request_id="stable-id", camera="top_cam",
                                 captured_at=stamp, height=4, width=5)
    assert len(detections) == 1
    assert detections[0].category == "red_block"
    assert detections[0].mask.sum() == 4
    assert detections[0].captured_at == stamp
    assert body["detections"][0]["captured_at"] == stamp


@pytest.mark.parametrize(("change", "error"), [
    (lambda body: body.update(joint_pos=[0] * 14), "invalid_protocol"),
    (lambda body: body.update(height=99), "size_mismatch"),
    (lambda body: body.update(camera="side_cam"), "invalid_camera"),
    (lambda body: body.update(version=3), "unsupported_version"),
])
def test_service_rejects_non_rgb_or_bad_metadata(change, error: str) -> None:
    request = encode_request(np.zeros((4, 5, 3), dtype=np.uint8), "top_cam", "id", 10.0)
    change(request)
    status, body = post(request)
    assert status == 400
    assert body["error"] == error


def test_tray_v2_echo_and_fingerprint() -> None:
    stamp = 123.125
    request = encode_request(np.zeros((4, 5, 3), dtype=np.uint8),
                             "left_cam", "tray-id", stamp, task_profile=TRAY_PROFILE)
    assert request["version"] == 2 and request["task_profile"] == TRAY_PROFILE
    assert set(request) == {"version", "task_profile", "request_id", "camera",
                            "captured_at", "height", "width", "rgb_base64"}
    status, body = post(request)
    assert status == 200
    assert body["version"] == 2 and body["task_profile"] == TRAY_PROFILE
    assert body["camera"] == "left_cam" and body["captured_at"] == stamp
    assert body["model_version"] == {**FakeBackend.model_version,
                                     "prompt_version": PROMPT_VERSIONS[TRAY_PROFILE],
                                     "service_version": SERVICE_VERSION}
    detections = decode_response(body, request_id="tray-id", camera="left_cam",
                                 captured_at=stamp, height=4, width=5,
                                 task_profile=TRAY_PROFILE)
    assert {d.category for d in detections} == {"red_block", "tray"}
    assert all(d.mask.any() and d.captured_at == stamp for d in detections)


def test_explicit_sim_v2_keeps_box_categories() -> None:
    request = encode_request(np.zeros((4, 5, 3), dtype=np.uint8),
                             "top_cam", "sim-v2", 11.0, task_profile=SIM_PROFILE)
    status, body = post(request)
    assert status == 200 and body["task_profile"] == SIM_PROFILE
    assert body["model_version"]["prompt_version"] == PROMPT_VERSIONS[SIM_PROFILE]
    assert {d.category for d in decode_response(body, request_id="sim-v2", camera="top_cam",
                                                captured_at=11.0, height=4, width=5,
                                                task_profile=SIM_PROFILE)} == {"red_block"}


@pytest.mark.parametrize(("change", "error"), [
    (lambda body: body.update(task_profile="other"), "unsupported_profile"),
    (lambda body: body.update(version=3), "unsupported_version"),
    (lambda body: body.update(joint_pos=[0] * 14), "invalid_protocol"),
])
def test_tray_rejects_unsupported_or_private_inputs(change, error: str) -> None:
    request = encode_request(np.zeros((4, 5, 3), dtype=np.uint8),
                             "top_cam", "id", 10.0, task_profile=TRAY_PROFILE)
    change(request)
    status, body = post(request)
    assert status == 400 and body["error"] == error


def test_backend_bad_mask_and_wrong_category_fail_explicitly() -> None:
    class BadBackend(FakeBackend):
        def __init__(self, category: str, mask_width: int) -> None:
            self.category = category
            self.mask_width = mask_width

        def infer(self, image: np.ndarray, task_profile: str = TRAY_PROFILE) -> list[Region]:
            mask = np.zeros((4, self.mask_width), dtype=np.bool_)
            return [Region(self.category, (1, 1, 3, 3), mask, 0.9, 0.9, False)]

    request = encode_request(np.zeros((4, 5, 3), dtype=np.uint8),
                             "top_cam", "bad-output", 10.0, task_profile=TRAY_PROFILE)
    assert post(request, BadBackend("tray", 4)) == (400, {"version": 2, "error": "invalid_mask"})
    assert post(request, BadBackend("box", 5)) == (400, {"version": 2, "error": "category_mismatch"})
    assert post(request, BadBackend("tray", 5))[1]["error"] == "invalid_mask"
