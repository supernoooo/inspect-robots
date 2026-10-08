"""Batch 05 end-to-end loop with fake RGB and Choice HTTP services."""

from __future__ import annotations

import base64
import builtins
import io
import json
import time
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import Observation, Scene
from inspect_robots_isaacsim_2yam.contract import CAMERA_NAMES
from inspect_robots_jev import policy as policy_module
from inspect_robots_jev.vision_protocol import VERSION, encode_mask


FIXTURES = Path(__file__).parent / "fixtures"
MODEL = {"detector_id": "fake-dino", "detector_revision": "a",
         "segmenter_id": "fake-sam", "segmenter_revision": "b"}


class PoisonExtra(Mapping[str, Any]):
    def __getitem__(self, key):
        raise AssertionError("truth accessed")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("truth accessed")

    def __len__(self) -> int:
        raise AssertionError("truth accessed")


def obs(q, *, image_value=0, at=None):
    stamp = time.monotonic() - 0.01 if at is None else at
    images = {name: np.full((101, 101, 3), image_value, dtype=np.uint8) for name in CAMERA_NAMES}
    return Observation(images=images,
                       state={"joint_pos": q.copy(), "truth_object_pose": [100, 100, 100]},
                       instruction="place objects in box", image_times={name: stamp for name in CAMERA_NAMES},
                       state_time=stamp, extra=PoisonExtra())


def detection(request, category, u):
    mask = np.zeros((101, 101), dtype=np.bool_)
    mask[48:53, u - 2:u + 3] = True
    return {"category": category, "box": [u - 2, 48, u + 3, 53],
            "mask": encode_mask(mask), "detection_score": 0.9,
            "segmentation_score": 0.9, "occluded": False,
            "camera": request["camera"], "request_id": request["request_id"],
            "captured_at": request["captured_at"], "model_version": MODEL}


@pytest.fixture
def services(monkeypatch, synthetic_kinematics):
    original_import = builtins.__import__

    def forbid_molmo(name, *args, **kwargs):
        if "molmo" in name.lower():
            raise AssertionError("direct branch attempted to create a MolmoAct2 client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_molmo)
    monkeypatch.setattr(policy_module, "YamKinematics", lambda path: synthetic_kinematics)
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    calls = {"vision": [], "jev": []}
    mode = {"vision": "normal", "jev": "normal", "probabilities": True}

    def urlopen(request, timeout):
        sent = json.loads(request.data)
        if request.full_url.endswith("/v1/detect"):
            calls["vision"].append(sent)
            if mode["vision"] == "timeout":
                raise TimeoutError("secret-test-key")
            rows = []
            if sent["camera"] == "left_cam" and mode["vision"] != "empty":
                pixel = base64.b64decode(sent["rgb_base64"])[0]
                if mode["vision"] != "box_only":
                    rows.append(detection(sent, "red_block", 50 + 2 * pixel))
                rows.append(detection(sent, "box", 78))
            reply = {"version": VERSION, "request_id": sent["request_id"],
                     "camera": sent["camera"], "captured_at": sent["captured_at"],
                     "height": 101, "width": 101, "model_version": MODEL,
                     "detections": rows}
        elif request.full_url.endswith("/v1/systemone"):
            calls["jev"].append(sent)
            if mode["jev"] == "timeout":
                raise TimeoutError("secret-test-key")
            ids = list(sent["questions"]["action"]["criteria"])
            chosen = next(key for key in ids if key not in ("hold", "reobserve"))
            if mode["jev"] == "unknown":
                chosen = "foreign:id"
            elif mode["jev"] == "reobserve":
                chosen = "reobserve"
            answer = {"type": "choice", "choice": chosen}
            if mode["probabilities"]:
                answer["probabilities"] = {key: float(key == chosen) for key in ids}
            reply = {"model": "jev-1.13.0", "answers": {"action": answer}}
        else:
            raise AssertionError("unexpected service URL")
        stream = io.BytesIO(json.dumps(reply).encode())
        stream.status = 200
        stream.headers = {}
        return stream

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    policy = policy_module.JevDirectPolicy(
        vision_url="http://127.0.0.1:8765/v1/detect",
        calibration_path=FIXTURES / "synthetic_calibration_v1.json",
        mjcf_path=FIXTURES / "tiny_yam.xml",
        jev_model="jev-1.13.0", jev_url="http://127.0.0.1:9999/v1/systemone",
        cam_height=101, cam_width=101, max_image_age_s=1.0)
    return policy, calls, mode


def test_full_round_recomputes_from_new_rgb_and_joint_state(services, q0):
    policy, calls, mode = services
    first = policy.act(obs(q0))
    assert 1 <= len(first) <= 6
    assert first.meta["candidate_id"].endswith(":approach")
    assert first.meta["reobserve_after_chunk"] is True
    audit1 = policy.audit_records[-1]
    assert audit1["selected_id"] == first.meta["candidate_id"]
    assert audit1["trajectory"] == [action.data.tolist() for action in first.actions]
    assert audit1["probabilities_status"] == "returned"
    assert audit1["model"] == "jev-1.13.0" and audit1["latency_s"] >= 0
    assert len(calls["jev"]) == 1 and len(calls["vision"]) == 3
    second = policy.act(obs(first.actions[-1].data, image_value=1))
    assert second.meta["candidate_id"].endswith(":align")
    assert second.meta["candidate_id"] != first.meta["candidate_id"]
    assert len(calls["jev"]) == 2 and len(calls["vision"]) == 6
    assert calls["vision"][1]["request_id"] != calls["vision"][4]["request_id"]
    assert calls["vision"][1]["rgb_base64"] != calls["vision"][4]["rgb_base64"]
    assert policy.audit_records[-1]["candidates"] != audit1["candidates"]
    assert policy.audit_records[-1]["trajectory"] != audit1["trajectory"]
    assert all("truth" not in json.dumps(call).lower() and "secret-test-key" not in json.dumps(call)
               for group in calls.values() for call in group)
    assert "secret-test-key" not in json.dumps(policy.audit_records)
    policy.reset(Scene(id="next", instruction="place objects in box"))
    assert policy.audit_records == () and policy.candidates.stage.value == "approach"
    assert policy.episode_calls == 0


@pytest.mark.parametrize(("vision", "jev", "expected"), [
    ("timeout", "normal", "vision_timeout"),
    ("empty", "normal", "vision_empty_detection"),
    ("box_only", "normal", "no_safe_candidate"),
    ("normal", "timeout", "jev_timeout"),
    ("normal", "unknown", "jev_unknown_candidate_id"),
    ("normal", "reobserve", "reobserve"),
])
def test_failure_holds_without_fallback(services, q0, vision, jev, expected):
    policy, calls, mode = services
    mode.update(vision=vision, jev=jev)
    result = policy.act(obs(q0))
    assert result.meta == {"kind": "hold", "reason": expected}
    np.testing.assert_array_equal(result.actions[0].data, q0)
    assert policy.audit_records[-1]["reason"] == expected
    assert (policy.audit_records[-1]["trajectory"] is not None) == (jev == "reobserve")
    assert len(calls["jev"]) == (1 if jev != "normal" and vision == "normal" else 0)


def test_stale_duplicate_and_missing_probability(services, q0):
    policy, calls, mode = services
    stale = policy.act(obs(q0, at=time.monotonic() - 2))
    assert stale.meta["reason"] == "stale_time"
    assert calls["vision"] == calls["jev"] == []
    mode["probabilities"] = False
    current = obs(q0)
    selected = policy.act(current)
    assert "candidate_id" in selected.meta
    audit = policy.audit_records[-1]
    assert audit["probabilities"] is None and audit["probabilities_status"] == "missing"
    before = len(calls["vision"])
    repeated = policy.act(current)
    assert repeated.meta["reason"] == "observation_not_new"
    assert len(calls["vision"]) == before


def test_paths_model_and_dimensions_validated(services):
    policy, _, _ = services
    kwargs = {"calibration_path": FIXTURES / "synthetic_calibration_v1.json",
              "mjcf_path": FIXTURES / "tiny_yam.xml", "cam_height": 101, "cam_width": 101}
    with pytest.raises(ValueError, match="calibration_path"):
        policy_module.JevDirectPolicy(**dict(kwargs, calibration_path="/tmp/nonexistent-jev-calibration"))
    with pytest.raises(ValueError, match="mjcf_path"):
        policy_module.JevDirectPolicy(**dict(kwargs, mjcf_path="/tmp/nonexistent-jev-mjcf"))
    with pytest.raises(ValueError, match="camera dimensions"):
        policy_module.JevDirectPolicy(**dict(kwargs, cam_width=100))
    with pytest.raises(ValueError, match="jev_model"):
        policy_module.JevDirectPolicy(**dict(kwargs, jev_model="jev-latest"))


def test_service_supplied_audit_text_is_redacted(services, q0, monkeypatch):
    policy, calls, _ = services
    monkeypatch.setitem(MODEL, "detector_revision", "secret-test-key")
    result = policy.act(obs(q0))
    assert "candidate_id" in result.meta
    assert "secret-test-key" not in json.dumps(policy.audit_records)
    assert "[REDACTED]" in json.dumps(policy.audit_records)
    assert "secret-test-key" not in json.dumps(calls["jev"])
