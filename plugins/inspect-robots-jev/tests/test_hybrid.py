"""Batch 06 closed loop using local fake vision, /act, and Choice endpoints."""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

import json_numpy
import numpy as np
import pytest

from inspect_robots import Scene
from inspect_robots_jev import policy as policy_module
from inspect_robots_jev.molmo_client import NORM_TAG, REPO_ID
from inspect_robots_jev.vision_protocol import VERSION

from test_direct import FIXTURES, MODEL, detection, obs
from test_molmo_client import REVISION, identity


def stream(reply):
    body = io.BytesIO(json_numpy.dumps(reply).encode())
    body.status = 200
    body.headers = {}
    return body


@pytest.fixture
def services(monkeypatch, synthetic_kinematics):
    monkeypatch.setattr(policy_module, "YamKinematics", lambda path: synthetic_kinematics)
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    calls = {"molmo_get": [], "molmo_post": [], "vision": [], "jev": []}
    mode = {"molmo": "normal", "jev": "molmo", "vision": "normal"}

    def urlopen(request, timeout):
        url = request.full_url
        method = request.get_method()
        if url.endswith("/act") and method == "GET":
            calls["molmo_get"].append(url)
            return stream(identity())
        if url.endswith("/act") and method == "POST":
            if mode["molmo"] == "timeout":
                calls["molmo_post"].append(None)
                raise TimeoutError("secret-test-key")
            sent = json_numpy.loads(request.data)
            calls["molmo_post"].append(sent)
            q = sent["state"].astype(np.float64)
            rows = []
            for delta in (0.02, 0.04, 0.8, 0.9):
                row = q.copy()
                row[0] += delta
                row[7] += delta
                rows.append(row)
            if mode["molmo"] == "collision":
                rows[0] = q.copy()
                rows[0][2] = -0.15
                rows[1] = q.copy()
                rows[1][2] = -0.25
            elif mode["molmo"] == "nan":
                rows[2][0] = np.nan
            elif mode["molmo"] == "dimension":
                rows = [q[:13]]
            elif mode["molmo"] == "joint_limit":
                rows[2][0] = 4.0
            elif mode["molmo"] == "gripper":
                rows[2][6] = 1.5
            reply = {"actions": np.stack(rows), "dt_ms": 17.0,
                     "repo_id": REPO_ID, "norm_tag": NORM_TAG,
                     "checkpoint": REPO_ID, "revision": REVISION}
            return stream(reply)
        if url.endswith("/v1/detect"):
            sent = json.loads(request.data)
            calls["vision"].append(sent)
            if mode["vision"] == "timeout":
                raise TimeoutError("secret-test-key")
            rows = []
            if sent["camera"] == "left_cam" and mode["vision"] != "empty":
                pixel = base64.b64decode(sent["rgb_base64"])[0]
                if mode["vision"] != "box_only":
                    rows.append(detection(sent, "red_block", 50 + 2 * pixel))
                rows.append(detection(sent, "box", 78))
            return stream({"version": VERSION, "request_id": sent["request_id"],
                           "camera": sent["camera"], "captured_at": sent["captured_at"],
                           "height": 101, "width": 101, "model_version": MODEL,
                           "detections": rows})
        if url.endswith("/v1/systemone"):
            sent = json.loads(request.data)
            calls["jev"].append(sent)
            if mode["jev"] == "timeout":
                raise TimeoutError("secret-test-key")
            criteria = sent["questions"]["action"]["criteria"]
            if mode["jev"] == "unknown":
                chosen = "foreign:id"
            elif mode["jev"] == "eef":
                chosen = next(key for key in criteria if key not in ("molmo:prefix", "hold", "reobserve"))
            else:
                chosen = "molmo:prefix"
            return stream({"model": "jev-1.13.0", "answers": {"action":
                           {"type": "choice", "choice": chosen,
                            "probabilities": {key: float(key == chosen) for key in criteria}}}})
        raise AssertionError("unexpected service URL")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    policy = policy_module.JevHybridPolicy(
        molmo_url="http://127.0.0.1:8202", molmo_prefix_steps=2, molmo_timeout_s=0.25,
        vision_url="http://127.0.0.1:8765/v1/detect",
        calibration_path=FIXTURES / "synthetic_calibration_v1.json",
        mjcf_path=FIXTURES / "tiny_yam.xml",
        jev_model="jev-1.13.0", jev_url="http://127.0.0.1:9999/v1/systemone",
        cam_height=101, cam_width=101, max_image_age_s=1.0)
    return policy, calls, mode


def test_molmo_prefix_reobserves_and_never_reuses_suffix(services, q0):
    policy, calls, _ = services
    first_obs = obs(q0)
    first = policy.act(first_obs)
    assert policy.info.name == "jev-hybrid"
    assert first.meta["candidate_id"] == "molmo:prefix"
    assert first.meta["reobserve_after_chunk"] is True and len(first) == 2
    assert first.actions[-1].data[0] == pytest.approx(0.04)
    assert len(calls["molmo_post"]) == 1 and len(calls["jev"]) == 1
    audit = policy.audit_records[-1]
    assert audit["molmo_checkpoint"] == REPO_ID and audit["molmo_revision"] == REVISION
    assert audit["molmo_latency_s"] >= 0 and audit["jev_latency_s"] >= 0
    assert audit["molmo_server_latency_ms"] == 17.0
    assert audit["selected_source"] == "molmo"
    assert audit["trajectory"] == [item.data.tolist() for item in first.actions]
    summaries = calls["jev"][0]["questions"]["action"]["criteria"]
    assert summaries["molmo:prefix"]["source"] == "molmo"
    assert summaries["molmo:prefix"]["span"] == {"start": 0, "steps": 2}
    assert any(item["source"] == "eef" for item in summaries.values())
    assert policy.act(first_obs).meta["reason"] == "observation_not_new"
    assert len(calls["molmo_post"]) == 1

    second = policy.act(obs(first.actions[-1].data, image_value=1))
    assert second.meta["candidate_id"] == "molmo:prefix" and len(second) == 2
    assert second.actions[0].data[0] == pytest.approx(0.06)
    assert second.actions[-1].data[0] == pytest.approx(0.08)
    assert len(calls["molmo_post"]) == 2 and len(calls["jev"]) == 2
    assert calls["molmo_post"][1]["state"][0] == pytest.approx(0.04)
    assert calls["molmo_post"][0]["timestamp"] != calls["molmo_post"][1]["timestamp"]
    assert "secret-test-key" not in json.dumps(policy.audit_records)
    assert all("truth" not in str(call).lower() for call in calls["molmo_post"])
    policy.reset(Scene(id="new", instruction="put in box"))
    assert policy.audit_records == () and policy.episode_calls == 0


def test_eef_choice_executes_only_shared_short_segment(services, q0):
    policy, calls, mode = services
    mode["jev"] = "eef"
    result = policy.act(obs(q0))
    assert result.meta["candidate_id"].endswith(":approach")
    assert result.meta["candidate_id"] != "molmo:prefix"
    assert 1 <= len(result) <= policy.candidates.motion.limits.max_steps
    assert policy.audit_records[-1]["selected_source"] == "eef"
    assert policy.audit_records[-1]["trajectory"] == [item.data.tolist() for item in result.actions]
    assert len(calls["molmo_post"]) == 1 and len(calls["jev"]) == 1


@pytest.mark.parametrize(("fault", "expected"), [
    ("nan", "molmo_nonfinite_actions"),
    ("dimension", "molmo_invalid_shape"),
    ("joint_limit", "molmo_joint_limit"),
    ("gripper", "molmo_invalid_gripper"),
    ("collision", "molmo_table_collision"),
    ("timeout", "molmo_timeout"),
])
def test_bad_molmo_holds_without_eef_fallback(services, q0, fault, expected):
    policy, calls, mode = services
    mode["molmo"] = fault
    result = policy.act(obs(q0))
    assert result.meta == {"kind": "hold", "reason": expected}
    np.testing.assert_array_equal(result.actions[0].data, q0)
    assert policy.audit_records[-1]["reason"] == expected
    assert len(calls["molmo_post"]) == 1 and calls["jev"] == []


@pytest.mark.parametrize(("fault", "expected"), [
    ("timeout", "jev_timeout"), ("unknown", "jev_unknown_candidate_id"),
])
def test_jev_failure_holds_without_molmo_fallback(services, q0, fault, expected):
    policy, calls, mode = services
    mode["jev"] = fault
    result = policy.act(obs(q0))
    assert result.meta == {"kind": "hold", "reason": expected}
    np.testing.assert_array_equal(result.actions[0].data, q0)
    assert len(calls["molmo_post"]) == 1 and len(calls["jev"]) == 1
    assert policy.audit_records[-1]["trajectory"] is None


def test_stale_or_vision_failure_never_requests_molmo(services, q0):
    policy, calls, mode = services
    stale = policy.act(obs(q0, at=time.monotonic() - 2))
    assert stale.meta["reason"] == "stale_time" and calls["molmo_post"] == []
    mode["vision"] = "timeout"
    failed = policy.act(obs(q0))
    assert failed.meta["reason"] == "vision_timeout" and calls["molmo_post"] == []


@pytest.mark.parametrize("steps", [0, 7, True, 1.5])
def test_prefix_horizon_is_bounded(services, steps):
    policy, _, _ = services
    with pytest.raises(ValueError, match="molmo_prefix_steps"):
        policy_module.JevHybridPolicy(
            molmo_prefix_steps=steps,
            calibration_path=FIXTURES / "synthetic_calibration_v1.json",
            mjcf_path=FIXTURES / "tiny_yam.xml", cam_height=101, cam_width=101)
