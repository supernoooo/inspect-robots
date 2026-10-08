"""Local fake YAM /act transport: identity, wire shape, and unsafe replies."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import json_numpy
import numpy as np
import pytest

from inspect_robots_isaacsim_2yam.contract import CAMERA_NAMES, DIM_LABELS
from inspect_robots_jev.contract import DecodedInput
from inspect_robots_jev.molmo_client import MolmoActClient, MolmoError, NORM_TAG, REPO_ID


REVISION = "a" * 40


def identity(**change):
    return {"status": "ok", "repo_id": REPO_ID, "norm_tag": NORM_TAG,
            "num_cameras": 3, "state_dim": 14, "checkpoint": REPO_ID,
            "revision": REVISION, "action_order": list(DIM_LABELS), **change}


def decoded(q):
    return DecodedInput({name: np.full((2, 3, 3), i, dtype=np.uint8)
                         for i, name in enumerate(CAMERA_NAMES)}, q.copy(), "put in box",
                        {name: 1.0 for name in CAMERA_NAMES}, 1.0)


def stream(reply, status=200):
    body = io.BytesIO(json_numpy.dumps(reply).encode())
    body.status = status
    body.headers = {}
    return body


@pytest.fixture
def transport(monkeypatch, q0):
    calls = []
    mode = {"identity": identity(), "response": {"actions": np.stack([q0, q0]), "dt_ms": 12.0},
            "status": 200, "timeout": False}

    def urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:8202/act"
        assert timeout == 0.25
        calls.append((request.get_method(), request.data))
        if request.get_method() == "GET":
            return stream(mode["identity"])
        if mode["timeout"]:
            raise TimeoutError("server text must not leak")
        if mode["status"] != 200:
            raise urllib.error.HTTPError(request.full_url, mode["status"],
                                         "server text must not leak", {}, None)
        return stream(mode["response"])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return mode, calls


def test_yam_wire_and_one_post_per_call(transport, q0):
    mode, calls = transport
    client = MolmoActClient(timeout_s=0.25)
    result = client.infer(decoded(q0))
    assert [method for method, _ in calls] == ["GET", "POST"]
    request = json_numpy.loads(calls[1][1])
    assert list(request) == [*CAMERA_NAMES, "instruction", "state", "timestamp", "num_steps"]
    assert request["state"].shape == (14,) and request["state"].dtype == np.float32
    assert [request[name][0, 0, 0] for name in CAMERA_NAMES] == [0, 1, 2]
    assert request["instruction"] == "put in box" and request["num_steps"] == 10
    assert result.checkpoint == REPO_ID and result.revision == REVISION
    assert result.server_latency_ms == 12.0 and result.latency_s >= 0
    assert len(result.chunk) == 2
    client.infer(decoded(q0))
    assert [method for method, _ in calls] == ["GET", "POST", "POST"]


@pytest.mark.parametrize(("change", "code"), [
    ({"repo_id": "other"}, "invalid_identity"),
    ({"revision": "latest"}, "invalid_identity"),
    ({"checkpoint": None}, "invalid_identity"),
    ({"action_order": list(reversed(DIM_LABELS))}, "action_order_mismatch"),
])
def test_identity_rejected_before_inference(transport, q0, change, code):
    mode, calls = transport
    mode["identity"].update(change)
    with pytest.raises(MolmoError) as caught:
        MolmoActClient(timeout_s=0.25).infer(decoded(q0))
    assert caught.value.code == code
    assert [method for method, _ in calls] == ["GET"]


@pytest.mark.parametrize(("response", "code"), [
    ({"actions": [[0.0] * 13]}, "invalid_shape"),
    ({"actions": [[float("nan")] * 14]}, "nonfinite_actions"),
    ({"actions": [[0.0] * 6 + [1.2] + [0.0] * 7]}, "invalid_gripper"),
    ({"actions": [[4.0] + [0.0] * 13]}, "joint_limit"),
    ({"actions": []}, "invalid_shape"),
    ({"actions": [[0.0] * 14], "dt_ms": float("inf")}, "invalid_latency"),
    ({"actions": [[0.0] * 14], "revision": "b" * 40}, "identity_mismatch"),
    ({"actions": [[0.0] * 14], "action_order": list(reversed(DIM_LABELS))},
     "action_order_mismatch"),
])
def test_bad_action_response_is_rejected(transport, q0, response, code):
    mode, calls = transport
    mode["response"] = response
    with pytest.raises(MolmoError) as caught:
        MolmoActClient(timeout_s=0.25).infer(decoded(q0))
    assert caught.value.code == code
    assert [method for method, _ in calls] == ["GET", "POST"]


@pytest.mark.parametrize(("failure", "code"), [("timeout", "timeout"), ("status", "http_status")])
def test_transport_failure_has_safe_code(transport, q0, failure, code):
    mode, _ = transport
    mode["timeout"] = failure == "timeout"
    mode["status"] = 503 if failure == "status" else 200
    with pytest.raises(MolmoError) as caught:
        MolmoActClient(timeout_s=0.25).infer(decoded(q0))
    assert caught.value.code == code
    assert "server text" not in str(caught.value)


@pytest.mark.parametrize("url", ["http://user:secret@host:8202", "http://host:8202/act",
                                  "http://host:8202/?token=secret"])
def test_url_rejects_embedded_credentials_or_non_origin(url):
    with pytest.raises(ValueError, match="molmo_url"):
        MolmoActClient(url)
