"""Offline TypeSafe Choice wire and failure checks."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from inspect_robots_jev.candidates import CandidateGenerator
from inspect_robots_jev.jev_choice import ChoiceError, JevChoiceClient

from conftest import location, observation


def candidates(motion, q0):
    return CandidateGenerator(motion).generate(
        observation(q0), {"red_block": location("red_block", (0.6, 0.2, 0)),
                          "box": location("box", (0.8, 0.2, 0.14))})


def response(ids, selected, *, probabilities=True, model="jev-1.13.0"):
    answer = {"type": "choice", "choice": selected}
    if probabilities:
        answer["probabilities"] = {key: float(key == selected) for key in ids}
    return {"model": model, "answers": {"action": answer}}


def fake_server(monkeypatch, reply, *, status=200):
    calls = []

    def urlopen(request, timeout):
        calls.append((request, timeout))
        if isinstance(reply, Exception):
            raise reply
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "secret body", {}, None)
        stream = io.BytesIO(json.dumps(reply).encode())
        stream.status = 200
        return stream

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_choice_only_current_ids_and_safe_json(monkeypatch, motion, q0):
    found = candidates(motion, q0)
    ids = [item.id for item in found.available]
    selected = next(key for key in ids if key not in ("hold", "reobserve"))
    calls = fake_server(monkeypatch, response(ids, selected))
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    result = JevChoiceClient(url="http://127.0.0.1:9999/v1/systemone", timeout_s=0.2).choose(
        instruction="place objects in box", candidates=found)
    assert result.selected_id == selected and result.probabilities[selected] == 1.0
    request, timeout = calls[0]
    body = json.loads(request.data)
    assert timeout == 0.2
    assert request.get_header("Authorization") == "Bearer secret-test-key"
    assert body["model"] == result.model == "jev-1.13.0"
    assert body["questions"]["action"]["type"] == "choice"
    assert set(body["questions"]["action"]["criteria"]) == set(ids)
    assert "secret-test-key" not in request.data.decode()
    assert "truth" not in request.data.decode().lower()
    assert "joint_pos" not in request.data.decode()


@pytest.mark.parametrize(("reply", "status", "reason"), [
    (TimeoutError("key in exception"), 200, "timeout"),
    ({}, 401, "authentication_error"),
    ({}, 500, "service_error"),
    ({"model": "jev-1.13.0", "answers": {}}, 200, "missing_response"),
    (response(["hold", "reobserve"], "foreign"), 200, "unknown_candidate_id"),
    (response(["hold", "reobserve"], "hold", model="jev-1.14.0"), 200, "model_mismatch"),
])
def test_transport_and_invalid_selection_are_safe(monkeypatch, motion, q0, reply, status, reason):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    fake_server(monkeypatch, reply, status=status)
    with pytest.raises(ChoiceError) as caught:
        JevChoiceClient(url="http://localhost:9999/v1/systemone").choose(
            instruction="place", candidates=candidates(motion, q0))
    assert caught.value.code == reason
    assert "secret" not in str(caught.value)


def test_missing_probability_and_configuration(monkeypatch, motion, q0):
    found = candidates(motion, q0)
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    fake_server(monkeypatch, response([item.id for item in found.available], "hold", probabilities=False))
    result = JevChoiceClient(url="http://localhost:9999/v1/systemone").choose(
        instruction="place", candidates=found)
    assert result.probabilities is None
    for model in ("jev-latest", "jev-preview", "other", "jev-1.13"):
        with pytest.raises(ValueError, match="jev_model"):
            JevChoiceClient(model=model)
    with pytest.raises(ValueError, match="jev_url"):
        JevChoiceClient(url="http://example.com/v1/systemone")
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ChoiceError, match="missing_api_key"):
        JevChoiceClient().choose(instruction="place", candidates=found)
