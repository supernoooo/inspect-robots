"""Generic Choice wire, safety boundary and injected transport failures."""

from __future__ import annotations

import io
import json
import socket
import urllib.error

import pytest

from inspect_robots_jev import ChoiceError, ChoiceOption, JevChoiceClient

from test_jev_choice import candidates as legacy_candidates


URL = "http://127.0.0.1:9999/v1/systemone"
OPTIONS = (
    ChoiceOption("move:left", {"operation": "move", "risk": "checked", "steps": 2,
                               "note": "approach red block"}),
    ChoiceOption("move:right", {"operation": "move", "risk": "checked", "steps": 3}),
    ChoiceOption("hold", {"operation": "hold", "risk": "none"}),
)


class FakeTransport:
    def __init__(self, reply: object, *, status: int = 200) -> None:
        self.reply = reply
        self.status = status
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        stream = io.BytesIO(json.dumps(self.reply).encode())
        stream.status = self.status
        return stream


def reply(ids, chosen, *, probabilities=True, usage=None):
    answer = {"type": "choice", "choice": chosen}
    if probabilities:
        answer["probabilities"] = {key: float(key == chosen) for key in ids}
    result = {"model": "jev-1.13.0", "answers": {"action": answer}}
    if usage is not None:
        result["usage"] = usage
    return result


def client(monkeypatch, transport):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    return JevChoiceClient(url=URL, timeout_s=0.25, transport=transport)


def test_multiple_checked_options_and_usage(monkeypatch):
    ids = [item.id for item in OPTIONS]
    usage = {"input_tokens": 72, "output_tokens": 4, "total_tokens": 76}
    transport = FakeTransport(reply(ids, "move:right", usage=usage))
    result = client(monkeypatch, transport).choose_generic(
        instruction="put the block in the box", observation_context="red block visible; gripper open",
        candidates=OPTIONS)
    assert result.selected_id == "move:right"
    assert result.probabilities == {"move:left": 0.0, "move:right": 1.0, "hold": 0.0}
    assert result.model == "jev-1.13.0" and result.latency_s >= 0
    assert result.usage == usage
    request, timeout = transport.calls[0]
    body = json.loads(request.data)
    assert timeout == 0.25
    assert request.get_header("Authorization") == "Bearer secret-test-key"
    assert body == {
        "state": {"instruction": "put the block in the box",
                  "observation_context": "red block visible; gripper open"},
        "model": "jev-1.13.0",
        "questions": {"action": {"type": "choice",
                                 "instructions": ("Select the candidate that makes justified progress "
                                                  "toward the current phase. Hold only when moving is "
                                                  "unjustified or the evidence is insufficient. "
                                                  "A selected action is still subject to execution gates."),
                                 "criteria": {item.id: item.summary for item in OPTIONS}}},
    }
    assert "secret-test-key" not in request.data.decode()


def test_one_action_plus_hold_without_probabilities(monkeypatch):
    options = (OPTIONS[0], OPTIONS[-1])
    transport = FakeTransport(reply([item.id for item in options], "hold", probabilities=False))
    result = client(monkeypatch, transport).choose_generic(
        instruction="place block", observation_context="one checked motion", candidates=options)
    assert result.selected_id == "hold"
    assert result.probabilities is None and result.usage is None
    assert list(json.loads(transport.calls[0][0].data)["questions"]["action"]["criteria"]) == [
        "move:left", "hold"]


def test_default_transport_passes_timeout_as_keyword(monkeypatch):
    calls = []

    def urlopen(request, *, timeout):
        calls.append(timeout)
        stream = io.BytesIO(json.dumps(reply(["move:left", "hold"], "hold")).encode())
        stream.status = 200
        return stream

    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = JevChoiceClient(url=URL, timeout_s=0.25).choose_generic(
        instruction="place", observation_context="current view",
        candidates=(OPTIONS[0], OPTIONS[-1]))
    assert result.selected_id == "hold" and calls == [0.25]


@pytest.mark.parametrize(("fault", "expected"), [
    (reply(["move:left", "hold"], "foreign"), "unknown_candidate_id"),
    (reply(["move:left", "move:right", "hold"], "hold", usage={"input_tokens": -1}), "invalid_usage"),
    (TimeoutError("secret-test-key"), "timeout"),
    (urllib.error.URLError(socket.timeout("secret-test-key")), "timeout"),
    (RuntimeError("secret-test-key"), "service_error"),
])
def test_response_faults_never_choose(monkeypatch, fault, expected):
    transport = FakeTransport(fault)
    with pytest.raises(ChoiceError) as caught:
        client(monkeypatch, transport).choose_generic(
            instruction="place", observation_context="current view", candidates=OPTIONS)
    assert caught.value.code == expected
    assert "secret-test-key" not in str(caught.value)


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failure(monkeypatch, status):
    transport = FakeTransport({}, status=status)
    with pytest.raises(ChoiceError) as caught:
        client(monkeypatch, transport).choose_generic(
            instruction="place", observation_context="current view", candidates=OPTIONS)
    assert caught.value.code == "authentication_error"


@pytest.mark.parametrize(("options", "expected"), [
    ((), "invalid_candidates"),
    ((OPTIONS[0], OPTIONS[0]), "duplicate_candidate_id"),
    ((ChoiceOption("bad id", "safe"),), "invalid_candidates"),
    ((ChoiceOption("move", {"preferred_id": "move"}),), "unsafe_summary"),
    ((ChoiceOption("move", {"rgb_base64": "pixels"}),), "unsafe_summary"),
    ((ChoiceOption("move", {"api_key": "secret"}),), "unsafe_summary"),
    ((ChoiceOption("move", {"eef_target_m": 0.2}),), "unsafe_summary"),
    ((ChoiceOption("move", {"coords": "0.2,0.3"}),), "unsafe_summary"),
    ((ChoiceOption("move", {"joint_pos": 0.2}),), "unsafe_summary"),
    ((ChoiceOption("move", {"safe": [1, 2, 3]}),), "invalid_candidates"),
    ((ChoiceOption("move", {"risk": float("nan")}),), "invalid_candidates"),
])
def test_invalid_or_sensitive_candidates_never_post(monkeypatch, options, expected):
    transport = FakeTransport(reply(["move"], "move"))
    with pytest.raises(ChoiceError) as caught:
        client(monkeypatch, transport).choose_generic(
            instruction="place", observation_context="current view", candidates=options)
    assert caught.value.code == expected
    assert transport.calls == []


def test_sensitive_context_and_api_key_bait_never_post(monkeypatch):
    transport = FakeTransport(reply(["move"], "move"))
    choice = client(monkeypatch, transport)
    for context in ("preferred_id: move", "rgb_base64=AAAA", "secret-test-key"):
        with pytest.raises(ChoiceError, match="unsafe_observation_context"):
            choice.choose_generic(instruction="place", observation_context=context,
                                  candidates=(ChoiceOption("move", "checked"),))
    with pytest.raises(ChoiceError, match="unsafe_summary"):
        choice.choose_generic(instruction="place", observation_context="current view",
                              candidates=(ChoiceOption("move", "secret-test-key"),))
    with pytest.raises(ChoiceError, match="unsafe_instruction"):
        choice.choose_generic(instruction="secret-test-key", observation_context="current view",
                              candidates=(ChoiceOption("move", "checked"),))
    assert transport.calls == []


def test_normalized_2d_summary_is_allowed_but_unrestricted_coordinates_are_not(monkeypatch):
    safe = ChoiceOption("move", {
        "target_object": {"u": 0.4, "v": 0.6, "visible": True, "confidence": 0.8},
        "predicted_visual_deltas": {
            "top_cam": {"du": 0.1, "dv": -0.1, "confidence": 0.7}},
        "predicted_error": 0.2, "error_reduction": 0.1,
    })
    transport = FakeTransport(reply(["move"], "move"))
    result = client(monkeypatch, transport).choose_generic(
        instruction="place", observation_context="bounded visual evidence",
        candidates=(safe,))
    assert result.selected_id == "move"
    with pytest.raises(ChoiceError, match="unsafe_summary"):
        client(monkeypatch, FakeTransport(reply(["move"], "move"))).choose_generic(
            instruction="place", observation_context="current view",
            candidates=(ChoiceOption("move", {"coords": "0.4,0.6"}),))
    with pytest.raises(ChoiceError, match="unsafe_summary"):
        client(monkeypatch, FakeTransport(reply(["move"], "move"))).choose_generic(
            instruction="place", observation_context="current view",
            candidates=(ChoiceOption("move", {"target_object": {"x": 0.4, "y": 0.6}}),))


@pytest.mark.parametrize("probabilities", [
    {"move:left": 1.0},
    {"move:left": 0.5, "move:right": 0.5, "hold": 0.5},
    {"move:left": True, "move:right": 0.0, "hold": 0.0},
    {"move:left": -0.1, "move:right": 1.1, "hold": 0.0},
    {"move:left": 10 ** 1000, "move:right": 0.0, "hold": 0.0},
])
def test_invalid_probabilities(monkeypatch, probabilities):
    response = reply([item.id for item in OPTIONS], "hold", probabilities=False)
    response["answers"]["action"]["probabilities"] = probabilities
    with pytest.raises(ChoiceError, match="invalid_probabilities"):
        client(monkeypatch, FakeTransport(response)).choose_generic(
            instruction="place", observation_context="current view", candidates=OPTIONS)


def test_legacy_adapter_preserves_summary_and_choice(monkeypatch, motion, q0):
    found = legacy_candidates(motion, q0)
    ids = [item.id for item in found.available]
    selected = next(key for key in ids if key not in ("hold", "reobserve"))
    transport = FakeTransport(reply(ids, selected, usage={"total_tokens": 12}))
    result = client(monkeypatch, transport).choose(instruction="place", candidates=found)
    body = json.loads(transport.calls[0][0].data)
    assert body["state"] == {"instruction": "place", "stage": found.stage.value}
    assert body["questions"]["action"]["criteria"] == {
        item.id: item.jev_summary() for item in found.available}
    assert result.selected_id == selected and result.probabilities[selected] == 1.0
    assert result.usage == {"total_tokens": 12}
