"""One-call proposal contract without an embodiment, controller or hardware."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation
from inspect_robots_agent import (
    AgentProposer,
    CameraVisualEstimate,
    PredictedVisualDelta,
    ProposalBatch,
    ProposalFailure,
    ProposalFeedback,
    ProposalTermination,
    VisualPoint,
)

LABELS = tuple(f"{side}_{part}" for side in ("left", "right") for part in
               (*(f"j{i}" for i in range(6)), "gripper"))
CAMERAS = ("top_cam", "left_cam", "right_cam")


def _spaces() -> tuple[Box, ObservationSpace]:
    box = Box(
        shape=(14,), low=np.full(14, -3.0), high=np.full(14, 3.0),
        semantics=ActionSemantics(control_mode="joint_pos", rotation_repr="none",
                                  gripper="continuous", frame="base", dim_labels=LABELS),
    )
    observation = ObservationSpace(
        cameras=tuple(CameraSpec(name, 2, 2, 3) for name in CAMERAS),
        state=StateSpec(fields=(StateField("joint_pos", (14,), unit="rad+normalized"),)),
    )
    return box, observation


def _observation(level: int = 0) -> Observation:
    return Observation(
        images={name: np.full((2, 2, 3), level, dtype=np.uint8) for name in CAMERAS},
        state={"joint_pos": np.full(14, level / 10, dtype=np.float64)},
    )


def _candidate(index: int) -> dict[str, Any]:
    return {
        "id": f"c{index}", "targets": {"left_j0": index / 10},
        "note": f"Candidate {index} moves the left arm.",
        "intended_effect": "Approach the object",
        "expected_visual_change": "The gripper looks closer to the object",
    }


def _arguments(count: int = 3) -> dict[str, Any]:
    return {
        "scene_summary": "Object ahead of left gripper",
        "phase": "approach",
        "visual_relation": "far",
        "object_state": "on_table",
        "candidates": [_candidate(index) for index in range(count)],
        "preferred_id": "c1",
    }


def _visual_arguments(count: int = 3) -> dict[str, Any]:
    args = _arguments(count)
    visible = {"visible": True, "u": 0.4, "v": 0.6, "confidence": 0.8}
    hidden = {"visible": False, "u": None, "v": None, "confidence": 0.2}
    args["visual_estimates"] = {
        camera: {"target_object": dict(visible), "left_gripper": dict(visible),
                 "right_gripper": dict(hidden), "placement_region": dict(hidden)}
        for camera in CAMERAS
    }
    for candidate in args["candidates"]:
        candidate.update({
            "acting_arm": "left",
            "predicted_visual_deltas": {
                camera: {"du": 0.03, "dv": -0.02, "confidence": 0.7}
                for camera in CAMERAS
            },
            "visual_effect": "closer", "prediction_confidence": 0.7,
            "verifiable_result": "left gripper is closer in at least two views",
        })
    return args


def _call(name: str = "propose_actions", args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "function_call", "call_id": "call_1", "name": name,
            "arguments": json.dumps(_arguments() if args is None else args)}


def _response(*calls: dict[str, Any], status: str = "completed") -> dict[str, Any]:
    return {"id": "resp_1", "model": "gpt-6-astra", "status": status,
            "output": list(calls),
            "usage": {"input_tokens": 101, "output_tokens": 32, "total_tokens": 133}}


class FakeResponsesClient:
    def __init__(self, *responses: dict[str, Any] | Exception):
        self.responses = list(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete_raw(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        temperature: float | None = None, reasoning_effort: str | float | None = None,
    ) -> dict[str, Any]:
        self.requests.append((messages, tools))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _proposer(fake: FakeResponsesClient, **kwargs: Any) -> AgentProposer:
    action, observation = _spaces()
    return AgentProposer(
        model="openai/gpt-6-astra", base_url="http://llm.test/v1", env={},
        action_space=action, observation_space=observation, client=fake, **kwargs,
    )


@pytest.mark.parametrize("count", [2, 3, 4])
def test_one_call_produces_unexecuted_candidates(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_robots_agent._tools import Toolset
    from inspect_robots_agent.policy import LLMAgentPolicy

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("proposal must never execute a tool or call act")

    monkeypatch.setattr(Toolset, "execute", forbidden)
    monkeypatch.setattr(LLMAgentPolicy, "act", forbidden)
    raw = _response(_call(args=_arguments(count)))
    fake = FakeResponsesClient(raw)
    proposer = _proposer(fake)
    result = proposer.propose("pick up the cube", _observation())
    assert isinstance(result, ProposalBatch)
    assert result.status == "proposed"
    assert len(result.candidates) == count
    assert all(candidate.status == "proposed" for candidate in result.candidates)
    assert result.preferred_id == "c1"
    assert (result.phase, result.visual_relation, result.object_state) == (
        "approach", "far", "on_table")
    assert result.candidates[0].expected_visual_change
    assert result.raw_response is raw
    assert result.model == "gpt-6-astra"
    assert result.duration_s >= 0
    assert result.usage == {"input_tokens": 101, "output_tokens": 32, "total_tokens": 133}
    assert proposer.records == (result,)
    messages, tools = fake.requests[0]
    assert [tool["function"]["name"] for tool in tools] == ["propose_actions", "done", "give_up"]
    assert len([part for part in messages[-1]["content"] if part["type"] == "image_url"]) == 3
    assert "left_j0=" in messages[-1]["content"][0]["text"]
    assert all(message["role"] != "tool" for message in messages)


@pytest.mark.parametrize("name,field", [("done", "summary"), ("give_up", "reason")])
def test_terminal_result(name: str, field: str) -> None:
    fake = FakeResponsesClient(_response(_call(name, {field: "Finished", "hindsight": "none"})))
    result = _proposer(fake).propose("task", _observation())
    assert isinstance(result, ProposalTermination)
    assert result.status == name
    assert result.summary == "Finished"
    assert result.hindsight == "none"


@pytest.mark.parametrize("mutate,code", [
    (lambda a: a.pop("scene_summary"), "missing_field"),
    (lambda a: a["candidates"][0].pop("note"), "missing_field"),
    (lambda a: a["candidates"][0].pop("intended_effect"), "missing_field"),
    (lambda a: a["candidates"][0].pop("expected_visual_change"), "missing_field"),
    (lambda a: a.update(phase="teleport"), "invalid_scene_assessment"),
    (lambda a: a["candidates"][1].update(id="c0"), "duplicate_id"),
    (lambda a: a.update(preferred_id="missing"), "invalid_preferred_id"),
    (lambda a: a["candidates"][0]["targets"].update(left_j0=float("nan")), "non_finite_target"),
    (lambda a: a["candidates"][0]["targets"].update(left_j0=float("inf")), "non_finite_target"),
    (lambda a: a["candidates"][0]["targets"].update(bogus=0.1), "unknown_joint"),
    (lambda a: a.update(candidates=a["candidates"][:1]), "candidate_count"),
    (
        lambda a: a.update(candidates=a["candidates"] + [_candidate(3), _candidate(4)]),
        "candidate_count",
    ),
])
def test_invalid_proposal_returns_auditable_failure(mutate: Any, code: str) -> None:
    args = _arguments()
    mutate(args)
    raw = _response(_call(args=args))
    result = _proposer(FakeResponsesClient(raw)).propose("task", _observation())
    assert isinstance(result, ProposalFailure)
    assert result.status == "failed" and result.code == code
    assert result.raw_response is raw
    assert result.usage is not None


@pytest.mark.parametrize("raw,code", [
    (_response(_call(), _call("move_joints", {"targets": {"left_j0": 0.2}})), "tool_call_count"),
    (_response(), "tool_call_count"),
    (_response(_call("move_joints", {"targets": {"left_j0": 0.2}})), "unexpected_tool"),
    (_response(_call(), status="incomplete"), "incomplete_response"),
    (_response({"type": "function_call", "call_id": "x", "name": "propose_actions",
                "arguments": "{broken"}), "invalid_arguments"),
])
def test_invalid_wire_response_is_audited(raw: dict[str, Any], code: str) -> None:
    result = _proposer(FakeResponsesClient(raw)).propose("task", _observation())
    assert isinstance(result, ProposalFailure)
    assert result.code == code and result.raw_response is raw


def test_budget_and_transport_error_are_failures() -> None:
    fake = FakeResponsesClient(
        _response(_call()), RuntimeError("secret in transport error"), _response(_call())
    )
    proposer = _proposer(fake, max_llm_calls=2)
    assert isinstance(proposer.propose("task", _observation()), ProposalBatch)
    failure = proposer.propose("task", _observation())
    assert isinstance(failure, ProposalFailure)
    assert failure.code == "request_error" and "secret" not in failure.detail
    exhausted = proposer.propose("task", _observation())
    assert isinstance(exhausted, ProposalFailure)
    assert exhausted.code == "budget_exhausted" and exhausted.raw_response is None
    assert proposer.calls_used == 2 and len(fake.requests) == 2
    assert proposer.records == (proposer.records[0], failure, exhausted)
    proposer.reset()
    assert proposer.calls_used == 0 and proposer.records == ()
    assert isinstance(proposer.propose("new task", _observation()), ProposalBatch)


def test_next_round_uses_fresh_observation_and_only_selected_feedback() -> None:
    fake = FakeResponsesClient(_response(_call()), _response(_call()))
    proposer = _proposer(fake)
    first = proposer.propose("task", _observation(0))
    assert isinstance(first, ProposalBatch)
    second = proposer.propose(
        "task", _observation(4),
        ProposalFeedback(selected_id="c1", outcome="next measured state is 0.4 on all joints"),
    )
    assert isinstance(second, ProposalBatch)
    messages, _ = fake.requests[1]
    body = json.dumps(messages)
    assert "c1" in body and "c0" not in body and "c2" not in body
    assert "requested absolute targets" in body
    assert "left_j0" in body and "0.1" in body
    assert "left_j0=0.4" in body
    assert "left_j0=0.0" not in body
    assert len([part for part in messages[-1]["content"] if part["type"] == "image_url"]) == 3
    assert all(message["role"] != "tool" for message in messages)
    with pytest.raises(ValueError, match="selected_id"):
        proposer.propose("task", _observation(5), ProposalFeedback("not-a-candidate", "unknown"))


def test_first_request_can_report_a_caller_hold_before_any_model_result() -> None:
    fake = FakeResponsesClient(_response(_call()))
    proposer = _proposer(fake)
    result = proposer.propose(
        "task", _observation(), ProposalFeedback(None, "previous input was stale; held position")
    )
    assert isinstance(result, ProposalBatch)
    messages, _ = fake.requests[0]
    assert "previous input was stale" in json.dumps(messages)
    assert "hold (no candidate selected)" in json.dumps(messages)


def test_real_responses_transport_encodes_current_images_and_returns_raw_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response(_call()))

    action, observation = _spaces()
    proposer = AgentProposer(
        model="openai/gpt-6-astra", base_url="http://llm.test/v1", env={},
        action_space=action, observation_space=observation,
        transport=httpx.MockTransport(handler),
    )
    result = proposer.propose("task", _observation())
    assert isinstance(result, ProposalBatch)
    assert requests[0].url.path == "/v1/responses"
    body = json.loads(requests[0].content)
    assert body["store"] is False
    assert [tool["name"] for tool in body["tools"]] == ["propose_actions", "done", "give_up"]
    images = [part for message in body["input"] if isinstance(message.get("content"), list)
              for part in message["content"] if part["type"] == "input_image"]
    assert len(images) == 3
    assert all(part["image_url"].startswith("data:image/png;base64,") for part in images)
    proposer.close()


@pytest.mark.parametrize("raw,code", [
    ({"status": "completed", "output": [123]}, "invalid_response"),
    ({"status": "completed", "output": "bad"}, "invalid_response"),
    ({"status": "completed", "output": [_call()], "usage": {"input_tokens": -1}}, "invalid_usage"),
    (_response(_call("done", {"summary": "ok"})), "missing_field"),
])
def test_other_malformed_responses_are_failures(raw: dict[str, Any], code: str) -> None:
    result = _proposer(FakeResponsesClient(raw)).propose("task", _observation())
    assert isinstance(result, ProposalFailure)
    assert result.code == code and result.raw_response is raw


def test_structured_visual_estimates_predictions_and_embodiment_docs() -> None:
    fake = FakeResponsesClient(_response(_call(args=_visual_arguments())))
    proposer = _proposer(fake, embodiment_docs="left j0 positive moves toward the box")
    result = proposer.propose("task", _observation())
    assert isinstance(result, ProposalBatch)
    assert isinstance(result.visual_estimates["top_cam"], CameraVisualEstimate)
    assert isinstance(result.visual_estimates["top_cam"].target_object, VisualPoint)
    assert result.visual_estimates["top_cam"].target_object.source == "agent_unverified"
    assert isinstance(result.candidates[0].predicted_visual_deltas["left_cam"],
                      PredictedVisualDelta)
    assert result.candidates[0].acting_arm == "left"
    assert result.candidates[0].visual_effect == "closer"
    assert "left j0 positive" in fake.requests[0][0][0]["content"]


@pytest.mark.parametrize("mutate", [
    lambda a: a["visual_estimates"]["top_cam"]["target_object"].update(u=float("nan")),
    lambda a: a["visual_estimates"]["top_cam"]["target_object"].update(
        visible=False, u=0.2),
    lambda a: a["visual_estimates"].update(unknown_cam=a["visual_estimates"].pop("top_cam")),
])
def test_invalid_structured_visual_estimate_is_auditable(mutate: Any) -> None:
    args = _visual_arguments()
    mutate(args)
    result = _proposer(FakeResponsesClient(_response(_call(args=args)))).propose(
        "task", _observation())
    assert isinstance(result, ProposalFailure)
    assert result.code == "invalid_visual_estimate"
