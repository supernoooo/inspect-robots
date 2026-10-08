"""Offline Batch 04 loop: fake YAM, Agent, Choice and monotonic clock."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import Policy, Scene
from inspect_robots.compat import assert_compatible, check_compatibility
from inspect_robots.errors import ConfigError
from inspect_robots_agent.proposals import (
    CameraVisualEstimate, PredictedVisualDelta, ProposalBatch, ProposalFailure,
    ProposalTermination, VisualPoint,
)
from inspect_robots_jev import JevAgentPolicy
from inspect_robots_jev.agent_candidates import AgentCandidateValidator
from inspect_robots_jev.contract import InputError
from inspect_robots_jev.jev_choice import ChoiceError, ChoiceResult
from inspect_robots_yam.config import YamConfig

from test_agent_candidates import FakeCollisionChecker, FakeRig, candidate, observation


class Clock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Agent:
    def __init__(self, *results: object, clock: Clock | None = None,
                 cost: float = 0.0) -> None:
        self.results = list(results)
        self.calls = 0
        self.clock = clock
        self.cost = cost

    def reset(self) -> None:
        self.calls = 0

    def propose(self, task, obs):
        assert task == "place objects in box"
        assert set(obs.images) == {"top_cam", "left_cam", "right_cam"}
        self.calls += 1
        if self.clock:
            self.clock.now += self.cost
        return self.results.pop(0)


class Choice:
    def __init__(self, *results: object, clock: Clock | None = None,
                 cost: float = 0.0) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.clock = clock
        self.cost = cost

    def choose_generic(self, *, instruction, observation_context, candidates):
        assert instruction == "place objects in box"
        assert "preferred" not in observation_context
        self.calls.append([item.id for item in candidates])
        wire = json.dumps([item.summary for item in candidates])
        assert "preferred_id" not in wire
        assert "verified_prefix" not in wire
        if self.clock:
            self.clock.now += self.cost
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return ChoiceResult(result, None, "jev-1.13.0", self.cost)


def batch(*items, preferred: str | None = None):
    return ProposalBatch("scene", tuple(items), preferred or items[0].id,
                         "gpt-6-astra", 0.01, None, {})


def visual_batch(*items, gripper_u: float = 0.2, target_u: float = 0.6,
                 preferred: str | None = None):
    visible = lambda u, v=0.5: VisualPoint(True, u, v, 0.9)
    hidden = VisualPoint(False, None, None, 0.0)
    estimates = {
        name: CameraVisualEstimate(
            visible(target_u), visible(gripper_u), hidden, visible(0.9))
        for name in ("top_cam", "left_cam", "right_cam")
    }
    return replace(batch(*items, preferred=preferred), visual_estimates=estimates)


def visual_candidate(id, targets, du):
    return replace(
        candidate(id, targets), acting_arm="left", visual_effect="closer",
        prediction_confidence=0.8,
        verifiable_result="left gripper becomes closer to the target",
        predicted_visual_deltas={
            name: PredictedVisualDelta(du, 0.0, 0.8)
            for name in ("top_cam", "left_cam", "right_cam")
        },
    )


@pytest.fixture
def rig_and_validator(motion):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_left_base_pos=(0.0, 0.2, 0.2),
                    collision_right_base_pos=(0.0, -0.2, 0.2),
                    collision_left_base_yaw=0.0, collision_right_base_yaw=0.0,
                    collision_table_height=0.0)
    rig = FakeRig(cfg)
    validator = AgentCandidateValidator(
        rig, motion, collision_checker=FakeCollisionChecker(cfg),
        max_image_age_s=0.5, max_skew_s=0.2)
    return rig, validator


def make_policy(rig_and_validator, agent, choice, clock, *, selector="jev",
                max_dispatch_age_s=0.5, inference_budget_s=1.0):
    rig, validator = rig_and_validator
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, max_image_age_s=0.5,
        max_skew_s=0.2, max_dispatch_age_s=max_dispatch_age_s,
        inference_budget_s=inference_budget_s, selector=selector,
        validation_mode="measured",
        freshness_mode="capture_age",
        proposer=agent, validator=validator, choice=choice, clock=clock)
    assert policy.pairing_preflight(rig).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    return policy


def assert_hold(chunk, q, reason):
    assert len(chunk) == 1
    assert chunk.meta["kind"] == "hold" and chunk.meta["reason"] == reason
    np.testing.assert_array_equal(chunk.actions[0].data, q)
    assert chunk.inference_latency_s is not None


def test_two_observations_three_proposals_filter_choice_and_short_prefix(rig_and_validator, q0):
    clock = Clock()
    first = batch(candidate("left", {"left_j0": 0.2}),
                  candidate("right", {"right_j1": 0.2}),
                  candidate("bad", {"left_bad": 0.1}), preferred="left")
    second = batch(candidate("next", {"left_j0": 0.15}),
                   candidate("other", {"right_j0": 0.1}))
    agent = Agent(first, second, clock=clock, cost=0.02)
    choice = Choice("right", "next", clock=clock, cost=0.03)
    policy = make_policy(rig_and_validator, agent, choice, clock)
    one = policy.act(observation(q0))
    assert len(one) == 3 and one.inference_latency_s == pytest.approx(0.05)
    assert one.meta["proposed_ids"] == ["left", "right", "bad"]
    assert one.meta["candidate_ids"] == ["left", "right"]
    assert one.meta["filtered"][0]["id"] == "bad"
    assert one.meta["selected_id"] == "right"
    assert one.meta["decision_mode"] == "ranking"
    assert choice.calls[0] == ["left", "right", "hold"]
    assert one.actions[0].data[8] > q0[8] and one.actions[0].data[0] == 0
    assert all(action.data[0] == 0 for action in one.actions)
    assert policy.config.action_horizon == 3 and policy.config.replan_interval is None
    row = policy.audit_records[0]
    assert row["decision_id"] == one.meta["decision_id"]
    assert row["dispatch_status"] == "selected_for_dispatch"
    assert row["selected_prefix"] == [action.data.tolist() for action in one.actions]
    assert [item["id"] for item in row["candidates"]] == ["left", "right"]
    clock.now = 10.2
    two = policy.act(observation(one.actions[-1].data, stamp=10.1))
    assert two.meta["selected_id"] == "next" and len(two) == 3
    assert choice.calls[1] == ["next", "other", "hold"]
    assert agent.calls == 2


@pytest.mark.parametrize("selector", ["jev", "agent_preferred"])
def test_same_filter_for_both_selectors(rig_and_validator, q0, selector):
    clock = Clock()
    proposal = batch(candidate("bad", {"left_bad": 0.1}),
                     candidate("good", {"left_j0": 0.2}), preferred="bad")
    choice = Choice("good")
    policy = make_policy(rig_and_validator, Agent(proposal), choice, clock,
                         selector=selector)
    result = policy.act(observation(q0))
    assert result.meta["candidate_ids"] == ["good"]
    assert result.meta["decision_mode"] == "gate"
    if selector == "jev":
        assert result.meta["selected_id"] == "good"
        assert choice.calls == [["good", "hold"]]
    else:
        assert_hold(result, q0, "preferred_filtered")
        assert choice.calls == []


@pytest.mark.parametrize(("fault", "reason"), [
    (ChoiceError("timeout"), "jev_timeout"),
    ("unknown", "jev_unknown_candidate_id"),
    ("hold", "jev_hold"),
])
def test_jev_faults_and_hold(rig_and_validator, q0, fault, reason):
    clock = Clock()
    proposal = batch(candidate("one", {"left_j0": 0.2}))
    policy = make_policy(rig_and_validator, Agent(proposal), Choice(fault), clock)
    result = policy.act(observation(q0))
    assert_hold(result, q0, reason)
    assert policy.audit_records[-1]["reason"] == reason


def test_dispatch_age_and_inference_budget(rig_and_validator, q0):
    proposal = batch(candidate("one", {"left_j0": 0.2}))
    clock = Clock()
    choice = Choice("one", clock=clock, cost=0.25)
    policy = make_policy(rig_and_validator, Agent(proposal), choice, clock,
                         max_dispatch_age_s=0.05)
    assert_hold(policy.act(observation(q0)), q0, "dispatch_stale")
    assert choice.calls == []

    clock = Clock()
    choice = Choice("one", clock=clock, cost=0.25)
    policy = make_policy(rig_and_validator, Agent(proposal), choice, clock,
                         max_dispatch_age_s=0.3)
    assert_hold(policy.act(observation(q0)), q0, "dispatch_stale")
    assert choice.calls == [["one", "hold"]]

    clock = Clock()
    agent = Agent(proposal, clock=clock, cost=0.2)
    policy = make_policy(rig_and_validator, agent, Choice("one"), clock,
                         inference_budget_s=0.1)
    assert_hold(policy.act(observation(q0)), q0, "inference_timeout")
    assert policy.audit_records[-1]["inference_budget_s"] == 0.1


def test_inference_latency_uses_dispatch_age_not_capture_age(rig_and_validator, q0):
    clock = Clock()
    proposal = batch(candidate("one", {"left_j0": 0.2}))
    policy = make_policy(
        rig_and_validator, Agent(proposal, clock=clock, cost=0.6),
        Choice("one"), clock, max_dispatch_age_s=2.0,
        inference_budget_s=2.0)
    result = policy.act(observation(q0))
    assert result.meta["selected_id"] == "one"
    assert result.meta.get("kind") != "hold"


def test_missing_threshold_empty_candidates_and_invalid_state(rig_and_validator, q0):
    clock = Clock()
    proposal = batch(candidate("bad", {"left_bad": 0.1}))
    agent = Agent(proposal)
    choice = Choice("bad")
    policy = make_policy(rig_and_validator, agent, choice, clock,
                         max_dispatch_age_s=None)
    assert_hold(policy.act(observation(q0)), q0, "dispatch_age_unconfigured")
    assert agent.calls == 0
    policy = make_policy(rig_and_validator, Agent(proposal), choice, clock)
    assert_hold(policy.act(observation(q0)), q0, "no_safe_candidate")
    assert choice.calls == []
    bad = observation(q0)
    bad.state["joint_pos"][0] = np.nan
    with pytest.raises(InputError) as caught:
        policy.act(bad)
    assert caught.value.code == "invalid_joint_pos"


def test_missing_threshold_needs_no_agent_credentials(rig_and_validator, q0, monkeypatch):
    rig, validator = rig_and_validator
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    policy = JevAgentPolicy(cam_height=101, cam_width=101, validator=validator,
                            max_image_age_s=0.5, max_dispatch_age_s=None,
                            freshness_mode="capture_age", clock=Clock())
    assert policy.pairing_preflight(rig).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    assert_hold(policy.act(observation(q0)), q0, "dispatch_age_unconfigured")
    with pytest.raises(ValueError, match="another embodiment"):
        policy.pairing_preflight(FakeRig(rig._cfg))


def test_actual_rig_bounds_reject_state_even_on_hold_path(rig_and_validator, motion, q0):
    original, _ = rig_and_validator
    cfg = replace(original._cfg,
                  joint_high=(0.05, *original._cfg.joint_high[1:]))
    rig = FakeRig(cfg)
    validator = AgentCandidateValidator(
        rig, motion, collision_checker=FakeCollisionChecker(cfg),
        max_image_age_s=0.5, max_skew_s=0.2)
    policy = JevAgentPolicy(cam_height=101, cam_width=101, validator=validator,
                            max_image_age_s=0.5, max_dispatch_age_s=None,
                            clock=Clock())
    assert policy.pairing_preflight(rig).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    q = q0.copy()
    q[0] = 0.1  # Legal in the generic YAM wire, outside this rig's bound.
    for stale_camera in (False, True):
        source = observation(q)
        if stale_camera:
            source.image_times["top_cam"] = 8.0
        with pytest.raises(InputError) as caught:
            policy.act(source)
        assert caught.value.code == "invalid_joint_pos"
        assert policy.audit_records[-1]["reason"] == "invalid_joint_pos"


@pytest.mark.parametrize("status", ["done", "give_up"])
def test_termination_skips_jev_and_requests_stop(rig_and_validator, q0, status):
    termination = ProposalTermination(status, "model says stop", "hindsight",
                                      "gpt-6-astra", 0.01, None, {})
    required = 2 if status == "done" else 3
    choice = Choice("hold")
    policy = make_policy(rig_and_validator, Agent(*([termination] * required)), choice, Clock())
    for index in range(required - 1):
        pending = policy.act(observation(q0, stamp=9.8 + index / 100))
        assert_hold(pending, q0, status + "_confirmation_pending")
    result = policy.act(observation(q0, stamp=9.8 + (required - 1) / 100))
    assert_hold(result, q0, status)
    assert result.actions[0].meta["request_stop"] is True
    assert result.actions[0].meta["stop_reason"] == status
    assert result.actions[0].meta["stop_detail"] == "model says stop"
    assert bool(result.actions[0].meta.get("operator_review_required")) is (status == "give_up")
    assert choice.calls == []
    assert policy.audit_records[-1]["termination"]["status"] == status


def test_proposal_failure_and_duplicate_observation(rig_and_validator, q0):
    failure = ProposalFailure("request_error", "timeout", "gpt-6-astra", 0.1, None, None)
    policy = make_policy(rig_and_validator, Agent(failure), Choice("hold"), Clock())
    assert_hold(policy.act(observation(q0)), q0, "agent_request_error")
    assert_hold(policy.act(observation(q0)), q0, "observation_not_new")


def test_registry_four_policies_and_strict_pairing(rig_and_validator):
    names = {ep.name: ep for ep in entry_points(group="inspect_robots.policies")}
    assert {"agent", "jev-direct", "jev-hybrid", "jev-agent"} <= names.keys()
    for name in ("agent", "jev-direct", "jev-hybrid", "jev-agent"):
        assert callable(names[name].load())
    rig, validator = rig_and_validator
    fixtures = Path(__file__).parent / "fixtures"
    for name in ("jev-direct", "jev-hybrid"):
        legacy = names[name].load()(cam_height=101, cam_width=101,
                                    calibration_path=fixtures / "synthetic_calibration_v1.json",
                                    mjcf_path=fixtures / "tiny_yam.xml")
        assert isinstance(legacy, Policy)
        assert_compatible(legacy, rig)
    agent = names["agent"].load()(model="openai/gpt-6-astra",
                                    base_url="http://llm.test/v1", wire="responses",
                                    wire_capture=False, env={})
    agent.bind(rig.info)
    assert isinstance(agent, Policy)
    assert_compatible(agent, rig)
    policy = names["jev-agent"].load()(
        cam_height=101, cam_width=101, proposer=Agent(), validator=validator,
        choice=Choice(), clock=Clock(), max_dispatch_age_s=0.5)
    assert isinstance(policy, Policy)
    assert check_compatibility(policy, rig).ok
    assert_compatible(policy, rig)
    assert policy.info.action_space.shape == (14,)
    assert policy.config.replan_interval is None


def test_reject_unsupported_model_and_wire():
    with pytest.raises(ValueError, match="responses"):
        JevAgentPolicy(wire="chat")
    with pytest.raises(ValueError, match="gpt-6-astra"):
        JevAgentPolicy(model="openai/gpt-5")


def test_forwards_effort_to_agent_proposer(rig_and_validator, monkeypatch):
    import inspect_robots_jev.agent_policy as policy_module

    rig, validator = rig_and_validator
    rig.info = replace(rig.info, docs="real YAM joint directions and gripper polarity")
    captured = {}
    monkeypatch.setattr(policy_module, "resolve_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "ResponsesClient", lambda *args, **kwargs: object())

    def proposer_factory(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr(policy_module, "AgentProposer", proposer_factory)
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, validator=validator,
        selector="agent_preferred", max_dispatch_age_s=0.5,
        effort="medium", max_speed_frac=0.25)
    assert policy.pairing_preflight(rig).ok
    assert captured["effort"] == "medium"
    assert captured["candidate_count"] == 3
    assert captured["embodiment_docs"] == "real YAM joint directions and gripper polarity"


def test_reject_bad_speed_fraction_and_effort():
    with pytest.raises(ValueError, match="max_speed_frac"):
        JevAgentPolicy(max_speed_frac=0)
    with pytest.raises(ValueError, match="max_speed_frac.*''"):
        JevAgentPolicy(max_speed_frac="")
    with pytest.raises(ValueError, match="max_dispatch_age_s.*''"):
        JevAgentPolicy(max_dispatch_age_s="")
    with pytest.raises(ConfigError, match="effort"):
        JevAgentPolicy(effort="unlimited")
    with pytest.raises(ValueError, match="validation_mode"):
        JevAgentPolicy(validation_mode="unknown")


def test_yam_mode_runs_without_calibration_or_mjcf(q0):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_guardrail=False, collision_table=False)
    rig = FakeRig(cfg)
    choice = RecordingChoice("two")
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, validation_mode="yam",
        max_image_age_s=0.5, max_dispatch_age_s=0.5,
        proposer=Agent(batch(candidate("one", {"left_j0": 0.2}),
                             candidate("two", {"right_j0": 0.2}))),
        choice=choice, clock=Clock())
    assert policy.pairing_preflight(rig).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    result = policy.act(observation(q0))
    assert_hold(result, q0, "missing_frame_ids")
    source = replace(observation(q0, stamp=10.1), extra={"camera_frame_ids": {
        name: (1, 1) for name in ("top_cam", "left_cam", "right_cam")}})
    assert_hold(policy.act(source), q0, "awaiting_fresh_frames")
    fresh = replace(observation(q0, stamp=10.2), extra={"camera_frame_ids": {
        name: (1, 2) for name in ("top_cam", "left_cam", "right_cam")}})
    selected = policy.act(fresh)
    assert selected.meta["selected_id"] == "two"
    assert len(selected.actions) == 3
    assert selected.actions[0].data[7] > q0[7]
    assert choice.calls == [["one", "two", "hold"]]
    assert choice.menus[0]["two"]["risk"] == "path_not_checked"
    row = policy.audit_records[-1]
    assert row["selected_for_dispatch"] == "two"
    assert row["dispatch_status"] == "selected_for_dispatch"
    assert row["validation_mode"] == "yam"
    assert row["rig_fingerprint"]["mjcf_sha256"] is None
    assert all(item["summary"]["local_checks"]["full_trajectory"] == "not_checked"
               for item in row["candidates"])
    assert all(item["summary"]["local_checks"]["interpolated_collision"] == "not_checked"
               for item in row["candidates"])

    default_yam = JevAgentPolicy(cam_height=101, cam_width=101,
                                 proposer=Agent(), clock=Clock())
    assert default_yam.pairing_preflight(rig).ok


def test_yam_mode_accepts_old_skewed_and_repeated_capture_times(q0):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_guardrail=False, collision_table=False)
    rig = FakeRig(cfg)
    clock = Clock()
    proposal = batch(candidate("one", {"left_j0": 0.2}))
    agent = Agent(proposal)
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, validation_mode="yam",
        max_image_age_s=0.5, max_skew_s=0.2, max_dispatch_age_s=0.5,
        proposer=agent, choice=Choice("one"), clock=clock)
    assert policy.pairing_preflight(rig).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    old = observation(q0, stamp=1.0)
    old.image_times["left_cam"] = 3.0
    ids = lambda number: {name: (1, number) for name in ("top_cam", "left_cam", "right_cam")}
    assert_hold(policy.act(replace(old, extra={"camera_frame_ids": ids(1)})), q0,
                "awaiting_fresh_frames")
    selected = policy.act(replace(old, state_time=1.1,
                                  extra={"camera_frame_ids": ids(2)}))
    assert selected.meta["selected_id"] == "one"
    assert selected.actions[0].data[0] > q0[0]
    assert agent.calls == 1
    assert policy.audit_records[0]["observed_state"] is not None
    row = policy.audit_records[-1]
    assert row["observation_freshness"] == "frame_sequence"
    assert row["dispatch_age_basis"] == "decision_start"
    assert row["rig_fingerprint"]["observation_age_limit_s"] is None
    assert_hold(policy.act(replace(old, state_time=1.2,
                                   extra={"camera_frame_ids": ids(2)})), q0,
                "observation_not_new")
    assert_hold(policy.act(replace(old, state_time=float("nan"))), q0, "invalid_time")
    assert agent.calls == 1


def test_yam_mode_still_limits_time_spent_on_decision(q0):
    cfg = YamConfig(cam_height=101, cam_width=101,
                    collision_guardrail=False, collision_table=False)
    clock = Clock()
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, validation_mode="yam",
        max_dispatch_age_s=0.5, inference_budget_s=2.0,
        proposer=Agent(batch(candidate("one", {"left_j0": 0.2})),
                       clock=clock, cost=0.6),
        choice=Choice("one"), clock=clock)
    assert policy.pairing_preflight(FakeRig(cfg)).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    ids = lambda number: {name: (1, number) for name in ("top_cam", "left_cam", "right_cam")}
    assert_hold(policy.act(replace(observation(q0, stamp=1.0),
                                   extra={"camera_frame_ids": ids(1)})), q0,
                "awaiting_fresh_frames")
    assert_hold(policy.act(replace(observation(q0, stamp=1.1),
                                   extra={"camera_frame_ids": ids(2)})), q0,
                "dispatch_stale")


def test_yam_mode_accepts_openai_key_for_responses_without_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = YamConfig(cam_height=101, cam_width=101)
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, validation_mode="yam",
        selector="agent_preferred", max_dispatch_age_s=30.0)
    assert policy.pairing_preflight(FakeRig(cfg)).ok
    assert policy._proposer is not None
    assert policy._proposer.model == "gpt-6-astra"
    assert policy._proposer._client._provider.base_url == "https://api.openai.com/v1"


def _frame_observation(q, number, *, stamp=1.0):
    source = observation(q, stamp=stamp)
    source.extra["camera_frame_ids"] = {
        name: (7, number) for name in ("top_cam", "left_cam", "right_cam")}
    return source


def _frame_policy(rig_and_validator, motion, agent, choice, clock):
    rig, _ = rig_and_validator
    validator = AgentCandidateValidator(
        rig, motion, collision_checker=FakeCollisionChecker(rig._cfg),
        max_image_age_s=None, max_skew_s=None)
    policy = JevAgentPolicy(
        cam_height=101, cam_width=101, max_dispatch_age_s=1.0,
        validation_mode="measured",
        proposer=agent, validator=validator, choice=choice, clock=clock)
    assert policy.pairing_preflight(rig).ok
    policy.reset(Scene(id="scene", instruction="place objects in box"))
    return policy


class RecordingChoice(Choice):
    def __init__(self, *results):
        super().__init__(*results)
        self.contexts = []
        self.menus = []

    def choose_generic(self, *, instruction, observation_context, candidates):
        self.contexts.append(json.loads(observation_context))
        self.menus.append({item.id: item.summary for item in candidates})
        return super().choose_generic(instruction=instruction,
                                      observation_context=observation_context,
                                      candidates=candidates)


def test_old_timestamps_new_frames_and_approach_feedback_reach_jev(
        rig_and_validator, motion, q0):
    first = replace(batch(candidate("first", {"left_j0": 0.2})),
                    phase="approach", visual_relation="far", object_state="on_table")
    second = replace(batch(candidate("second", {"left_j0": 0.3})),
                     phase="approach", visual_relation="near", object_state="on_table")
    choice = RecordingChoice("first", "second")
    policy = _frame_policy(rig_and_validator, motion, Agent(first, second), choice, Clock())
    assert_hold(policy.act(_frame_observation(q0, 1, stamp=1.0)), q0,
                "awaiting_fresh_frames")
    source = _frame_observation(q0, 2, stamp=1.1)
    source.image_times["left_cam"] = 3.0  # old and skewed, still finite
    motion_chunk = policy.act(source)
    assert motion_chunk.meta["selected_id"] == "first"
    assert choice.contexts[0]["phase"] == "approach"
    assert choice.contexts[0]["object_state"] == "on_table"
    assert "normally remains still" in choice.contexts[0]["guidance"]
    assert choice.menus[0]["first"]["expected_visual_change"]
    assert "path_checked" == choice.menus[0]["first"]["risk"]
    assert "joint_delta" not in json.dumps(choice.menus[0])
    next_q = motion_chunk.actions[-1].data
    policy.act(_frame_observation(next_q, 3, stamp=1.2))
    assert choice.contexts[1]["visual_relation"] == "near"
    assert choice.contexts[1]["previous_result"]["execution_status"] == "observed_after_dispatch"
    assert choice.contexts[1]["previous_result"]["max_joint_error"] == 0
    assert policy.audit_records[-1]["no_progress_rounds"] == 0
    assert policy.audit_records[-1]["progress_evidence"] == {
        "evaluated": True, "joint_progress": True,
        "reported_visual_progress": True, "counts_as_progress": True}


def test_frame_mode_rejects_missing_repeated_and_restarted_frames(
        rig_and_validator, motion, q0):
    policy = _frame_policy(rig_and_validator, motion,
                           Agent(batch(candidate("one", {"left_j0": 0.2}))),
                           Choice("one"), Clock())
    assert_hold(policy.act(observation(q0, stamp=1.0)), q0, "missing_frame_ids")
    assert_hold(policy.act(_frame_observation(q0, 1, stamp=1.0)), q0,
                "awaiting_fresh_frames")
    assert_hold(policy.act(_frame_observation(q0, 1, stamp=1.1)), q0,
                "observation_not_new")
    restarted = _frame_observation(q0, 2, stamp=1.2)
    restarted.extra["camera_frame_ids"]["left_cam"] = (8, 2)
    assert_hold(policy.act(restarted), q0, "observation_not_new")


def test_three_jev_holds_request_operator_review(rig_and_validator, motion, q0):
    proposals = [batch(candidate("one", {"left_j0": 0.2})) for _ in range(3)]
    policy = _frame_policy(rig_and_validator, motion, Agent(*proposals),
                           Choice("hold", "hold", "hold"), Clock())
    policy.act(_frame_observation(q0, 1))
    for number in (2, 3):
        assert_hold(policy.act(_frame_observation(q0, number, stamp=1 + number / 10)),
                    q0, "jev_hold")
    stopped = policy.act(_frame_observation(q0, 4, stamp=1.4))
    assert_hold(stopped, q0, "stalled_review")
    assert stopped.actions[0].meta["request_stop"] is True
    assert policy.audit_records[-1]["consecutive_jev_holds"] == 3


def test_three_motion_rounds_without_visual_progress_stop_after_observation(
        rig_and_validator, motion, q0):
    proposals = [replace(batch(candidate(f"step{index}", {"left_j0": index / 10})),
                         phase="approach", visual_relation="far",
                         object_state="on_table") for index in (1, 2, 3, 4)]
    choice = Choice("step1", "step2", "step3")
    policy = _frame_policy(rig_and_validator, motion, Agent(*proposals), choice, Clock())
    policy.act(_frame_observation(q0, 1))
    measured = q0
    for number in (2, 3, 4):
        result = policy.act(_frame_observation(measured, number, stamp=1 + number / 10))
        assert result.meta["selected_id"] == f"step{number - 1}"
        measured = result.actions[-1].data
    stopped = policy.act(_frame_observation(measured, 5, stamp=1.5))
    assert_hold(stopped, measured, "stalled_review")
    assert stopped.actions[0].meta["request_stop"] is True
    assert policy.audit_records[-1]["no_progress_rounds"] == 3
    assert len(choice.calls) == 3


def test_visual_error_prediction_reaches_jev_without_joint_targets(
        rig_and_validator, motion, q0):
    proposal = visual_batch(
        visual_candidate("closer", {"left_j0": 0.2}, 0.1),
        visual_candidate("farther", {"left_j1": 0.2}, -0.1),
    )
    choice = RecordingChoice("closer")
    policy = _frame_policy(rig_and_validator, motion, Agent(proposal), choice, Clock())
    policy.act(_frame_observation(q0, 1))
    result = policy.act(_frame_observation(q0, 2, stamp=1.2))
    assert result.meta["selected_id"] == "closer"
    menu = choice.menus[0]
    assert menu["closer"]["predicted_error_change"]["top_cam"]["left"][
        "error_reduction"] > 0
    assert menu["farther"]["predicted_error_change"]["top_cam"]["left"][
        "error_reduction"] < 0
    wire = json.dumps({"context": choice.contexts[0], "menu": menu})
    assert "preferred_id" not in wire and "verified_prefix" not in wire
    assert "targets" not in wire and "joint_pos" not in wire
    assert choice.contexts[0]["normalized_visual_points"]["top_cam"][
        "target_object"]["source"] == "agent_unverified"


def test_invalid_visual_contract_holds_before_jev(rig_and_validator, motion, q0):
    proposal = visual_batch(visual_candidate("move", {"left_j0": 0.2}, 0.1))
    invalid = dict(proposal.visual_estimates)
    invalid["top_cam"] = {
        "target_object": {"visible": False, "u": 0.4, "v": None, "confidence": 0.8},
        "left_gripper": {"visible": True, "u": 0.2, "v": 0.5, "confidence": 0.8},
        "right_gripper": {"visible": False, "u": None, "v": None, "confidence": 0.0},
        "placement_region": {"visible": True, "u": 0.9, "v": 0.5, "confidence": 0.8},
    }
    choice = RecordingChoice("move")
    policy = _frame_policy(
        rig_and_validator, motion, Agent(replace(proposal, visual_estimates=invalid)),
        choice, Clock())
    policy.act(_frame_observation(q0, 1))
    result = policy.act(_frame_observation(q0, 2, stamp=1.2))
    assert_hold(result, q0, "agent_invalid_visual_estimate")
    assert choice.calls == []


def test_stationary_object_with_closer_gripper_counts_as_progress(
        rig_and_validator, motion, q0):
    first = visual_batch(
        visual_candidate("first", {"left_j0": 0.2}, 0.1), gripper_u=0.2, target_u=0.6)
    second = visual_batch(
        visual_candidate("second", {"left_j0": 0.3}, 0.1), gripper_u=0.35, target_u=0.6)
    policy = _frame_policy(rig_and_validator, motion, Agent(first, second),
                           Choice("first", "second"), Clock())
    policy.act(_frame_observation(q0, 1))
    motion_chunk = policy.act(_frame_observation(q0, 2, stamp=1.2))
    policy.act(_frame_observation(motion_chunk.actions[-1].data, 3, stamp=1.3))
    row = policy.audit_records[-1]
    assert row["progress_evidence"]["counts_as_progress"] is True
    assert row["progress_evidence"]["visual_error_reduction"] > 0.02
    assert row["no_progress_rounds"] == 0
    executed = policy.audit_records[-2]
    assert executed["actual_visual_change"]["top_cam"]["target_object"]["distance"] == 0
    assert executed["prediction_consistency"]["status"] == "consistent"


@pytest.mark.parametrize(("status", "required"), [("give_up", 3), ("done", 2)])
def test_termination_requires_fresh_frame_confirmation(
        rig_and_validator, motion, q0, status, required):
    terminations = [ProposalTermination(status, "model says stop", "hindsight",
                                        "gpt-6-astra", 0.01, None, {})
                    for _ in range(required)]
    policy = _frame_policy(rig_and_validator, motion, Agent(*terminations),
                           Choice(), Clock())
    policy.act(_frame_observation(q0, 1))
    for index in range(1, required):
        pending = policy.act(_frame_observation(q0, index + 1, stamp=1 + index / 10))
        assert_hold(pending, q0, status + "_confirmation_pending")
        assert pending.actions[0].meta.get("request_stop") is None
    stopped = policy.act(_frame_observation(q0, required + 1,
                                            stamp=1 + required / 10))
    assert_hold(stopped, q0, status)
    assert stopped.actions[0].meta["request_stop"] is True
    assert policy.audit_records[-1]["termination"]["confirmation_count"] == required


def test_three_real_transitions_fit_local_visual_direction_mapping(
        rig_and_validator, motion, q0):
    proposals = [
        visual_batch(
            visual_candidate(f"step{index}", {"left_j0": index / 10}, 0.05),
            gripper_u=0.15 + index * 0.05, target_u=0.7,
        )
        for index in range(1, 5)
    ]
    choice = RecordingChoice(*(f"step{index}" for index in range(1, 5)))
    policy = _frame_policy(rig_and_validator, motion, Agent(*proposals), choice, Clock())
    policy.act(_frame_observation(q0, 1))
    measured = q0
    for frame in range(2, 6):
        chunk = policy.act(_frame_observation(measured, frame, stamp=1 + frame / 10))
        measured = chunk.actions[-1].data
    assert policy.audit_records[-2]["local_visual_mapping"]["top_cam:left"] == {
        "status": "estimated", "sample_count": 3, "source": "observed_real_motion"}
    assert choice.menus[-1]["step4"]["history_direction_consistency"][
        "status"] == "consistent"
    assert len(policy._visual_history[("top_cam", "left")]) == 3
