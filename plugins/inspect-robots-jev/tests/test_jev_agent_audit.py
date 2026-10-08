"""Batch 05: dispatch requests, controller reports and later measurements."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots import Scene
from inspect_robots_agent.proposals import ProposalFailure, ProposalFeedback, ProposalTermination
from inspect_robots_jev.audit import load_decision
from inspect_robots_jev.audit import save_decision as real_save_decision
from inspect_robots_jev.agent_policy import JevAgentPolicy

from test_agent_candidates import candidate, observation
from test_jev_agent_policy import Choice, Clock, batch, make_policy, rig_and_validator


class FeedbackAgent:
    def __init__(self, *results):
        self.results = list(results)
        self.feedback: list[ProposalFeedback | None] = []

    def reset(self):
        self.feedback.clear()

    def propose(self, task, obs, feedback=None):
        assert task == "place objects in box"
        assert set(obs.images) == {"top_cam", "left_cam", "right_cam"}
        assert not obs.extra
        self.feedback.append(feedback)
        return self.results.pop(0)


def _record(policy: JevAgentPolicy, tmp_path: Path, steps=(), **kwargs):
    record = SimpleNamespace(
        policy_transcript=policy.transcript(), steps=list(steps), metadata={},
        status=kwargs.get("status", "success"), error=kwargs.get("error"),
        termination_reason=kwargs.get("termination_reason"),
        operator_judgement=kwargs.get("operator_judgement"),
    )
    policy.on_trial_end(record, str(tmp_path), "run")
    return record


def _frames(tmp_path: Path, obs):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(exist_ok=True)
    refs = {}
    for name in obs.images:
        path = frame_dir / f"{obs.state_time}-{name}.png"
        path.write_bytes(b"frame-ref")
        refs[name] = SimpleNamespace(path=str(path))
    return SimpleNamespace(observation=obs, image_refs=refs)


def test_selected_and_unselected_trace_to_new_measurement(
    rig_and_validator, q0, tmp_path: Path
):
    clock = Clock()
    agent = FeedbackAgent(
        batch(candidate("left", {"left_j0": 0.2}),
              candidate("right", {"right_j0": 0.2}),
              candidate("bad", {"left_bad": 0.1})),
        batch(candidate("next", {"left_j0": 0.1})),
    )
    policy = make_policy(rig_and_validator, agent, Choice("right", "hold"), clock)
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first_obs = replace(observation(q0), extra={"env_step": 0})
    first = policy.act(first_obs)
    row = load_decision(tmp_path, first.meta["decision_id"])
    assert row["execution"]["status"] == "awaiting_observation"
    assert row["selected_for_dispatch"] == "right"
    assert [c["status"] for c in row["proposed"]] == ["proposed"] * 3
    assert [c["status"] for c in row["candidates"]] == ["proposed"] * 2
    assert row["filtered"][0]["id"] == "bad"
    assert row["task"]["instruction"] == "place objects in box"
    assert row["rig_fingerprint"]["rig_config_sha256"]
    assert row["selected_prefix"] == [a.data.tolist() for a in first.actions]
    assert row["observed_state"] is None

    clock.now = 10.2
    second_obs = replace(observation(first.actions[-1].data, stamp=10.1),
                         extra={"env_step": 3, "approvals": []})
    policy.act(second_obs)
    updated = load_decision(tmp_path, first.meta["decision_id"])
    assert updated["execution"]["status"] == "observed_after_dispatch"
    assert updated["execution"]["max_joint_error_to_requested"] == pytest.approx(0)
    assert updated["observed_state"]["joint_pos"] == first.actions[-1].data.tolist()
    assert updated["approval"]["status"] == "none_reported"
    assert agent.feedback[0] is None
    assert agent.feedback[1].selected_id == "right"
    assert '"selection":"right"' in agent.feedback[1].outcome
    assert "operator judges completion" in agent.feedback[1].outcome
    assert '"joint_delta"' in agent.feedback[1].outcome
    assert "bad" not in agent.feedback[1].outcome

    record = _record(policy, tmp_path, [_frames(tmp_path, first_obs), _frames(tmp_path, second_obs)])
    index = record.metadata["jev_audit"]
    assert index["missing_files"] == index["missing_frames"] == index["write_failures"] == 0
    assert not index["incomplete"]
    assert len(load_decision(tmp_path, first.meta["decision_id"])["observed_state"]["frame_refs"]) == 3
    assert record.policy_transcript[0]["decision_id"] == first.meta["decision_id"]


@pytest.mark.parametrize(("detail", "expected"), [
    ("delta_clamped", "delta_clamped"),
    ("clamped", "clamped"),
    ("rejected", "modified"),
])
def test_yam_report_is_explicit_and_does_not_prove_execution(
    rig_and_validator, q0, detail, expected
):
    clock = Clock()
    agent = FeedbackAgent(batch(candidate("one", {"left_j0": 0.2})),
                          batch(candidate("two", {"left_j0": 0.1})))
    policy = make_policy(rig_and_validator, agent, Choice("one", "hold"), clock)
    policy.act(replace(observation(q0), extra={"env_step": 0}))
    clock.now = 10.2
    policy.act(replace(observation(q0, stamp=10.1),
                       extra={"env_step": 3, "approvals": [{"t": 1, "detail": detail}]}))
    row = policy.audit_records[0]
    assert row["approval"] == {"status": "modified", "events": [{"t": 1, "detail": expected}]}
    assert row["execution"]["status"] == "observed_after_dispatch"
    assert row["execution"]["max_joint_error_to_requested"] > 0
    assert expected in agent.feedback[1].outcome
    assert '"joint_delta":[0.0' in agent.feedback[1].outcome


def test_early_stop_and_safety_abort_leave_dispatch_unverified(
    rig_and_validator, q0, tmp_path: Path
):
    policy = make_policy(rig_and_validator,
                         FeedbackAgent(batch(candidate("one", {"left_j0": 0.2}))),
                         Choice("one"), Clock())
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first = policy.act(observation(q0))
    record = _record(policy, tmp_path, status="error", error="SafetyAbort: rejected by YAM")
    row = load_decision(tmp_path, first.meta["decision_id"])
    assert row["execution"]["status"] == "unverified_no_next_observation"
    assert row["observed_state"] is None
    assert row["trial_end"]["abort_type"] == "SafetyAbort"
    assert record.metadata["jev_audit"]["unverified_decisions"] == 1


def test_trial_trace_distinguishes_requested_from_reviewed_action(
    rig_and_validator, q0, tmp_path: Path
):
    clock = Clock()
    agent = FeedbackAgent(batch(candidate("one", {"left_j0": 0.2})),
                          batch(candidate("next", {"left_j0": 0.1})))
    policy = make_policy(rig_and_validator, agent, Choice("one", "hold"), clock)
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first_obs = replace(observation(q0), extra={"env_step": 0})
    first = policy.act(first_obs)
    changed = first.actions[0].data.copy()
    changed[0] = 0.0
    clock.now = 10.2
    second_obs = replace(observation(q0, stamp=10.1), extra={
        "env_step": 3, "approvals": [{"t": 0, "detail": "delta_clamped"}]})
    policy.act(second_obs)
    assert "requested prefix may differ" in agent.feedback[1].outcome
    steps = []
    for t, action in enumerate(first.actions):
        step = _frames(tmp_path, first_obs) if t == 0 else SimpleNamespace(
            observation=first_obs, image_refs={})
        step.t = t
        step.action = SimpleNamespace(data=changed if t == 0 else action.data,
                                      meta={"decision_id": first.meta["decision_id"]})
        steps.append(step)
    record = _record(policy, tmp_path, steps)
    row = load_decision(tmp_path, first.meta["decision_id"])
    trace = row["controller_trace"]
    assert trace["status"] == "reviewed_action_recorded"
    assert trace["recorded_steps"] == trace["requested_steps"] == 3
    assert trace["changed_steps"] == [0]
    assert trace["reviewed_prefix"][0] == changed.tolist()
    assert trace["reviewed_prefix"][0] != row["selected_prefix"][0]
    assert row["execution"]["status"] == "observed_after_dispatch"
    assert record.metadata["jev_audit"]["missing_frames"] >= 1


def test_hold_and_failure_are_audited_without_motion_claim(rig_and_validator, q0):
    clock = Clock()
    agent = FeedbackAgent(ProposalFailure("request_error", "timeout", "gpt", 0.1, None, None),
                          batch(candidate("later", {"left_j0": 0.1})))
    policy = make_policy(rig_and_validator, agent, Choice("hold"), clock)
    first = policy.act(observation(q0))
    assert first.meta["kind"] == "hold"
    assert policy.audit_records[0]["execution"]["status"] == "not_requested"
    clock.now = 10.2
    policy.act(observation(q0, stamp=10.1))
    assert agent.feedback[1].selected_id is None
    assert "agent_request_error" in agent.feedback[1].outcome
    assert policy.audit_records[0]["execution"]["status"] == "observed_after_hold"
    assert policy.audit_records[1]["execution"]["status"] == "hold_requested"


def test_pre_agent_hold_is_reported_when_first_proposal_arrives(rig_and_validator, q0):
    clock = Clock()
    agent = FeedbackAgent(batch(candidate("later", {"left_j0": 0.1})))
    policy = make_policy(rig_and_validator, agent, Choice("hold"), clock)
    policy.act(observation(q0, stamp=9.0))
    assert policy.audit_records[0]["reason"] == "stale_time"
    clock.now = 10.2
    policy.act(observation(q0, stamp=10.1))
    assert agent.feedback[0].selected_id is None
    assert "stale_time" in agent.feedback[0].outcome


def test_stale_intervening_hold_does_not_erase_pending_dispatch(rig_and_validator, q0):
    clock = Clock()
    agent = FeedbackAgent(batch(candidate("one", {"left_j0": 0.2})),
                          batch(candidate("next", {"left_j0": 0.1})))
    policy = make_policy(rig_and_validator, agent, Choice("one", "hold"), clock)
    policy.act(observation(q0))
    clock.now = 10.2
    stale = policy.act(observation(q0, stamp=9.8))
    assert stale.meta["reason"] == "observation_not_new"
    assert policy.audit_records[0]["execution"]["status"] == "awaiting_observation"
    clock.now = 10.4
    policy.act(observation(q0, stamp=10.3))
    assert policy.audit_records[0]["execution"]["status"] == "observed_after_dispatch"
    assert policy.audit_records[1]["execution"]["status"] == "observed_after_hold"
    assert '"earlier_dispatch"' in agent.feedback[1].outcome
    assert '"selection":"one"' in agent.feedback[1].outcome


@pytest.mark.parametrize("status", ["done", "give_up"])
def test_stop_request_is_audited_and_not_operator_success(rig_and_validator, q0, status):
    termination = ProposalTermination(status, "model asks to stop", "hindsight",
                                      "gpt", 0.1, {"input_tokens": 3}, {})
    required = 2 if status == "done" else 3
    agent = FeedbackAgent(
        *([termination] * required),
        batch(candidate("later", {"left_j0": 0.1})),
    )
    clock = Clock()
    policy = make_policy(rig_and_validator, agent, Choice("hold"), clock)
    for index in range(required - 1):
        pending = policy.act(observation(q0, stamp=9.9 + index / 100))
        assert pending.meta["reason"] == status + "_confirmation_pending"
        clock.now += 0.1
    stop = policy.act(observation(q0, stamp=9.9 + (required - 1) / 100))
    assert stop.actions[0].meta["request_stop"] is True
    assert policy.audit_records[-1]["termination"]["status"] == status
    assert policy.audit_records[-1]["execution"]["status"] == "not_requested"
    assert policy.audit_records[-1]["agent_usage"]["input_tokens"] == 3
    clock.now += 0.1
    policy.act(observation(q0, stamp=clock.now - 0.01))
    assert agent.feedback[required].selected_id is None
    assert status in agent.feedback[required].outcome


def test_corrupt_sidecar_id_is_counted_as_missing(rig_and_validator, q0, tmp_path: Path):
    policy = make_policy(rig_and_validator,
                         FeedbackAgent(batch(candidate("one", {"left_j0": 0.2}))),
                         Choice("one"), Clock())
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first = policy.act(observation(q0))
    path = tmp_path / policy.audit_records[0]["audit_file"]
    payload = json.loads(path.read_text())
    payload["decision_id"] = "wrong-id"
    path.write_text(json.dumps(payload))
    record = _record(policy, tmp_path)
    assert record.metadata["jev_audit"]["missing_files"] == 1
    assert record.metadata["jev_audit"]["incomplete"] is True
    assert record.policy_transcript[0]["audit_sidecar_error"] == "missing_or_invalid"
    with pytest.raises(ValueError, match="mismatch"):
        load_decision(tmp_path, first.meta["decision_id"])


def test_wrong_approval_id_alone_marks_index_incomplete(rig_and_validator, q0, tmp_path: Path):
    clock = Clock()
    policy = make_policy(rig_and_validator,
                         FeedbackAgent(batch(candidate("one", {"left_j0": 0.2})),
                                       batch(candidate("next", {"left_j0": 0.1}))),
                         Choice("one", "hold"), clock)
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first_obs = observation(q0)
    policy.act(first_obs)
    clock.now = 10.2
    second_obs = replace(observation(q0, stamp=10.1),
                         extra={"approvals": [{"decision_id": "wrong-id", "detail": "clamped"}]})
    policy.act(second_obs)
    record = _record(policy, tmp_path, [_frames(tmp_path, first_obs),
                                        _frames(tmp_path, second_obs)])
    index = record.metadata["jev_audit"]
    assert index["missing_frames"] == index["missing_files"] == index["write_failures"] == 0
    assert index["incomplete_rows"] == 1 and index["incomplete"] is True


def test_sidecars_retain_all_decisions_after_inline_window(rig_and_validator, q0, tmp_path: Path):
    policy = make_policy(rig_and_validator, FeedbackAgent(), Choice(), Clock())
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first_id = None
    for _ in range(51):
        decision = policy.act(observation(q0, stamp=9.0))
        first_id = first_id or decision.meta["decision_id"]
    assert len(policy.transcript()) == 48
    assert policy.audit_records[-1]["audit_omitted_prior"] == 3
    record = _record(policy, tmp_path)
    assert record.metadata["jev_audit"]["decision_count"] == 51
    assert record.metadata["jev_audit"]["missing_files"] == 0
    assert load_decision(tmp_path, first_id)["reason"] == "stale_time"


def test_trial_end_recovers_latest_in_memory_observation_after_transient_write_failure(
    rig_and_validator, q0, tmp_path: Path, monkeypatch
):
    clock = Clock()
    policy = make_policy(rig_and_validator,
                         FeedbackAgent(batch(candidate("one", {"left_j0": 0.2})),
                                       batch(candidate("next", {"left_j0": 0.1}))),
                         Choice("one", "hold"), clock)
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first_obs = observation(q0)
    first = policy.act(first_obs)
    attempts = 0

    def fail_once(log_dir, row):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient disk error")
        return real_save_decision(log_dir, row)

    monkeypatch.setattr("inspect_robots_jev.agent_policy.save_decision", fail_once)
    clock.now = 10.2
    second_obs = observation(q0, stamp=10.1)
    policy.act(second_obs)
    assert load_decision(tmp_path, first.meta["decision_id"])["observed_state"] is None
    record = _record(policy, tmp_path, [_frames(tmp_path, first_obs),
                                        _frames(tmp_path, second_obs)])
    row = load_decision(tmp_path, first.meta["decision_id"])
    assert row["observed_state"]["state_time"] == 10.1
    assert row["audit_sidecar_error"] == "write_failed"
    assert record.metadata["jev_audit"]["write_failures"] == 1


def test_redaction_missing_frames_failed_write_and_wrong_id(
    rig_and_validator, q0, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    clock = Clock()
    bait = "secret-test-key"
    agent = FeedbackAgent(batch(replace(candidate("one", {"left_j0": 0.2}), note=bait)),
                          batch(candidate("next", {"left_j0": 0.1})))
    policy = make_policy(rig_and_validator, agent, Choice("one", "hold"), clock)
    policy.on_trial_start("scene", 0, str(tmp_path), "run")
    first_obs = observation(q0)
    first = policy.act(first_obs)
    clock.now = 10.2
    second_obs = replace(observation(q0, stamp=10.1), extra={
        "approvals": [{"decision_id": "wrong-id", "t": 1,
                       "detail": bait + " clamped"}],
        "truth": "BAIT_SUCCESS"})
    policy.act(second_obs)
    row = policy.audit_records[0]
    assert row["approval"]["status"] == "id_mismatch"
    assert row["approval"]["events"] == []
    assert row["audit_incomplete"] is True
    assert "secret-test-key" not in json.dumps(policy.transcript())
    assert "[REDACTED]" in json.dumps(policy.transcript())
    assert "BAIT_SUCCESS" not in json.dumps(policy.transcript())
    assert "rgb_base64" not in json.dumps(policy.transcript())
    assert agent.feedback[1].outcome.find("id_mismatch") >= 0

    # A failed sidecar update is visible even if the original file still exists.
    def fail_write(*args):
        raise OSError("disk full")
    monkeypatch.setattr("inspect_robots_jev.agent_policy.save_decision", fail_write)
    record = _record(policy, tmp_path, [SimpleNamespace(observation=first_obs, image_refs={})])
    index = record.metadata["jev_audit"]
    assert index["missing_frames"] >= 1 and index["write_failures"] >= 1
    assert index["incomplete"] is True
    assert record.policy_transcript[0]["audit_sidecar_error"] == "write_failed"
    assert load_decision(tmp_path, first.meta["decision_id"])["approval"]["status"] == "id_mismatch"
