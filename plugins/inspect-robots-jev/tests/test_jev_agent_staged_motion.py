"""Batch 08 software gate on a fake motor only; no YAM device is opened."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots import Action, ActionChunk, Observation
from inspect_robots.approver import ClampApprover
from inspect_robots.frames import FrameStore
from inspect_robots.types import StepResult
from inspect_robots_jev.staged_motion import StageLimits, StageStop, StagedMotionGate
from inspect_robots_yam.collision import build_yam_guardrails
from inspect_robots_yam.config import action_box

from test_agent_candidates import candidate
from test_jev_agent_policy import Agent, Choice, Clock, batch, make_policy, rig_and_validator


def obs(q, stamp):
    names = ("top_cam", "left_cam", "right_cam")
    return Observation(images={n: np.zeros((101, 101, 3), np.uint8) for n in names},
                       state={"joint_pos": np.asarray(q).copy()},
                       instruction="place objects in box",
                       image_times={n: stamp for n in names}, state_time=stamp)


class FakeMotor:
    def __init__(self, q, clock):
        self.q = q.copy()
        self.clock = clock
        self.commands = []
        self.fault = None

    def step(self, action):
        self.commands.append(action)
        if self.fault == "missing_observation":
            return StepResult(obs(self.q, self.clock.now - 0.1))
        self.clock.now += 0.01
        self.q = np.asarray(action.data).copy()
        if self.fault == "drift":
            self.q[0] += 0.01
        return StepResult(obs(self.q, self.clock.now),
                          info={"collision_warning": self.fault == "collision"})


class CountApprover:
    def __init__(self, inner, *, modify=False, veto=False):
        self.inner = inner
        self.modify, self.veto = modify, veto
        self.calls = []

    def review(self, action, store):
        self.calls.append(action.meta.get("decision_id", "candidate_preflight"))
        if self.veto:
            raise StageStop("synthetic YAM guardrail veto")
        approved = self.inner.review(action, store)
        if self.modify:
            return replace(approved, data=np.asarray(approved.data).copy())
        return approved


def synthetic(q, targets, decision_id, fingerprint, *, selected="move", jev=True):
    actions = [Action(np.asarray(t).copy(), {"decision_id": decision_id}) for t in targets]
    row = {"decision_id": decision_id, "rig_fingerprint": fingerprint,
           "agent_model": "gpt-6-astra", "jev_model": "jev-1.13.0",
           "decision_started_at": 10.0,
           "validation_mode": "measured",
           "candidate_guardrail": "active",
           "selected_id": selected, "selected_for_dispatch": selected if selected != "hold" else None,
           "selected_prefix": [list(t) for t in targets],
           "dispatch_status": "selected_for_dispatch" if selected != "hold" else "hold_requested",
           "candidates": [{"id": key, "summary": {"local_checks": {
               "full_trajectory": "passed", "interpolated_collision": "passed"}}}
                          for key in (selected, "other")] if selected != "hold" else [],
           "selector": "jev" if jev else "agent_preferred", "jev_id": selected if jev else None}
    return ActionChunk(actions), row


@pytest.fixture
def offline(tmp_path, rig_and_validator, q0):
    rig, validator = rig_and_validator
    clock = Clock()
    limits = StageLimits(validator._provenance, "synthetic Batch 07 signature fixture",
                         "fixture-only", "gpt-6-astra", "jev-1.13.0", 0.5, 0.2, 0.5, 0.001,
                         freshness_mode="capture_age")
    gate = StagedMotionGate(limits, lock_path=tmp_path / "yam.lock",
                            frame_store=FrameStore(str(tmp_path / "frames")),
                            trial_id="synthetic", clock=clock)
    motor = FakeMotor(q0, clock)
    approver = CountApprover(build_yam_guardrails(action_box(rig._cfg.low, rig._cfg.high), rig._cfg))
    yield gate, motor, approver, clock, rig, validator
    gate.close()


def run(gate, motor, approver, source, targets, decision_id, *, direction=False):
    chunk, row = synthetic(motor.q, targets, decision_id, gate.limits.rig_fingerprint)
    gate.prepare(chunk, row, source)
    before = len(motor.commands)
    with pytest.raises(StageStop, match="human permit"):
        gate.execute(motor, approver)
    assert len(motor.commands) == before
    gate.permit(decision_id, operator="operator A", estop_watcher="watcher B",
                workspace_checked=True, direction_checked=direction)
    return gate.execute(motor, approver)


def advance_to_choice(gate, motor, approver, clock):
    source = obs(motor.q, 9.9)
    gate.prepare_hold(source, decision_id="hold-1")
    with pytest.raises(StageStop, match="human permit"):
        gate.execute(motor, approver)
    gate.permit("hold-1", operator="operator A", estop_watcher="watcher B",
                workspace_checked=True)
    gate.execute(motor, approver)
    gate.sign_stage(operator="operator A", estop_watcher="watcher B", evidence_reviewed=True)
    for number, index in enumerate((0, 7, 6, 13), 1):
        source = obs(motor.q, clock.now)
        target = motor.q.copy()
        target[index] += -0.005 if index in (6, 13) else 0.005
        run(gate, motor, approver, source, [target], f"micro-{number}", direction=True)
    gate.sign_stage(operator="operator A", estop_watcher="watcher B", evidence_reviewed=True)
    assert gate.stage == "jev_choice"


def test_four_stage_fake_loop_requires_signatures_feedback_and_fresh_prefix(offline, tmp_path):
    gate, motor, approver, clock, rig, validator = offline
    advance_to_choice(gate, motor, approver, clock)
    proposals = [batch(candidate("left", {"left_j0": 0.15}),
                       candidate("right", {"right_j1": 0.15})),
                 batch(candidate("next", {"left_j0": 0.2}),
                       candidate("other", {"right_j0": 0.15})),
                 batch(candidate("final", {"left_j0": 0.25}),
                       candidate("alternative", {"right_j1": 0.1}))]
    class FeedbackAgent(Agent):
        def __init__(self, *results):
            super().__init__(*results)
            self.feedbacks = []

        def propose(self, task, source, feedback=None):
            self.feedbacks.append(feedback)
            return super().propose(task, source)

    agent = FeedbackAgent(*proposals)
    choice = Choice("right", "next", "final")
    policy = make_policy((rig, validator), agent, choice, clock)
    for index in range(3):
        source = obs(motor.q, clock.now)
        gate.prepare_from_policy(policy, source, approver)
        decision = policy.audit_records[-1]
        chunk = gate._prepared[0]
        assert decision["selected_id"] == ("right", "next", "final")[index]
        assert choice.calls[index][-1] == "hold"
        gate.permit(decision["decision_id"], operator="operator A", estop_watcher="watcher B",
                    workspace_checked=True)
        before = len(motor.commands)
        evidence = gate.execute(motor, approver)
        assert len(motor.commands) - before == len(chunk) <= 3
        assert evidence["decision_id"] == decision["decision_id"]
        assert evidence["max_joint_error"] == 0
        assert len(evidence["controller_trace"]) == len(chunk)
        assert all(set(refs) == {"top_cam", "left_cam", "right_cam"}
                   for refs in evidence["step_frame_refs"])
        assert all(a.meta["decision_id"] == decision["decision_id"]
                   for a in motor.commands[before:])
        assert evidence["selected_id"] not in set(evidence["candidate_ids"]) - {evidence["selected_id"]}
        if index == 0:
            previous_calls = agent.calls
            with pytest.raises(StageStop, match="signature"):
                gate.prepare_from_policy(policy, obs(motor.q, clock.now), approver)
            assert agent.calls == previous_calls
            with pytest.raises(StageStop, match="signature"):
                gate.prepare(chunk, decision, source)
            gate.sign_stage(operator="operator A", estop_watcher="watcher B", evidence_reviewed=True)
        if index == 1:
            with pytest.raises(StageStop, match="two new decisions"):
                gate.sign_stage(operator="operator A", estop_watcher="watcher B", evidence_reviewed=True)
    assert agent.feedbacks[1].outcome.find(gate.segments[-3]["decision_id"]) >= 0
    assert agent.feedbacks[2].outcome.find(gate.segments[-2]["decision_id"]) >= 0
    assert gate.segments[-1]["agent_feedback_decision_id"] == gate.segments[-2]["decision_id"]
    gate.sign_stage(operator="operator A", estop_watcher="watcher B", evidence_reviewed=True)
    assert gate.stage == "complete" and len(gate.signed) == 4
    assert len(set(approver.calls)) == 9 and "candidate_preflight" in approver.calls
    record_path = tmp_path / "stage-record.json"
    gate.save_record(record_path)
    saved = json.loads(record_path.read_text(encoding="utf-8"))
    assert saved["current_stage"] == "complete"
    assert len(saved["segments"]) == 8 and len(saved["signed_stages"]) == 4
    assert saved["segments"][-1]["agent_feedback_decision_id"] == saved["segments"][-2]["decision_id"]
    assert all((tmp_path / "frames").exists() and all(
        os.path.isfile(path) for path in row["input_frame_refs"].values())
        for row in saved["segments"])


@pytest.mark.parametrize("fault", ["prefix", "fingerprint", "guardrail_missing",
                                    "micro_size", "micro_two_joints",
                                    "old_observation", "jev_mismatch", "few_candidates"])
def test_invalid_prepared_motion_never_reaches_fake_motor(offline, fault):
    gate, motor, approver, clock, _, _ = offline
    if fault in ("jev_mismatch", "few_candidates"):
        advance_to_choice(gate, motor, approver, clock)
    elif fault not in ("old_observation",):
        run(gate, motor, approver, obs(motor.q, 9.9), [motor.q], "hold-1")
        gate.sign_stage(operator="operator A", estop_watcher="watcher B", evidence_reviewed=True)
    source = obs(motor.q, clock.now)
    target = motor.q.copy()
    target[0] += 0.02 if fault == "micro_size" else 0.005
    if fault == "micro_two_joints":
        target[1] += 0.005
    chunk, row = synthetic(motor.q, [target], "bad", gate.limits.rig_fingerprint)
    if fault == "prefix":
        row["selected_prefix"] = [motor.q.tolist()]
    if fault == "fingerprint":
        row["rig_fingerprint"] = {"wrong": True}
    if fault == "guardrail_missing":
        row["candidate_guardrail"] = "not_configured"
    if fault == "old_observation":
        source = obs(motor.q, 9.0)
    if fault == "jev_mismatch":
        row["jev_id"] = "other"
    if fault == "few_candidates":
        row["candidates"] = [{"id": "move"}]
    before = len(motor.commands)
    with pytest.raises((StageStop, ValueError)):
        gate.prepare(chunk, row, source)
    assert len(motor.commands) == before


@pytest.mark.parametrize("fault", ["veto", "modified", "drift", "collision", "missing_observation"])
def test_fault_stops_before_next_fake_motor_command_and_requires_recovery(offline, fault):
    gate, motor, approver, clock, _, _ = offline
    if fault == "veto":
        approver.veto = True
    if fault == "modified":
        approver.modify = True
    if fault in ("drift", "collision", "missing_observation"):
        motor.fault = fault
    chunk, row = synthetic(motor.q, [motor.q], "hold-fault", gate.limits.rig_fingerprint,
                           selected="hold")
    gate.prepare(chunk, row, obs(motor.q, 9.9))
    gate.permit("hold-fault", operator="A", estop_watcher="B", workspace_checked=True)
    with pytest.raises(StageStop):
        gate.execute(motor, approver)
    assert len(motor.commands) == (0 if fault in ("veto", "modified") else 1)
    with pytest.raises(StageStop):
        gate.sign_stage(operator="A", estop_watcher="B", evidence_reviewed=True)
    with pytest.raises(StageStop):
        gate.recover(obs(motor.q, clock.now), operator="A", estop_watcher="B",
                     fault_reviewed=False)
    gate.recover(obs(motor.q, clock.now), operator="A", estop_watcher="B",
                 fault_reviewed=True)
    assert gate.stage == "hold" and gate.segments[-1]["result"] == "failed"


def test_actual_yam_guardrail_blocks_synthetic_collision_before_fake_motor(offline):
    gate, motor, _, clock, rig, _ = offline
    safe = CountApprover(build_yam_guardrails(action_box(rig._cfg.low, rig._cfg.high), rig._cfg))
    advance_to_choice(gate, motor, safe, clock)
    target = motor.q.copy()
    target[1] = 2.0  # legal joint value; installed YAM model predicts collision
    chunk, row = synthetic(motor.q, [target], "collision", gate.limits.rig_fingerprint)
    gate.prepare(chunk, row, obs(motor.q, clock.now))
    gate.permit("collision", operator="A", estop_watcher="B", workspace_checked=True)
    before = len(motor.commands)
    with pytest.raises(StageStop, match="approver modified"):
        gate.execute(motor, safe)
    assert len(motor.commands) == before


def test_candidate_preflight_filters_before_jev_without_mutating_approval_state(offline, monkeypatch):
    gate, motor, approver, clock, rig, validator = offline
    advance_to_choice(gate, motor, approver, clock)
    collision = approver.inner._approvers[-1]._checker
    original_check = collision.check

    def extra_collision(q):
        if q[0] > motor.q[0] + 0.001:
            return SimpleNamespace(collided=True, geom1="synthetic_link", geom2="table")
        return original_check(q)

    monkeypatch.setattr(collision, "check", extra_collision)
    agent = Agent(batch(candidate("left", {"left_j0": 0.15}),
                        candidate("right", {"right_j1": 0.15})))
    choice = Choice("right")
    policy = make_policy((rig, validator), agent, choice, clock)
    before_commands = len(motor.commands)
    before_store = repr(gate._store)
    with pytest.raises(StageStop, match="two safe candidates"):
        gate.prepare_from_policy(policy, obs(motor.q, clock.now), approver)
    row = policy.audit_records[-1]
    assert {f["id"]: f["code"] for f in row["filtered"]}["left"] == "yam_guardrail"
    assert choice.calls == [["right", "hold"]]
    assert repr(gate._store) == before_store
    assert len(motor.commands) == before_commands


def test_model_change_and_operator_stop_are_fail_closed(offline):
    gate, motor, approver, _, _, _ = offline
    chunk, row = synthetic(motor.q, [motor.q], "hold", gate.limits.rig_fingerprint,
                           selected="hold")
    row["jev_model"] = "unexpected-version"
    with pytest.raises(StageStop, match="model changed"):
        gate.prepare(chunk, row, obs(motor.q, 9.9))
    gate.prepare_hold(obs(motor.q, 9.9), decision_id="hold")
    gate.stop("operator saw wrong orientation")
    with pytest.raises(StageStop):
        gate.execute(motor, approver)
    assert motor.commands == []


def test_missing_yam_guardrail_rejected_before_motor(offline):
    gate, motor, _, _, rig, _ = offline
    gate.prepare_hold(obs(motor.q, 9.9), decision_id="hold")
    gate.permit("hold", operator="A", estop_watcher="B", workspace_checked=True)
    with pytest.raises(StageStop, match="YAM clamp, delta and collision approver"):
        gate.execute(motor, ClampApprover(action_box(rig._cfg.low, rig._cfg.high)))
    assert motor.commands == []


def test_exclusive_lock_and_expired_operator_review_stop_before_motor(offline):
    gate, motor, approver, clock, _, _ = offline
    gate.prepare_hold(obs(motor.q, 9.9), decision_id="hold")
    gate.permit("hold", operator="A", estop_watcher="B", workspace_checked=True)
    fd = os.open(gate.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    assert motor.commands == []
    clock.now = 10.51
    with pytest.raises(StageStop, match="decision expired"):
        gate.execute(motor, approver)
    assert motor.commands == []


def test_rejects_four_step_segment_before_motor(offline):
    gate, motor, approver, clock, _, _ = offline
    advance_to_choice(gate, motor, approver, clock)
    target = motor.q.copy()
    target[0] += 0.005
    chunk, row = synthetic(motor.q, [target] * 4, "too-long", gate.limits.rig_fingerprint)
    before = len(motor.commands)
    with pytest.raises(StageStop, match="one to three"):
        gate.prepare(chunk, row, obs(motor.q, clock.now))
    assert len(motor.commands) == before


def test_micro_segment_cannot_switch_joints_or_reverse_direction(offline):
    gate, motor, approver, clock, _, _ = offline
    run(gate, motor, approver, obs(motor.q, 9.9), [motor.q], "hold-1")
    gate.sign_stage(operator="A", estop_watcher="B", evidence_reviewed=True)
    first = motor.q.copy()
    first[0] += 0.005
    switched = first.copy()
    switched[1] += 0.005
    reversed_step = first.copy()
    reversed_step[0] -= 0.005
    for last in (switched, reversed_step):
        chunk, row = synthetic(motor.q, [first, last], "micro-invalid",
                               gate.limits.rig_fingerprint)
        before = len(motor.commands)
        with pytest.raises(StageStop, match="one joint and direction"):
            gate.prepare(chunk, row, obs(motor.q, clock.now))
        assert len(motor.commands) == before


def test_operator_stop_between_steps_blocks_remaining_fake_command(offline):
    gate, motor, approver, clock, _, _ = offline
    run(gate, motor, approver, obs(motor.q, 9.9), [motor.q], "hold-1")
    gate.sign_stage(operator="A", estop_watcher="B", evidence_reviewed=True)
    first = motor.q.copy()
    first[0] += 0.005
    second = first.copy()
    second[0] += 0.005
    chunk, row = synthetic(motor.q, [first, second], "micro-stop",
                           gate.limits.rig_fingerprint)
    gate.prepare(chunk, row, obs(motor.q, clock.now))
    gate.permit("micro-stop", operator="A", estop_watcher="B",
                workspace_checked=True, direction_checked=True)
    original_step = motor.step

    def stopped_after_first(action):
        result = original_step(action)
        gate.stop("operator observed wrong direction")
        return result

    motor.step = stopped_after_first
    before = len(motor.commands)
    with pytest.raises(StageStop, match="operator stopped"):
        gate.execute(motor, approver)
    assert len(motor.commands) - before == 1


def test_recovery_invalidates_earlier_unsigned_segments_in_same_stage(offline):
    gate, motor, approver, clock, _, _ = offline
    run(gate, motor, approver, obs(motor.q, 9.9), [motor.q], "hold-1")
    gate.sign_stage(operator="A", estop_watcher="B", evidence_reviewed=True)
    first = motor.q.copy()
    first[0] += 0.005
    run(gate, motor, approver, obs(motor.q, clock.now), [first], "micro-good",
        direction=True)
    motor.fault = "collision"
    second = motor.q.copy()
    second[7] += 0.005
    chunk, row = synthetic(motor.q, [second], "micro-fault",
                           gate.limits.rig_fingerprint)
    gate.prepare(chunk, row, obs(motor.q, clock.now))
    gate.permit("micro-fault", operator="A", estop_watcher="B",
                workspace_checked=True, direction_checked=True)
    with pytest.raises(StageStop):
        gate.execute(motor, approver)
    gate.recover(obs(motor.q, clock.now), operator="A", estop_watcher="B",
                 fault_reviewed=True)
    assert gate.segments[-2]["result"] == "invalidated_by_fault"
    with pytest.raises(StageStop, match="successful evidence"):
        gate.sign_stage(operator="A", estop_watcher="B", evidence_reviewed=True)


def test_frame_gate_accepts_old_capture_time_and_requires_measured_checks(
        tmp_path, rig_and_validator, q0):
    rig, validator = rig_and_validator
    clock = Clock()
    limits = StageLimits(validator._provenance, "signed-record", "operator",
                         "gpt-6-astra", "jev-1.13.0", 0.5, 0.2, 0.5, 0.001)

    def frame(q, number, stamp):
        source = obs(q, stamp)
        source.extra["camera_frame_ids"] = {
            name: (4, number) for name in ("top_cam", "left_cam", "right_cam")}
        return source

    class FrameMotor:
        def __init__(self):
            self.q = q0.copy()
            self.calls = 0

        def step(self, action):
            self.calls += 1
            self.q = np.asarray(action.data).copy()
            return StepResult(frame(self.q, self.calls + 1, 1.0 + self.calls / 10))

    motor = FrameMotor()
    approver = CountApprover(build_yam_guardrails(action_box(rig._cfg.low, rig._cfg.high), rig._cfg))
    with StagedMotionGate(limits, lock_path=tmp_path / "frame.lock",
                          frame_store=FrameStore(str(tmp_path / "frames")),
                          trial_id="frame-test", clock=clock) as gate:
        gate.prepare_hold(frame(q0, 1, 1.0), decision_id="old-but-new")
        gate.permit("old-but-new", operator="A", estop_watcher="B", workspace_checked=True)
        evidence = gate.execute(motor, approver)
        assert tuple(evidence["next_frame_ids"]["top_cam"]) == (4, 2)
        gate.sign_stage(operator="A", estop_watcher="B", evidence_reviewed=True)
        target = q0.copy()
        target[0] += 0.005
        chunk, row = synthetic(q0, [target], "motion", limits.rig_fingerprint)
        row["validation_mode"] = "yam"
        with pytest.raises(StageStop, match="measured validation"):
            gate.prepare(chunk, row, frame(q0, 2, 1.1))
        row["validation_mode"] = "measured"
        row["candidates"][0]["summary"]["local_checks"]["full_trajectory"] = "not_checked"
        with pytest.raises(StageStop, match="full trajectory"):
            gate.prepare(chunk, row, frame(q0, 2, 1.1))
        with pytest.raises(StageStop, match="missing_frame_ids"):
            gate.prepare(chunk, row, obs(q0, 1.1))
