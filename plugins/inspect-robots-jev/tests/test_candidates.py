"""Batch 04 finite candidates and observation-gated task state."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from inspect_robots_jev.candidates import CandidateGenerator, Stage

from conftest import location, observation


def locations():
    return {"red_block": location("red_block", (0.6, 0.2, 0.0)),
            "yellow_ball": location("yellow_ball", (0.6, -0.2, 0.0)),
            "box": location("box", (0.8, 0.2, 0.14))}


def test_finite_stable_ids_and_json_only_summaries(motion, q0) -> None:
    planner = CandidateGenerator(motion)
    first = planner.generate(observation(q0), locations())
    second = planner.generate(observation(q0), locations())
    assert 2 <= len(first.available) <= 6
    assert [c.id for c in first.available] == [c.id for c in second.available]
    assert {"hold", "reobserve"}.issubset({c.id for c in first.available})
    candidate = first.by_id("red_block:left:approach")
    summary = candidate.jev_summary()
    assert summary["category"] == "red_block" and summary["side"] == "left"
    assert summary["operation"] == "eef" and summary["trajectory_steps"] >= 1
    assert "images" not in json.dumps(summary)
    assert json.loads(json.dumps(summary)) == summary
    assert candidate.chunk(q0).meta["reobserve_after_chunk"] is True


def test_unreliable_target_filters_motion_and_falls_back_to_hold(motion, q0) -> None:
    planner = CandidateGenerator(motion)
    bad = {"red_block": location("red_block", (0, 0, 0), usable=False)}
    result = planner.generate(observation(q0), bad)
    assert {item.id for item in result.available} == {"hold", "reobserve"}
    assert result.filtered["red_block:left:approach"] == "high_uncertainty"
    chunk = result.fallback(q0)
    assert len(chunk) == 1 and chunk.meta["reason"] == "no_safe_candidate"
    np.testing.assert_array_equal(chunk.actions[0].data, q0)


def test_missing_box_geometry_filters_motion(motion, q0) -> None:
    found = CandidateGenerator(motion).generate(
        observation(q0), {"red_block": location("red_block", (0.6, 0.2, 0))})
    assert found.filtered["red_block:left:approach"] == "unreliable_box:missing_localization"
    assert {c.id for c in found.available} == {"hold", "reobserve"}


def test_phase_requires_new_observation_and_observed_motion(motion, q0) -> None:
    planner = CandidateGenerator(motion)
    found = planner.generate(observation(q0, 1), locations())
    candidate = found.by_id("red_block:left:approach")
    planner.mark_dispatched(candidate, observation(q0, 1), found)
    assert planner.generate(observation(q0, 1), locations()).stage == Stage.APPROACH
    # A new image alone or a new state alone cannot confirm the chunk.
    stale_image = observation(candidate.motion.trajectory[-1], 1)
    stale_image = type(stale_image)(stale_image.images, stale_image.joint_pos,
                                    stale_image.instruction, stale_image.image_times, 2)
    assert planner.generate(stale_image, locations()).stage == Stage.APPROACH
    assert planner.generate(observation(q0, 2), locations()).stage == Stage.APPROACH
    # Re-dispatch after unchanged feedback; only observed endpoint advances.
    again = planner.generate(observation(q0, 2), locations())
    candidate = again.by_id("red_block:left:approach")
    planner.mark_dispatched(candidate, observation(q0, 2), again)
    reached = observation(candidate.motion.trajectory[-1], 3)
    assert planner.generate(reached, locations()).stage == Stage.ALIGN


def test_full_stage_cycle_uses_encoder_feedback(motion, q0) -> None:
    planner = CandidateGenerator(motion)
    q = q0.copy()
    for index, stage in enumerate((Stage.APPROACH, Stage.ALIGN, Stage.DESCEND,
                                   Stage.CLOSE, Stage.LIFT, Stage.MOVE_BOX, Stage.RELEASE)):
        observed = observation(q, float(index * 2 + 1))
        found = planner.generate(observed, locations())
        assert found.stage == stage
        candidate = found.by_id(f"red_block:left:{stage.value}")
        planner.mark_dispatched(candidate, observed, found)
        assert planner.generate(observed, locations()).stage == stage
        q = candidate.motion.trajectory[-1].copy()
        assert planner.generate(observation(q, float(index * 2 + 2)), locations()).stage == (
            Stage.REOBSERVE if stage == Stage.RELEASE else list(Stage)[index + 1])
    reobserve = planner.generate(observation(q, 16), locations())
    planner.mark_dispatched(reobserve.by_id("reobserve"), observation(q, 16), reobserve)
    assert planner.generate(observation(q, 17), locations()).stage == Stage.APPROACH


def test_ik_failure_is_machine_readable_in_candidate_filter(motion, synthetic_kinematics, q0) -> None:
    synthetic_kinematics.fail_ik = True
    found = CandidateGenerator(motion).generate(observation(q0), locations())
    assert found.filtered["red_block:left:approach"] == "ik_non_converged"
    assert {c.id for c in found.available} == {"hold", "reobserve"}


def test_target_drift_and_provenance_do_not_advance_or_generate(motion, q0) -> None:
    planner = CandidateGenerator(motion)
    found = planner.generate(observation(q0, 1), locations())
    chosen = found.by_id("red_block:left:approach")
    planner.mark_dispatched(chosen, observation(q0, 1), found)
    shifted = locations()
    shifted["red_block"] = location("red_block", (0.7, 0.2, 0))
    endpoint = chosen.motion.trajectory[-1]
    assert planner.generate(observation(endpoint, 2), shifted).stage == Stage.APPROACH
    wrong = locations()
    wrong["red_block"] = replace(wrong["red_block"], mjcf_sha256="different-model")
    result = planner.generate(observation(endpoint, 3), wrong)
    assert result.filtered["red_block:left:approach"] == "provenance_mismatch"


def test_stale_candidate_cannot_be_dispatched(motion, q0) -> None:
    planner = CandidateGenerator(motion)
    old = planner.generate(observation(q0, 1), locations())
    planner.generate(observation(q0, 2), locations())
    with pytest.raises(ValueError, match="current observation"):
        planner.mark_dispatched(old.by_id("red_block:left:approach"), observation(q0, 1), old)
