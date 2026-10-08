"""Evidence-based interpretation of Jev audit records, without simulator truth."""

from __future__ import annotations

from typing import Any


def diagnose_transcript(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each decision as a lead for review, never as proven task failure.

    Only the policy's allowed observation summary, model outputs, candidate
    decisions, and planned/observed joint targets are consulted.
    """
    results: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or "decision_id" not in row:
            continue
        vision = row.get("vision") if isinstance(row.get("vision"), dict) else {}
        localization = row.get("localization") if isinstance(row.get("localization"), dict) else {}
        filtered = row.get("filtered") if isinstance(row.get("filtered"), dict) else {}
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
        reason = row.get("reason")
        bad_vision = {camera: {"state": item.get("state"), "failure": item.get("failure")}
                      for camera, item in vision.items()
                      if isinstance(item, dict) and item.get("failure") not in (None, "empty_detection")}
        missing = {name: value.get("reason") for name, value in localization.items()
                   if isinstance(value, dict) and not value.get("usable")}
        vision_status = "suspected" if bad_vision or reason in (
            "vision_empty_detection", "vision_error", "stale_time") else "not_indicated"
        geometry_status = "suspected" if filtered or missing or reason == "no_safe_candidate" else "not_indicated"
        movement = [c.get("id") for c in candidates if isinstance(c, dict)
                    and c.get("operation") not in ("hold", "reobserve")]
        selected = row.get("selected_id")
        choice_status = ("review_required" if selected is not None and
                         (len(movement) > 1 or selected in ("hold", "reobserve"))
                         else "not_indicated")
        if reason in ("jev_unknown_candidate_id", "jev_invalid_probabilities"):
            choice_status = "suspected"
        exec_state = execution.get("status")
        execution_status = ("suspected" if exec_state == "not_reached_joint_target"
                            else "awaiting_evidence" if exec_state == "awaiting_observation"
                            else "not_indicated")
        results.append({
            "decision_id": row["decision_id"], "branch": row.get("branch"),
            "vision_error": {"status": vision_status,
                             "evidence": {"vision_failures": bad_vision,
                                          "reason": reason if vision_status == "suspected" else None,
                                          "frame_refs": (row.get("observation") or {}).get("frame_refs", {})}},
            "unreachable_or_filtered": {"status": geometry_status,
                                        "evidence": {"localization_failures": missing,
                                                     "filtered": filtered,
                                                     "available_motion_ids": movement}},
            "jev_choice": {"status": choice_status,
                           "evidence": {"selected_id": selected, "probabilities": row.get("probabilities"),
                                        "probabilities_status": row.get("probabilities_status"),
                                        "available_motion_ids": movement}},
            "execution_not_reached": {"status": execution_status,
                                      "evidence": {"state": exec_state,
                                                   "target_joint_pos": execution.get("target_joint_pos"),
                                                   "observed_joint_pos": execution.get("observed_joint_pos"),
                                                   "max_joint_error": execution.get("max_joint_error"),
                                                   "next_state_time": execution.get("observed_state_time")}},
        })
    return results
