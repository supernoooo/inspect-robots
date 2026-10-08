"""Batch 07 live, proposal-only runner using an externally audited read source.

The source factory is site-owned. Importing/calling it may have device side
effects, so field staff must audit its connection and shutdown separately.
This CLI itself never constructs a YAM embodiment or dispatches an action.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

# Avoid YAM's package-level eager import of its real embodiment. The source
# factory is loaded later, only at the explicit field command's request.
if "inspect_robots_yam" in sys.modules:
    raise RuntimeError("shadow must start in a fresh process before YAM imports")
yam_spec = importlib.util.find_spec("inspect_robots_yam")
if yam_spec is None or yam_spec.submodule_search_locations is None:
    raise RuntimeError("inspect_robots_yam package is unavailable")
sys.modules["inspect_robots_yam"] = importlib.util.module_from_spec(yam_spec)

from inspect_robots_agent.proposals import AgentProposer
from inspect_robots_yam.config import YamConfig, action_box, observation_space
from inspect_robots_jev.jev_choice import JevChoiceClient
from inspect_robots_jev.replay import make_data_validator
from inspect_robots_jev.shadow import ShadowRunner, _limit, validate_shadow_rig
from inspect_robots_jev.yam_contract import CAMERA_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path,
                        help="JSON with rig, calibration, mjcf, task, and signed age limits")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-factory", required=True, metavar="MODULE:FUNCTION",
                        help="Field-audited factory returning read_cameras/read_joints")
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.rounds > 10000:
        parser.error("--rounds must be from 1 to 10000")
    if args.source_factory.count(":") != 1:
        parser.error("--source-factory must be MODULE:FUNCTION")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("rig"), dict):
        parser.error("manifest must contain a rig object")
    cfg = YamConfig(**manifest["rig"])
    validate_shadow_rig(cfg)
    for name in ("max_dispatch_age_s", "max_image_age_s", "max_skew_s"):
        _limit(manifest[name], name)
    _limit(manifest.get("inference_budget_s", 30.0), "inference_budget_s")
    base = args.manifest.resolve().parent
    validator = make_data_validator(
        manifest["rig"], base / manifest["calibration"], base / manifest["mjcf"],
        max_image_age_s=manifest["max_image_age_s"],
        max_skew_s=manifest["max_skew_s"])
    proposer = AgentProposer(
        model="openai/gpt-6-astra", action_space=action_box(cfg.low, cfg.high),
        observation_space=observation_space(cfg.cam_height, cfg.cam_width, CAMERA_NAMES),
        candidate_count=3)
    choice = JevChoiceClient()
    def sha_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    rig_bytes = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    session = {
        "proposed_only": True, "status": "待现场验证",
        "versions": {name: version(name) for name in (
            "inspect-robots", "inspect-robots-agent", "inspect-robots-jev",
            "inspect-robots-yam", "mujoco")},
        "agent_model_requested": "openai/gpt-6-astra",
        "jev_model_requested": choice.model,
        "manifest_sha256": sha_file(args.manifest),
        "rig_config_sha256": hashlib.sha256(rig_bytes).hexdigest(),
        "calibration_sha256": sha_file(base / manifest["calibration"]),
        "mjcf_sha256": sha_file(base / manifest["mjcf"]),
        "source_factory": args.source_factory,
        "max_dispatch_age_s": manifest["max_dispatch_age_s"],
        "max_image_age_s": manifest["max_image_age_s"],
        "max_skew_s": manifest["max_skew_s"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shadow-session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    module_name, factory_name = args.source_factory.split(":")
    source = getattr(importlib.import_module(module_name), factory_name)(cfg)
    runner = ShadowRunner(
        source, validator, proposer, choice, out_dir=args.out,
        max_dispatch_age_s=manifest["max_dispatch_age_s"],
        max_image_age_s=manifest["max_image_age_s"],
        max_skew_s=manifest["max_skew_s"],
        inference_budget_s=manifest.get("inference_budget_s", 30.0))
    # The source owner controls its own connection/exit lifecycle. This script
    # deliberately has no reset/close/park call, including on exceptions.
    try:
        for _ in range(args.rounds):
            scene = input("Scene status before round? unchanged/changed/unknown: ").strip()
            if scene not in {"unchanged", "changed", "unknown"}:
                scene = "unknown"
            row = runner.run_round(manifest["instruction"], scene_change=scene)
            print(json.dumps({"decision_id": row["decision_id"], "proposed_only": True,
                              "selected_for_dispatch": None, "reason": row["reason"],
                              "audit_file": row["audit_file"]}, ensure_ascii=False))
    finally:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "shadow-summary.json").write_text(
            json.dumps(runner.summary(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
