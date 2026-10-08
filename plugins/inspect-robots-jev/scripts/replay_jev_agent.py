"""CLI for Batch 06 offline YAM replay. No embodiment or controller is created."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# The installed YAM package eagerly imports its real embodiment from __init__.
# Install a package shell so only explicitly requested pure-data submodules load.
# This must run before any inspect_robots_jev or inspect_robots_yam import.
if "inspect_robots_yam" in sys.modules:
    raise RuntimeError("replay must start in a fresh process before YAM imports")
yam_spec = importlib.util.find_spec("inspect_robots_yam")
if yam_spec is None or yam_spec.submodule_search_locations is None:
    raise RuntimeError("inspect_robots_yam package is unavailable")
sys.modules["inspect_robots_yam"] = importlib.util.module_from_spec(yam_spec)

from inspect_robots_agent.proposals import AgentProposer
from inspect_robots_yam.config import YamConfig, action_box, observation_space
from inspect_robots_jev.audit import MAX_RECORD_BYTES
from inspect_robots_jev.jev_choice import JevChoiceClient
from inspect_robots_jev.replay import make_data_validator, replay
from inspect_robots_jev.yam_contract import CAMERA_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live-services", action="store_true",
                        help="Call Agent and JEV services for offline proposals only")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.resolve().parent
    cfg = YamConfig(**manifest["rig"])
    validator = make_data_validator(
        manifest["rig"], base / manifest["calibration"], base / manifest["mjcf"],
        max_image_age_s=manifest.get("max_image_age_s", 1.0),
        max_skew_s=manifest.get("max_skew_s", 0.2))
    proposer = None
    choice = None
    if args.live_services:
        proposer = AgentProposer(model="openai/gpt-4o-mini",
                                 action_space=action_box(cfg.low, cfg.high),
                                 observation_space=observation_space(
                                     cfg.cam_height, cfg.cam_width, CAMERA_NAMES),
                                 candidate_count=3)
        choice = JevChoiceClient()
    rows = replay(manifest, base=base, validator=validator,
                  proposer=proposer, choice=choice, out_dir=args.out)
    index = {"proposed_only": True, "jev_audit": {
        "schema_version": 1, "episode_id": manifest["episode_id"],
        "relative_dir": f"jev-audit/{manifest['episode_id']}",
        "decision_count": len(rows), "record_limit_bytes": MAX_RECORD_BYTES,
        "unverified_decisions": sum(row["execution"]["status"] ==
                                    "unverified_no_next_observation" for row in rows),
        "source": "offline_replay", "proposed_only": True,
    }}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "replay-index.json").write_text(json.dumps(index, ensure_ascii=False),
                                                   encoding="utf-8")
    print(json.dumps({"proposed_only": True, "episode_id": manifest["episode_id"],
                      "decisions": [{"decision_id": row["decision_id"],
                                     "proposed_only": True,
                                     "reason": row["reason"],
                                     "agent_selection": row["agent_selection"],
                                     "jev_selection": row["jev_selection"],
                                     "audit_file": row["audit_file"]} for row in rows]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
