"""Print per-decision evidence from a Jev transcript or Inspect Robots eval log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_robots_jev.audit import load_decision, load_episode
from inspect_robots_jev.diagnostics import diagnose_transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="JSON transcript or Inspect Robots eval log")
    parser.add_argument("--decision-id", help="inspect one decision from its bounded sidecar")
    args = parser.parse_args()
    if args.decision_id:
        print(json.dumps(diagnose_transcript([load_decision(args.log.parent, args.decision_id)]),
                         ensure_ascii=False, indent=2, allow_nan=False))
        return
    data = json.loads(args.log.read_text(encoding="utf-8"))
    if isinstance(data, list):
        groups = [{"scene_id": None, "epoch": None, "rows": data}]
    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        groups = []
        for scene in data["samples"]:
            if not isinstance(scene, dict):
                continue
            for epoch, rows in enumerate(scene.get("policy_transcripts", [])):
                if not isinstance(rows, list):
                    continue
                metadata = scene.get("trial_metadata", [])
                trial_metadata = metadata[epoch] if isinstance(metadata, list) and epoch < len(metadata) else {}
                index = trial_metadata.get("jev_audit") if isinstance(trial_metadata, dict) else None
                if isinstance(index, dict):
                    rows = load_episode(args.log.parent, index["episode_id"], index["decision_count"])
                groups.append({"scene_id": scene.get("scene_id"), "epoch": epoch, "rows": rows})
    else:
        parser.error("expected a Jev transcript list or eval log with samples")
    print(json.dumps([{**{k: group[k] for k in ("scene_id", "epoch")},
                       "diagnoses": diagnose_transcript(group["rows"])} for group in groups],
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
