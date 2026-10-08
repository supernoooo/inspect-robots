"""Materialize synthetic saved RGB observations for the Batch 06 example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    template = Path(__file__).parents[1] / "examples" / "replay-manifest.example.json"
    manifest = json.loads(template.read_text(encoding="utf-8"))
    manifest["calibration"] = str((template.parent / manifest["calibration"]).resolve())
    manifest["mjcf"] = str((template.parent / manifest["mjcf"]).resolve())
    target = args.directory.resolve()
    target.mkdir(parents=True, exist_ok=True)
    size = (manifest["rig"]["cam_height"], manifest["rig"]["cam_width"], 3)
    for index, entry in enumerate(manifest["rounds"]):
        for filename in entry["rgb"].values():
            image_path = target / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(image_path, np.full(size, 73 + index, dtype=np.uint8))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(target / "manifest.json")


if __name__ == "__main__":
    main()
