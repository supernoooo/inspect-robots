"""Launch two resets and one step as a live Isaac Lab camera/reset smoke test."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np

from inspect_robots import Action, Scene
from inspect_robots_isaacsim_2yam import IsaacSim2YamEmbodiment


def main() -> None:
    """Run the smallest live simulator proof and always release the app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional directory for reset/step camera PNGs",
    )
    args = parser.parse_args()
    embodiment = IsaacSim2YamEmbodiment(asset_path=args.asset_path, device=args.device)
    try:
        observation = embodiment.reset(
            Scene(id="boot-proof", instruction="put everything into the box"), seed=0
        )
        action = np.zeros(14, dtype=np.float32)
        action[[6, 13]] = 1.0
        action[8] = 0.5
        result = embodiment.step(Action(data=action))
        reset_again = embodiment.reset(
            Scene(id="boot-proof-reset", instruction="put everything into the box"), seed=1
        )
        reset_refresh_mad = {
            camera: float(
                np.abs(
                    np.asarray(reset_again.images[camera], dtype=np.float32)
                    - np.asarray(result.observation.images[camera], dtype=np.float32)
                ).mean()
            )
            for camera in observation.images
        }
        stale = [camera for camera, delta in reset_refresh_mad.items() if delta == 0.0]
        if stale:
            raise RuntimeError(
                "reset returned stale camera buffers for: " + ", ".join(sorted(stale))
            )
        if args.output_dir is not None:
            from PIL import Image

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for phase, frames in (
                ("reset", observation.images),
                ("step", result.observation.images),
                ("reset_again", reset_again.images),
            ):
                for camera, frame in frames.items():
                    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                        output_dir / f"{phase}_{camera}.png"
                    )
        print(
            {
                "cameras": {key: value.shape for key, value in observation.images.items()},
                "camera_ranges": {
                    key: (int(value.min()), int(value.max()))
                    for key, value in observation.images.items()
                },
                "state": result.observation.state["joint_pos"].shape,
                "success": result.info["success"],
                "reset_refresh_mad": reset_refresh_mad,
                "output_dir": args.output_dir,
            },
            flush=True,
        )
    except Exception:
        # SimulationApp.close() may replace Python's normal exception shutdown,
        # so print the original failure before releasing the Kit application.
        traceback.print_exc()
        raise
    finally:
        embodiment.close()


if __name__ == "__main__":
    main()
