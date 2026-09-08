"""Run a model-free LIBERO reset/step/reset camera and state smoke test."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import numpy as np


def main() -> None:
    """Launch one official scene, validate observations, and always close MuJoCo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default=None)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.device_id))
    if args.libero_root is not None:
        os.environ["INSPECT_ROBOTS_LIBERO_ROOT"] = args.libero_root

    # Keep the heavyweight simulator import after EGL/root configuration.
    from inspect_robots import Action, Scene
    from inspect_robots_libero import LiberoEmbodiment
    from inspect_robots_libero.embodiment import load_suite

    suite = load_suite(args.suite)
    if args.task_id < 0 or args.task_id >= len(suite.tasks):
        raise ValueError(f"task id must be in 0..{len(suite.tasks) - 1}")
    task = suite.get_task(args.task_id)
    embodiment = LiberoEmbodiment(
        cam_height=args.height,
        cam_width=args.width,
        num_steps_wait=10,
    )
    try:
        scene = Scene(
            id="libero-boot-proof",
            instruction=str(task.language),
            init_seed=0,
            metadata={
                "libero_suite": args.suite,
                "libero_task_id": args.task_id,
                "libero_init_state_id": args.init_state_id,
            },
        )
        observation = embodiment.reset(scene)
        action = np.zeros(7, dtype=np.float64)
        action[-1] = -1.0
        result = embodiment.step(Action(data=action))
        reset_again = embodiment.reset(scene, seed=1)
        deltas = {
            camera: float(
                np.abs(
                    reset_again.images[camera].astype(np.float32)
                    - result.observation.images[camera].astype(np.float32)
                ).mean()
            )
            for camera in observation.images
        }
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
                    Image.fromarray(frame).save(output_dir / f"{phase}_{camera}.png")
        print(
            {
                "suite": args.suite,
                "task_id": args.task_id,
                "instruction": task.language,
                "cameras": {
                    name: {
                        "shape": frame.shape,
                        "dtype": str(frame.dtype),
                        "range": (int(frame.min()), int(frame.max())),
                    }
                    for name, frame in observation.images.items()
                },
                "state_shape": observation.state["state"].shape,
                "reset_refresh_mad": deltas,
                "output_dir": args.output_dir,
            },
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        embodiment.close()


if __name__ == "__main__":
    main()
