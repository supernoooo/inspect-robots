"""Check model authentication, vision and tool proposals without opening hardware."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from _run_so101 import configured_so101
from inspect_robots_agent import LLMAgentPolicy

from inspect_robots import Observation, Scene


def main() -> int:
    """Use a synthetic front image and joint state; never reset/step the arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wire", choices=("claude-code", "responses", "chat", "interactions"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--claude-command")
    arguments = parser.parse_args()
    options: dict[str, Any] = {
        "wire": arguments.wire,
        "model": arguments.model,
        "images": "always",
        "max_llm_calls": 2,
        "wire_capture": False,
    }
    if arguments.claude_command:
        options["claude_command"] = arguments.claude_command
    arm = configured_so101(camera_configs={})
    policy = None
    try:
        policy = LLMAgentPolicy(**options)
        policy.bind(arm.info)
        policy.reset(
            Scene(
                id="synthetic-login-probe",
                instruction=(
                    "This is a synthetic connection probe. "
                    "The front image is gray with a blue square. "
                    "Observe the image and joints, then call give_up. In its reason, describe the "
                    "square's color. Do not request movement. No robot is connected."
                ),
            )
        )
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        frame[160:320, 240:400] = (0, 0, 255)
        observation = Observation(
            state={"joint_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])},
            images={"front": frame},
        )
        chunk = policy.act(observation)
        last = chunk.actions[-1]
        if not last.meta.get("request_stop"):
            raise RuntimeError("Probe returned a motion proposal; it was not executed")
        print(
            json.dumps(
                {
                    "wire": arguments.wire,
                    "model": arguments.model,
                    "stop": dict(last.meta),
                    "usage": policy._usage_totals,
                },
                indent=2,
            )
        )
        print("Model probe finished. No serial port/camera was opened; no action was executed.")
    finally:
        if policy is not None:
            policy._client.close()
        arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
