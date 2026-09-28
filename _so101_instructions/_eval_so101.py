"""Launch one SO-101 bottle evaluation, or probe its model without hardware."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
PROVIDERS = {
    "claude": ("claude-code", "claude-opus-5-5", None),
    "gpt": ("responses", "openai/gpt-6-astra", "OPENAI_API_KEY"),
    "gemini": ("interactions", "google/gemini-3.8-flash", "GEMINI_API_KEY"),
    "grok": ("responses", "x-ai/grok-4.7", "XAI_API_KEY"),
}
PREFIXES = {"gpt": "openai/", "gemini": "google/", "grok": "x-ai/"}
BOTTLE_INSTRUCTION = (
    "Pick up the bottle standing on the table, lift it slightly clear of the table, "
    "and place it upright back at its original position. Remember its starting "
    "position from the first observation. Release the gripper after the bottle "
    "is supported by the table, then finish."
)


def check_arm_settings(parser: argparse.ArgumentParser) -> None:
    """Require existing calibration and explicit native-unit action bounds."""
    calibration_dir = Path(os.environ["SO101_CALIBRATION_DIR"]).expanduser()
    calibration_file = calibration_dir / f"{os.environ['SO101_ROBOT_ID']}.json"
    if not calibration_file.is_file():
        parser.error(
            f"Calibration file missing: {calibration_file}. "
            "Set SO101_CALIBRATION_DIR and SO101_ROBOT_ID to your existing calibration, "
            "or follow the lerobot-calibrate instructions first."
        )
    bounds = []
    for variable in ("SO101_JOINT_LOW", "SO101_JOINT_HIGH"):
        try:
            values = json.loads(os.environ[variable])
        except (KeyError, ValueError):
            parser.error(f"Set {variable} to a JSON array of six calibrated joint limits.")
        if (
            not isinstance(values, list)
            or len(values) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
        ):
            parser.error(f"{variable} must be a JSON array of six finite numbers.")
        bounds.append(values)
    if any(low >= high for low, high in zip(*bounds, strict=True)):
        parser.error("Every SO101_JOINT_LOW value must be below its SO101_JOINT_HIGH value.")
    if not 0 <= bounds[0][5] < bounds[1][5] <= 100:
        parser.error("The gripper limits (sixth values) must be within 0..100.")


def main() -> int:
    """Build the existing CLI command; only explicit invocation starts a run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=tuple(PROVIDERS))
    parser.add_argument("--model", help="Exact model ID; overrides the provider default")
    parser.add_argument("--claude-command", help="Path to the official Claude Code CLI")
    parser.add_argument("--config", default=str(ROOT / ".so101/config.ini"))
    parser.add_argument("--instruction", default=BOTTLE_INSTRUCTION)
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-action-delta", type=float, default=1.0)
    parser.add_argument("--max-speed-frac", type=float, default=0.05)
    parser.add_argument("--log-dir")
    parser.add_argument(
        "--probe", action="store_true", help="Call the model with a synthetic image; no hardware"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command; no model call or hardware"
    )
    args = parser.parse_args()
    if args.max_llm_calls < 1 or args.max_steps < 1:
        parser.error("--max-llm-calls and --max-steps must be positive.")
    if args.max_action_delta <= 0 or not 0 < args.max_speed_frac <= 1:
        parser.error("--max-action-delta must be > 0 and --max-speed-frac must be in (0, 1].")

    wire, default_model, key_env = PROVIDERS[args.provider]
    model = args.model or default_model
    if args.provider != "claude" and "/" not in model:
        model = PREFIXES[args.provider] + model
    if args.provider == "claude" and "/" in model:
        parser.error("Claude Code expects a bare Claude model ID, e.g. claude-opus-5-5.")
    if args.provider != "claude":
        prefix = model.partition("/")[0]
        allowed = {PREFIXES[args.provider].rstrip("/")}
        if args.provider == "grok":
            allowed.add("xai")
        if prefix not in allowed or not model.partition("/")[2]:
            parser.error(f"Model must belong to {args.provider}: {default_model}")

    claude_command = None
    if args.provider == "claude":
        requested_command = args.claude_command or os.environ.get("SO101_CLAUDE_COMMAND")
        if requested_command:
            claude_command = str(Path(requested_command).expanduser().absolute())
        else:
            claude_command = shutil.which("claude")

    if args.probe:
        command = [
            str(PYTHON),
            str(ROOT / "_so101_instructions/_so101_llm_probe.py"),
            "--wire",
            wire,
            "--model",
            model,
        ]
        if claude_command:
            command += ["--claude-command", claude_command]
    else:
        config = str(Path(args.config).expanduser().absolute())
        robot_id = os.environ.get("SO101_ROBOT_ID") or "my_so101"
        calibration_dir = str(
            Path(os.environ.get("SO101_CALIBRATION_DIR") or ROOT / ".so101/calibration")
            .expanduser()
            .absolute()
        )
        model_dir = model.replace("/", "_")
        log_dir = args.log_dir or f"logs/so101/{args.provider}/{model_dir}/bottle"
        command = [
            str(PYTHON),
            str(ROOT / "_so101_instructions/_run_so101.py"),
            "run",
            "--config",
            config,
            "--embodiment",
            "so101_configured",
            "--policy",
            "agent",
            "-E",
            f"robot_id={robot_id}",
            "-E",
            f"calibration_dir={calibration_dir}",
            "-E",
            "use_degrees=True",
            "-P",
            f"wire={wire}",
            "-P",
            f"model={model}",
            "-P",
            "images=always",
            "-P",
            f"max_speed_frac={args.max_speed_frac}",
            "-P",
            f"max_llm_calls={args.max_llm_calls}",
            "--instruction",
            args.instruction,
            "--max-action-delta",
            str(args.max_action_delta),
            "--max-steps",
            str(args.max_steps),
            "--epochs",
            "1",
            "--grader",
            "operator",
            "--scorer",
            "operator",
            "--store-frames",
            "--no-rerun",
            "--no-rerun-save",
            "--fail-on-error",
            "1",
            "--log-dir",
            log_dir,
        ]
        for name in ("joint_low", "joint_high"):
            if value := os.environ.get(f"SO101_{name.upper()}"):
                command += ["-E", f"{name}={value}"]
        if claude_command:
            command += ["-P", f"claude_command={claude_command}"]

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not PYTHON.is_file():
        parser.error(f"Project Python missing: {PYTHON}")
    if key_env and not os.environ.get(key_env):
        parser.error(f"Export {key_env} before calling {args.provider}.")
    if args.provider == "claude":
        if not claude_command:
            parser.error(
                "Claude Code CLI not found. Put claude on PATH or pass --claude-command PATH."
            )
        if not os.access(claude_command, os.X_OK):
            parser.error(f"Claude Code executable missing: {claude_command}")
    if not args.probe:
        if not Path(config).is_file():
            parser.error(f"Config missing: {config}; run _run_so101.py setup first.")
        os.environ["SO101_ROBOT_ID"] = robot_id
        os.environ["SO101_CALIBRATION_DIR"] = calibration_dir
        check_arm_settings(parser)
    os.chdir(ROOT)
    os.execv(str(PYTHON), command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
