"""Run one attended YAM agent evaluation, or probe its model without hardware."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSPECT_ROBOTS = ROOT / ".venv/bin/inspect-robots"


@dataclass(frozen=True)
class Provider:
    wire: str
    model: str
    key_env: str | None = None
    prefix: str | None = None


PROVIDERS = {
    "claude": Provider("claude-code", "claude-opus-5-5"),
    "claude-api": Provider(
        "messages", "anthropic/claude-opus-5-5", "ANTHROPIC_API_KEY", "anthropic/"
    ),
    "gpt": Provider("responses", "openai/gpt-6-astra", "OPENAI_API_KEY", "openai/"),
    "gemini": Provider("interactions", "google/gemini-3.8-flash", "GEMINI_API_KEY", "google/"),
    "grok": Provider("responses", "x-ai/grok-4.7", "XAI_API_KEY", "x-ai/"),
}

DEFAULT_INSTRUCTION = (
    "Use the appropriate YAM arm to pick up the bottle standing on the table, "
    "lift it slightly clear of the table, and place it upright back at its original "
    "position. Keep the other arm clear of the workspace. Release the gripper only "
    "after the table supports the bottle, then finish."
)

PROBE_INSTRUCTION = (
    "This is a no-hardware connection probe in the CubePick simulator. Observe the "
    "camera image and state, then call give_up without requesting any movement. In "
    "the reason, briefly describe what you see."
)


def normalize_model(parser: argparse.ArgumentParser, provider_name: str, model: str) -> str:
    """Add and validate the direct-provider prefix used by the agent plugin."""
    provider = PROVIDERS[provider_name]
    if provider_name == "claude":
        if model.startswith("anthropic/"):
            model = model.removeprefix("anthropic/")
        if not model or "/" in model:
            parser.error("Claude Code needs a bare model alias or ID, for example sonnet.")
        return model

    assert provider.prefix is not None
    if "/" not in model:
        model = provider.prefix + model
    allowed_prefixes = {provider.prefix.rstrip("/")}
    if provider_name == "grok":
        allowed_prefixes.add("xai")
    prefix, separator, suffix = model.partition("/")
    if not separator or not suffix or prefix not in allowed_prefixes:
        parser.error(f"Model must belong to {provider_name}: {provider.model}")
    return model


def resolve_claude_command(argument: str | None) -> str | None:
    """Return an explicit CLI path only when the operator supplied or installed one."""
    requested = argument or os.environ.get("YAM_CLAUDE_COMMAND")
    if requested:
        return str(Path(requested).expanduser().absolute())
    return shutil.which("claude")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=tuple(PROVIDERS))
    parser.add_argument("--model", help="Exact provider model ID; overrides the default")
    parser.add_argument("--claude-command", help="Path to the official Claude Code CLI")
    parser.add_argument("--config", default=str(ROOT / ".yam/config.ini"))
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=0.05,
        help="Per-step YAM joint limit in radians (default: 0.05)",
    )
    parser.add_argument("--max-speed-frac", type=float, default=0.05)
    parser.add_argument(
        "--log-dir", help="Root for dated run folders (default: logs/yam/<provider>)"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Test authentication, vision, and tool output in CubePick; never open YAM",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the exact command and do not run it"
    )
    args = parser.parse_args()

    if args.max_llm_calls < 1 or args.max_steps < 1:
        parser.error("--max-llm-calls and --max-steps must be positive.")
    if args.max_action_delta <= 0 or not 0 < args.max_speed_frac <= 1:
        parser.error("--max-action-delta must be > 0 and --max-speed-frac must be in (0, 1].")

    provider = PROVIDERS[args.provider]
    model = normalize_model(parser, args.provider, args.model or provider.model)
    claude_command = (
        resolve_claude_command(args.claude_command) if args.provider == "claude" else None
    )
    effective_max_llm_calls = min(args.max_llm_calls, 3) if args.probe else args.max_llm_calls

    policy_args = [
        "-P",
        f"wire={provider.wire}",
        "-P",
        f"model={model}",
        "-P",
        "images=always",
        "-P",
        f"max_llm_calls={effective_max_llm_calls}",
    ]
    if claude_command:
        policy_args += ["-P", f"claude_command={claude_command}"]

    log_dir = args.log_dir or f"logs/yam/{args.provider}"
    if args.probe:
        command = [
            str(INSPECT_ROBOTS),
            "run",
            "--embodiment",
            "cubepick",
            "--policy",
            "agent",
            *policy_args,
            "--instruction",
            PROBE_INSTRUCTION,
            "--max-steps",
            "2",
            "--epochs",
            "1",
            "--grader",
            "none",
            "--scorer",
            "episode_length",
            "--no-prompt",
            "--no-store-frames",
            "--no-rerun",
            "--no-rerun-save",
            "--log-dir",
            log_dir,
        ]
    else:
        config = str(Path(args.config).expanduser().absolute())
        command = [
            str(INSPECT_ROBOTS),
            "run",
            "--config",
            config,
            "--embodiment",
            "yam_arms",
            "--policy",
            "agent",
            *policy_args,
            "-P",
            f"max_speed_frac={args.max_speed_frac}",
            "-E",
            "auto_start=False",
            "-E",
            "unattended=False",
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

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not INSPECT_ROBOTS.is_file():
        parser.error(f"Inspect Robots executable missing: {INSPECT_ROBOTS}")
    if provider.key_env and not os.environ.get(provider.key_env):
        parser.error(f"Export {provider.key_env} before calling {args.provider}.")
    if args.provider == "claude":
        if not claude_command:
            parser.error(
                "Claude Code CLI not found. Put claude on PATH or pass --claude-command PATH."
            )
        if not os.access(claude_command, os.X_OK):
            parser.error(f"Claude Code executable is missing or not executable: {claude_command}")
    if not args.probe and not Path(config).is_file():
        parser.error(f"YAM config missing: {config}. Run: inspect-robots setup --config {config}")

    if not args.probe:
        print(
            "Starting attended YAM evaluation. Keep the E-stop ready and confirm the CLI "
            "prompt only after the workspace is clear.",
            flush=True,
        )
    os.chdir(ROOT)
    os.execv(str(INSPECT_ROBOTS), command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
