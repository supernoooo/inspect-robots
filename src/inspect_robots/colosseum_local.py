"""Select native LLM pairs locally and publish real recordings through import APIs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random as random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from inspect_robots import colosseum as bridge


class LocalRunSession:
    """Validate local execution events; no Router connection or acknowledgments."""

    def __init__(self, config: Any, assignment: dict[str, Any], trial: dict[str, Any]):
        self.preparation = {"model": trial["model"], "task": assignment["task"]}
        self.failure = None
        self.started = False
        self.steps = 0

    def prepare(self) -> dict[str, Any]:
        """Return the selected model without a network delivery."""
        return dict(self.preparation["model"])

    def ready(self) -> None:
        """Mark a bound native agent ready for local recording."""
        self.started = True

    def event(self, kind: str, reply: str, **fields: Any) -> None:
        """Check contiguous executed steps without sending them to Router."""
        if not self.started or kind != "step" or fields.get("step") != self.steps:
            raise ValueError("Invalid local execution sequence")
        self.steps += 1

    def finish(self) -> None:
        """Check local completion after native cleanup."""
        if not self.started or self.steps < 1:
            raise ValueError("Local run has no executed actions")

    def close(self) -> None:
        """Release no network resources; this session is entirely local."""
        pass


def _choice(prompt: str, choices: set[str]) -> str:
    while True:
        value = input(prompt).strip()
        if value in choices:
            return value
        print("Choose: " + ", ".join(sorted(choices)))


def _catalog(config: Any, options: Any, names: list[str] | None) -> list[dict[str, Any]]:
    if not options.runtime_config:
        raise ValueError("--runtime-config is required for unfinished local runs")
    catalog = bridge.read_catalog(
        options.runtime_config, control_hz=bridge.client_control_hz(config)
    )
    if names is None:
        available = set()
        for model in catalog:
            provider = next(
                value[0]
                for prefix, value in bridge._PROVIDERS.items()
                if model["name"].startswith(prefix)
            )
            if config.llm_api_keys.get(provider) or config.llm_account_logins.get(provider):
                available.add(model["name"])
        if len(available) < 2:
            raise ValueError("Automatic local pairing requires credentials for at least two LLMs")
        names = random.SystemRandom().sample(sorted(available), 2)
    selected = []
    for name in names:
        matches = [model for model in catalog if model["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Local model {name} needs exactly one runtime entry")
        # Every selected model must have credentials before either arm is initialized.
        bridge.validate_catalog_keys(config, matches)
        selected.extend(matches)
    return selected


def _validate_native(options: Any, task: dict[str, Any], root: Path, config: Any) -> None:
    from inspect_robots.cli import _parse_kvs

    preview = bridge.native_args(
        options, task, root / "preview", control_hz=bridge.client_control_hz(config)
    )
    settings = _parse_kvs(preview.policy_args)
    invalid = set(settings) - bridge._AGENT_OPTIONS
    if invalid:
        raise ValueError("Unsupported native -P options: " + ", ".join(sorted(invalid)))
    speed = float(settings.get("max_speed_frac", 0.05))
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("Native speed must be finite and in (0, 1]")


def _write_score_csv(path: Path, run_id: str, score: int, recorded: str) -> None:
    info = json.loads((path / "meta/info.json").read_text())
    target = path / "eval_trials.csv"
    if target.exists():
        with target.open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 1 or rows[0]["run"] != run_id or rows[0]["score"] != str(score):
            raise ValueError("Existing import CSV differs from saved local scoring")
        return
    with target.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["episode_index", "run", "score", "frames", "saved_at"])
        writer.writerow([0, run_id, score, info["total_frames"], recorded])


def validate_completed(
    local: dict[str, Any], folder: Path, config: Any, *, require_export: bool = True
) -> None:
    """Submission never initializes hardware, scores missing trials or trusts partial runs."""
    if local.get("bridge") != "inspect-robots-local-v1" or local.get("aborted"):
        raise ValueError("Local pair is missing or aborted")
    assignment = local["assignment"]
    if assignment["test"] != config.test or not local.get("result"):
        raise ValueError("Local mode/test flag differs or final scores have not been saved")
    for run in assignment["runs"]:
        bridge.validate_assigned_llm(run["model"])
        path = folder / bridge.safe_id(run["id"])
        marker = path / "bridge-trial.json"
        completed = json.loads(marker.read_text()) if marker.exists() else {}
        if not completed.get("native_complete") or not completed.get("local_finished"):
            raise ValueError("Cannot submit an unfinished local run")
        score = local.get("outcomes", {}).get(run["side"], {}).get("score_0_4")
        if type(score) is not int or not 0 <= score <= 4:
            raise ValueError("Every local run needs a saved score from 0 to 4")
        if require_export and not (path / "lerobot/meta/info.json").is_file():
            raise ValueError("Local run has no completed LeRobot dataset")


def run_local_pair(config: Any, options: Any) -> int:
    """Run an explicitly chosen pair, or publish its saved evidence after review."""
    from colosseum_client.evaluation import api_url, save
    from colosseum_client.lerobot_export import export_trial

    if config.robot_type != "yam" or set(config.cameras) - set(bridge._CAMERAS):
        raise ValueError("Local pair requires Yam and standard recording cameras")
    if not config.recording:
        raise ValueError("Local pair requires recording: true")
    if options.excel:
        from inspect_robots.colosseum_excel import validate_target

        validate_target(options.excel)
    if options.track not in {None, "open"} or config.track not in {"", "open"}:
        raise ValueError("Local pair uses Open Track manual evaluation/import APIs")
    root = Path(config.evaluation_dir).expanduser().resolve()
    selected_id = options.resume or options.abort
    if selected_id:
        folder = root / bridge.safe_id(selected_id)
        manifest = folder / "manifest.json"
        local = json.loads(manifest.read_text())
        if local.get("bridge") != "inspect-robots-local-v1":
            raise ValueError("This evaluation was allocated by Router; it is not a local pair")
        if options.model_a or options.model_b:
            raise ValueError("Saved local model identities cannot be changed during resume")
        if options.abort:
            if (
                options.resume
                or options.dry_run
                or not options.reason
                or not options.reason.strip()
            ):
                raise ValueError(
                    "Local --abort requires --reason and cannot combine with resume/dry-run"
                )
            if local.get("remote_evaluation_id"):
                raise ValueError("Local abort does not delete a published web evaluation")
            local.update(aborted=True, abort_reason=options.reason)
            save(manifest, local)
            print(f"Local pair {selected_id} aborted; data retained; no Router request sent.")
            return 0
        if local.get("aborted"):
            raise ValueError("Local pair is aborted")
        if local.get("record_only") and not local.get("result") and not options.record_only:
            raise ValueError("This recording awaits scoring in --review")
        if options.record_only and (local.get("submitted") or local.get("remote_evaluation_id")):
            raise ValueError("Cannot record a pair after publication starts")
        assignment = local["assignment"]
        if assignment["test"] != config.test:
            raise ValueError("Local test flag cannot be changed during resume")
        if options.submit_only:
            validate_completed(local, folder, config, require_export=False)
        catalog = []
        unfinished = [
            run["model"]["name"] for run in assignment["runs"] if run["state"] != "finished"
        ]
        if unfinished and not options.submit_only:
            catalog = _catalog(config, options, unfinished)
            _validate_native(options, assignment["task"], root, config)
    else:
        if bool(options.model_a) != bool(options.model_b) or (
            options.model_a and options.model_a == options.model_b
        ):
            raise ValueError(
                "Specify two different --model-a/--model-b LLMs, or omit both for automatic pairing"
            )
        names = [options.model_a, options.model_b] if options.model_a else None
        catalog = _catalog(config, options, names)
        names = [model["name"] for model in catalog]
        instruction = config.instruction or (
            "validation only" if options.dry_run else input("Instruction: ").strip()
        )
        if not instruction:
            raise ValueError("Instruction cannot be empty")
        task = {
            "instruction": instruction,
            "max_steps": config.max_trial_steps,
            "cameras": list(config.cameras),
        }
        _validate_native(options, task, root, config)
        if options.dry_run:
            print("Local LLM pair validated; no Router assignment, model inference or hardware.")
            print("Models: " + " vs ".join(names))
            print(f"Local data: {root}")
            return 0
        created = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
        evaluation_id = "local_" + uuid.uuid4().hex
        folder = root / evaluation_id
        manifest = folder / "manifest.json"
        assignment = {
            "id": evaluation_id,
            "state": "local",
            "robot_id": "yam",
            "track": "open",
            "inference_mode": "inspect-local",
            "test": config.test,
            "recording": True,
            "institution": config.institution,
            "evaluator": config.evaluator or input("Evaluator: ").strip(),
            "scene": config.scene or input("Scene / setup: ").strip(),
            "task": task,
            "runs": [
                {
                    "id": "run_" + uuid.uuid4().hex,
                    "side": side,
                    "state": "assigned",
                    "model": bridge.public_model(model),
                    "policy_id": model["bridge_options"].get(
                        "policy_id", f"{model['name']}-yam@{model['revision']}"
                    ),
                }
                for side, model in zip(("A", "B"), catalog, strict=True)
            ],
        }
        local = {
            "bridge": "inspect-robots-local-v1",
            "model_selection": "explicit" if options.model_a else "random",
            "assignment": assignment,
            "outcomes": {},
            "created": created,
            "submission_deferred": bool(options.defer_submit),
            "task_id": "inspect-yam-" + hashlib.sha256(instruction.encode()).hexdigest()[:12],
        }
        save(manifest, local)
    if options.dry_run:
        print("Saved local pair configuration validated; no inference, hardware or upload.")
        return 0
    if options.excel:
        from inspect_robots.colosseum_excel import validate_target

        local["excel_path"] = str(validate_target(options.excel))
    if local.get("excel_path"):
        local["excel_router"] = api_url(config)
    if options.defer_submit:
        local["submission_deferred"] = True
    if options.record_only:
        local.update(record_only=True, review="pending", submission_deferred=True)
    save(manifest, local)
    print(f"Local LLM evaluation: {assignment['id']}; resume with --resume {assignment['id']}.")
    print("Models: " + " vs ".join(run["model"]["name"] for run in assignment["runs"]))
    for run in assignment["runs"]:
        path = folder / run["id"]
        if run["state"] != "finished":
            if options.submit_only:
                raise ValueError("--submit-only cannot execute a local trial")
            marker = path / "bridge-trial.json"
            done = json.loads(marker.read_text()) if marker.exists() else {}
            if not done.get("native_complete") or not done.get("local_finished"):
                if path.exists():
                    raise ValueError(
                        "Interrupted local trial artifacts exist; abort this local pair "
                        "instead of rerunning its ID"
                    )
                print(f"\n{run['side']}: restore the initial scene for {run['model']['name']}.")
                bridge.execute_trial(
                    config, assignment, run, path, options, catalog, session_factory=LocalRunSession
                )
            run["state"] = "finished"
            save(manifest, local)
        context = json.loads((path / "recording-context.json").read_text())
        if context.get("native_execution"):
            local.setdefault("execution", {})[run["id"]] = context["native_execution"]
            save(manifest, local)
        if options.record_only:
            continue
        if run["side"] not in local["outcomes"]:
            score = int(
                _choice(
                    f"{run['side']} score (0=no progress, 4=fully completed; integer 0-4): ",
                    set("01234"),
                )
            )
            local["outcomes"][run["side"]] = {
                "success": score == 4,
                "partial_success": score / 4,
                "score_0_4": score,
            }
            save(manifest, local)
        from inspect_robots.colosseum_evidence import refresh_outcome

        refresh_outcome(path, local["outcomes"][run["side"]])
        export_trial(
            path,
            list(context["camera_map"]),
            fps=context["fps"],
            robot_type="yam",
            task=assignment["task"]["instruction"],
            metadata={
                "assignment_id": assignment["id"],
                "run_id": run["id"],
                "side": run["side"],
                "track": "open",
                "test": assignment["test"],
                "institution": assignment["institution"],
                "scene": assignment["scene"],
                "evaluator": assignment["evaluator"],
                "outcome": local["outcomes"][run["side"]],
                "action_space": "joint_position",
                "execution_enabled": True,
                "source": "inspect-robots local LLM pair",
                "recording_context": context,
            },
        )
        _write_score_csv(
            path / "lerobot",
            run["id"],
            local["outcomes"][run["side"]]["score_0_4"],
            local["created"],
        )
        bridge.sync_excel(manifest, local)
    if options.record_only:
        print("A/B runs recorded locally; no scoring, dataset export or uploads.")
        print(f"Review: inspect-robots colosseum --client-config {options.client_config} --review")
        return 0
    if "result" not in local:
        local["result"] = {
            "outcomes": local["outcomes"],
            "preference": _choice("Preference (A/B/tie): ", {"A", "B", "tie"}),
            "feedback": input("Feedback (optional): ").strip(),
            "confidence": _choice(
                "Preference confidence (clear/close/uncertain): ", {"clear", "close", "uncertain"}
            ),
        }
        save(manifest, local)
    bridge.sync_excel(manifest, local)
    if local.get("submission_deferred") and not options.submit_only:
        print("All runs and scores saved locally; no web result or dataset uploaded.")
        print(
            f"After review: inspect-robots colosseum --client-config {options.client_config} "
            f"--submit-only --resume {assignment['id']}"
        )
    else:
        validate_completed(local, folder, config)
        from inspect_robots.colosseum_import import publish_local_pair

        publish_local_pair(config, manifest, local)
    return 1 if local.get("excel_pending") else 0
