"""Run native Inspect evaluations with Colosseum assignments and dataset uploads.

Optional Colosseum dependencies are loaded only when this command is selected.
The Router's local protocol is used for actual native-policy readiness, executed
steps, heartbeats and finish acknowledgments; no inference service is simulated.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from inspect_robots.logging.sink import NullSink

_PUBLIC_FIELDS = (
    "model_type",
    "name",
    "url",
    "revision",
    "action_space",
    "action_dim",
    "control_hz",
    "max_horizon",
)
_CAMERAS = {"head_image": "top_cam", "left_image": "left_cam", "right_image": "right_cam"}
_JOINT_COMMAND_INFO_KEY = "yam_joint_command"
_LLM_MODELS = frozenset({"grok-4.7", "gpt-6-astra", "claude-opus-5-5"})
_PROVIDERS = {
    "gpt-": ("openai", "openai", "OPENAI_API_KEY", "responses"),
    "grok-": ("xai", "x-ai", "XAI_API_KEY", "responses"),
    "claude-": ("anthropic", "anthropic", "ANTHROPIC_API_KEY", "messages"),
}
_AGENT_OPTIONS = {
    "wire",
    "effort",
    "max_llm_calls",
    "max_output_tokens",
    "max_speed_frac",
    "images",
    "image_horizon",
    "temperature",
    "transcript_echo",
    "depth",
    "prior_learnings",
}


def add_parser(sub: Any) -> None:
    """Register a dependency-free CLI parser; native run flags follow ``--``."""
    parser = sub.add_parser("colosseum", help="run Inspect agents and upload Colosseum evaluations")
    parser.add_argument("--client-config", required=True, help="Colosseum Robot Client YAML")
    parser.add_argument(
        "--runtime-config", help="Policy Server YAML catalog (required for new trials)"
    )
    parser.add_argument("--resume", help="resume saved scoring, export, submission or uploads")
    parser.add_argument("--abort", help="abort an assignment without executing a robot")
    parser.add_argument("--reason", help="reason required with --abort")
    parser.add_argument("--track", choices=("open", "fine-tuning"))
    parser.add_argument("--instruction")
    parser.add_argument("--scene")
    parser.add_argument("--evaluator")
    parser.add_argument(
        "--local-pair", action="store_true", help="select an LLM pair locally; no Router assignment"
    )
    parser.add_argument(
        "--model-a",
        choices=sorted(_LLM_MODELS),
        help="local A model; omit both models for random pairing",
    )
    parser.add_argument(
        "--model-b",
        choices=sorted(_LLM_MODELS),
        help="local B model; omit both models for random pairing",
    )
    parser.add_argument("--excel", metavar="FILE.xlsx", help="update a local Excel workbook")
    parser.add_argument(
        "--export-excel-only",
        action="store_true",
        help="export saved scores locally; do not contact Router or initialize hardware",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="submit scores; retain datasets locally"
    )
    parser.add_argument(
        "--defer-submit",
        action="store_true",
        help="save scored runs locally for review; defer final result and dataset uploads",
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="record runs without scoring, exporting datasets or submitting; score in --review",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="open the local evaluation review server; no hardware or automatic uploads",
    )
    parser.add_argument("--review-port", type=int, default=8767)
    parser.add_argument("--review-open", action="store_true", help="open review in a browser")
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="submit and upload an already scored --resume evaluation; never execute hardware",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate config locally; no Router or hardware"
    )
    parser.add_argument(
        "inspect_args", nargs=argparse.REMAINDER, help="native Inspect run options after --"
    )


def load_dependencies() -> None:
    """Prefer the copied source in a checkout, or installed packages in a wheel."""
    root = Path(__file__).resolve().parents[2] / "integrations"
    for name in ("colosseum-client", "colosseum-policy-server"):
        source = root / name / "src"
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))
    try:
        import colosseum_client.evaluation
        import colosseum_client.lerobot_export
        import colosseum_client.local_protocol  # noqa: F401
        import websockets.sync.client  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install the bridge dependencies into your Inspect environment: "
            "uv pip install --python .venv/bin/python -e integrations/colosseum-client"
        ) from exc


def safe_id(value: Any) -> str:
    """Reject peer-supplied IDs that could escape local directories or API routes."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value):
        raise ValueError("Invalid Colosseum assignment/run ID")
    return value


def public_model(model: dict[str, Any]) -> dict[str, Any]:
    """Compare model identity without local endpoint or optional protocol defaults."""
    return {key: model[key] for key in _PUBLIC_FIELDS if key in model}


def client_control_hz(config: Any) -> int:
    """Use the Client's nominal control frequency for execution and recording."""
    hz = config.control_hz
    if type(hz) is not int or hz <= 0:
        raise ValueError("Client control_hz must be a positive integer")
    return hz


def read_catalog(path: str, *, control_hz: int | None = None) -> list[dict[str, Any]]:
    """Read registered cloud models from the copied Policy Server configuration."""
    import yaml
    from colosseum_policy_server.backends.base import LocalModel

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise ValueError("Runtime config must contain a models list")
    backend = value.get("backend", {})
    if backend.get("name", "auto") not in {"auto", "inspect_agent"}:
        raise ValueError("Native agent bridge requires backend.name: auto or inspect_agent")
    options = backend.get("options", {})
    shared = {k: v for k, v in options.items() if k not in {"llm", "vla"}}
    shared.update(options.get("llm", {}))
    models = []
    for model in value["models"]:
        if model.get("model_type") != "llm":
            raise ValueError(
                "This project is LLM-only; remove VLA entries from the runtime catalog"
            )
        if not all(key in model for key in _PUBLIC_FIELDS):
            raise ValueError("LLM catalog entry is missing a public model field")
        if model["name"] not in _LLM_MODELS:
            raise ValueError("This project supports only: " + ", ".join(sorted(_LLM_MODELS)))
        LocalModel.from_mapping(model)  # Validate against the copied Server's model contract.
        if model.get("endpoint") != "inprocess://inspect-agent":
            raise ValueError("LLM catalog entry must use endpoint: inprocess://inspect-agent")
        if model["action_space"] != "joint_position" or model["action_dim"] != 14:
            raise ValueError("This bridge currently supports YAM 14-D joint_position only")
        if type(model["control_hz"]) is not int or model["control_hz"] <= 0:
            raise ValueError("Registered model control_hz must be a positive integer")
        if control_hz is not None and model["control_hz"] != control_hz:
            raise ValueError("Runtime model control_hz must match Client control_hz")
        url = urlsplit(model["url"])
        if (
            url.scheme != "https"
            or not url.hostname
            or any((url.username, url.password, url.query, url.fragment))
        ):
            raise ValueError("LLM model URL must be a credential-free HTTPS API base")
        if not any(model["name"].startswith(prefix) for prefix in _PROVIDERS):
            raise ValueError("Unsupported LLM model name in catalog")
        merged = {**shared, **model.get("backend_options", {})}
        if merged.get("robot_type", "yam") != "yam":
            raise ValueError("Native bridge catalog must target robot_type: yam")
        models.append({**model, "bridge_options": merged})
    if not models:
        raise ValueError("Runtime config has no LLM models")
    return models


def validate_catalog_keys(config: Any, catalog: list[dict[str, Any]]) -> None:
    """Validate API keys/account logins and URLs before requesting an assignment."""
    from colosseum_client.llm_credentials import API_URLS

    eligible = []
    for model in catalog:
        provider = next(
            value[0] for prefix, value in _PROVIDERS.items() if model["name"].startswith(prefix)
        )
        account = config.llm_account_logins.get(provider)
        if not config.llm_api_keys.get(provider) and not account:
            continue
        declared = config.llm_api_base_urls.get(provider, API_URLS[provider]).rstrip("/")
        if declared != model["url"].rstrip("/"):
            raise ValueError(f"llm_api_base_urls.{provider} must match catalog URL {model['url']}")
        if account:
            from inspect_robots_agent._claude_code import ClaudeCodeClient

            if model["url"].rstrip("/") != API_URLS["anthropic"]:
                raise ValueError("Claude Code login cannot execute a relay-registered model")
            client = ClaudeCodeClient(
                model["name"],
                command=model["bridge_options"].get("claude_command"),
                timeout_s=model["bridge_options"].get("claude_timeout_s", 120),
                env=account_environment(config),
            )
            client.close()
        eligible.append(model)
    if not eligible:
        raise ValueError("No catalog model has a configured Client LLM key or account login")


def account_environment(config: Any) -> dict[str, str]:
    """Keep CLI login available without passing Router/provider API credentials."""
    secrets = {config.token, *config.llm_api_keys.values()}
    excluded = {value[2] for value in _PROVIDERS.values()} | {
        "COLOSSEUM_OPENAI_KEY",
        "COLOSSEUM_XAI_KEY",
        "COLOSSEUM_ANTHROPIC_KEY",
    }
    return {k: v for k, v in os.environ.items() if k not in excluded and v not in secrets}


def validate_assigned_llm(identity: dict[str, Any]) -> None:
    """Reject server-selected VLA/unlisted models before constructing any agent or robot."""
    if identity.get("model_type", "vla") != "llm":
        parsed = urlsplit(identity.get("url", ""))
        source = (parsed.hostname or "unknown") + parsed.path
        raise ValueError(
            f"Router assigned a VLA model ({source}). This project evaluates only "
            + ", ".join(sorted(_LLM_MODELS))
            + ". No agent or robot was initialized. Router must support an LLM-only "
            "assignment pool/filter; configuring Client LLM keys does not exclude VLA."
        )
    if identity.get("name") not in _LLM_MODELS:
        raise ValueError(
            "Router selected an unsupported LLM; this project supports only: "
            + ", ".join(sorted(_LLM_MODELS))
        )


def make_agent(model: dict[str, Any], config: Any, overrides: dict[str, Any]) -> Any:
    """Construct the exact assigned API policy without writing credentials to logs."""
    from inspect_robots.registry import resolve

    provider, _prefix, key_env, wire = next(
        value for name, value in _PROVIDERS.items() if model["name"].startswith(name)
    )
    key = config.llm_api_keys.get(provider)
    account = config.llm_account_logins.get(provider)
    if not key and not account:
        raise ValueError(f"Client config needs llm_api_keys.{provider} for {model['name']}")
    invalid = set(overrides) - _AGENT_OPTIONS
    if invalid:
        raise ValueError(f"Unsupported native -P options: {', '.join(sorted(invalid))}")
    options = model["bridge_options"]
    kwargs = {k: v for k, v in options.items() if k in _AGENT_OPTIONS}
    kwargs.setdefault("max_speed_frac", 0.05)
    kwargs.setdefault("max_llm_calls", 60)
    kwargs.setdefault("images", "always")
    kwargs.update(overrides)
    wire = kwargs.pop("wire", "claude-code" if account else wire)
    if account:
        if wire != "claude-code":
            raise ValueError("Claude Code account login requires wire=claude-code")
        if provider != "anthropic" or model["url"].rstrip("/") != "https://api.anthropic.com/v1":
            raise ValueError("Claude Code login requires an official Anthropic model registration")
        return resolve(
            "policy",
            "agent",
            model=f"anthropic/{model['name']}",
            wire="claude-code",
            env=account_environment(config),
            claude_command=options.get("claude_command"),
            claude_timeout_s=options.get("claude_timeout_s", 120),
            wire_capture=False,
            **kwargs,
        )
    if wire == "claude-code":
        raise ValueError("wire=claude-code requires Client llm_account_logins.anthropic")
    base = model["url"].rstrip("/")
    if provider == "anthropic" and not urlsplit(base).path.strip("/"):
        base += "/v1"
    transport = None
    if provider == "anthropic" and options.get("api_auth", "api_key") == "bearer":
        from colosseum_policy_server.backends.llm_transport import BearerTransport

        transport = BearerTransport(key)
    try:
        agent = resolve(
            "policy",
            "agent",
            # An explicit endpoint bypasses provider-prefix stripping.
            # The assigned name is already the endpoint's native model ID.
            model=model["name"],
            wire=wire,
            base_url=base,
            api_key_env=key_env,
            env={key_env: key},
            transport=transport,
            wire_capture=False,
            **kwargs,
        )
        # Match the copied backend's relay compatibility handling.
        client = getattr(agent, "_client", None)
        if (
            urlsplit(base).hostname != "api.openai.com"
            and client is not None
            and hasattr(client, "_cache_anchors")
        ):
            client._cache_anchors = False
        return agent
    except BaseException:
        if transport is not None:
            transport.close()
        raise


class RouterSession:
    """Synchronous local lifecycle with a serialized background heartbeat."""

    def __init__(self, config: Any, assignment: dict[str, Any], trial: dict[str, Any]):
        self.config, self.assignment, self.trial = config, assignment, trial
        self.run_id = safe_id(trial["id"])
        self.ws: Any = None
        self.preparation: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.worker: threading.Thread | None = None
        self.failure: Exception | None = None
        self.started = False

    def prepare(self) -> dict[str, Any]:
        """Receive the Router-assigned identity before constructing any hardware."""
        from colosseum_client.evaluation import api_url
        from colosseum_client.local_protocol import decode_control
        from websockets.sync.client import connect

        parts = urlsplit(api_url(self.config))
        base = urlunsplit(
            (
                "wss" if parts.scheme == "https" else "ws",
                parts.netloc,
                parts.path.rstrip("/"),
                "",
                "",
            )
        )
        self.ws = connect(
            base + f"/api/eval/runs/{self.run_id}/local",
            proxy=None,
            open_timeout=10,
            additional_headers={
                "Authorization": f"Bearer {self.config.token}",
                "X-Colosseum-Institution": quote(self.config.institution.strip(), safe=""),
            },
            max_size=65536,
        )
        self.preparation = decode_control(self.ws.recv(timeout=30))
        expected = self.preparation
        if (
            expected.get("type") != "prepare"
            or expected.get("protocol_version") != 1
            or expected.get("run_id") != self.run_id
            or bool(expected.get("test", False)) != self.config.test
        ):
            raise ValueError("Invalid Router preparation")
        if expected.get("task", {}).get("instruction") != self.assignment["task"]["instruction"]:
            raise ValueError("Router preparation instruction differs from the assignment")
        model = expected.get("model")
        if not isinstance(model, dict):
            raise ValueError("Router preparation has no model descriptor")
        return model

    def exchange(self, message: dict[str, Any], expected: dict[str, Any]) -> None:
        """Require durable acknowledgments; a disconnect cannot silently publish a trial."""
        from colosseum_client.local_protocol import decode_control, encode_control

        if self.failure is not None:
            raise RuntimeError("Router lifecycle connection failed") from self.failure
        with self.lock:
            try:
                self.ws.send(encode_control(message))
                reply = decode_control(self.ws.recv(timeout=10))
                if reply != expected:
                    raise RuntimeError(f"Router rejected {message['type']} lifecycle event")
            except Exception as exc:
                self.failure = exc
                raise

    def ready(self) -> None:
        """Report a real, constructed and bound native policy as ready."""
        self.exchange(
            {
                "type": "ready",
                "run_id": self.run_id,
                "preparation_id": self.preparation["preparation_id"],
                "model": self.preparation["model"],
                "loaded": True,
                "verification_only": False,
            },
            {"type": "started", "run_id": self.run_id},
        )
        self.started = True
        self.worker = threading.Thread(
            target=self.heartbeat, name="colosseum-heartbeat", daemon=True
        )
        self.worker.start()

    def heartbeat(self) -> None:
        """Keep the assignment alive during slow API inference and operator input."""
        while not self.stopping.wait(10):
            try:
                self.event("heartbeat", "heartbeat")
            except Exception:
                return

    def event(self, kind: str, reply: str, **fields: Any) -> None:
        """Report a step or finish using the existing Client's exact message contract."""
        self.exchange(
            {"type": kind, "run_id": self.run_id, **fields},
            {"type": reply, "run_id": self.run_id, **fields},
        )

    def finish(self) -> None:
        """Acknowledge finish after successful recording and native hardware cleanup."""
        self.stop_heartbeat()
        self.event("finish", "finished")

    def stop_heartbeat(self) -> None:
        """Join the heartbeat before sending finish or closing the socket."""
        self.stopping.set()
        if self.worker is not None:
            self.worker.join(timeout=15)

    def close(self) -> None:
        """Close the local lifecycle socket without marking interrupted runs finished."""
        self.stop_heartbeat()
        if self.ws is not None:
            self.ws.close()


class ReportingPolicy:
    """Delegate native agent tools and transcripts while reporting real readiness."""

    def __init__(
        self,
        agent: Any,
        session: RouterSession,
        config: Any,
        cameras: dict[str, str],
        *,
        control_mode: str = "joint_pos",
    ):
        self.agent, self.session, self.robot_config, self.cameras = agent, session, config, cameras
        self.control_mode = control_mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def bind(self, info: Any) -> None:
        """Reject incompatible native embodiments before any reset or policy action."""
        model = self.session.preparation["model"]
        semantics = info.action_space.semantics
        if (
            semantics is None
            or semantics.control_mode != self.control_mode
            or self.control_mode not in {"joint_pos", "eef_abs_pose"}
            or model["action_space"] != "joint_position"
            or info.action_space.dim != model["action_dim"]
            or info.control_hz != model["control_hz"]
        ):
            raise ValueError(
                "Assigned model requires 14-D joint execution at its registered frequency; "
                "native control must match the selected joint or EEF interface"
            )
        if bool(info.is_simulated) != self.robot_config.test:
            raise ValueError("Native embodiment simulation flag differs from Colosseum test flag")
        available = {camera.name for camera in info.observation_space.cameras}
        if set(self.cameras.values()) - available:
            raise ValueError("Native embodiment is missing mapped recording cameras")
        self.agent.bind(
            dataclasses.replace(
                info,
                docs=(info.docs or "") + "\nColosseum action chunks may be truncated to the "
                "registered max_horizon; use the next measured state to replan.",
            )
        )
        self.session.ready()

    def act(self, observation: Any) -> Any:
        """Stop requesting motions after loss of the Router lifecycle connection."""
        if self.session.failure is not None:
            raise RuntimeError(
                "Router disconnected during native evaluation"
            ) from self.session.failure
        chunk = self.agent.act(observation)
        model = self.session.preparation["model"]
        if chunk.control_hz is not None and chunk.control_hz != model["control_hz"]:
            raise ValueError("Agent action chunk frequency differs from the registered model")
        if len(chunk.actions) > model["max_horizon"]:
            chunk = dataclasses.replace(chunk, actions=tuple(chunk.actions[: model["max_horizon"]]))
        return chunk


class CaptureClockEmbodiment:
    """Timestamp native observation receipt before LLM inference or action execution."""

    def __init__(self, embodiment: Any):
        semantics = embodiment.info.action_space.semantics
        if (
            semantics is not None
            and semantics.control_mode == "eef_abs_pose"
            and getattr(embodiment, "joint_command_info_key", None) != _JOINT_COMMAND_INFO_KEY
        ):
            raise ValueError(
                "EEF embodiment must expose post-IK joint command telemetry; "
                "install the current plugins/inspect-robots-yam-eef source"
            )
        self.embodiment = embodiment

    def __getattr__(self, name: str) -> Any:
        return getattr(self.embodiment, name)

    @staticmethod
    def stamp(observation: Any) -> Any:
        """Preserve sensor timestamps and add an explicit bridge receipt clock."""
        return dataclasses.replace(
            observation, extra={**observation.extra, "_colosseum_received_at": time.monotonic()}
        )

    def reset(self, scene: Any, *, seed: int | None = None) -> Any:
        """Capture the initial observation immediately after the native reset."""
        return self.stamp(self.embodiment.reset(scene, seed=seed))

    def step(self, action: Any) -> Any:
        """Capture each measured result without altering action semantics."""
        result = self.embodiment.step(action)
        return dataclasses.replace(result, observation=self.stamp(result.observation))


class RecordingSink(NullSink):
    """Stream executed Inspect transitions into the existing Colosseum recorder."""

    def __init__(
        self,
        path: Path,
        config: Any,
        session: RouterSession,
        cameras: dict[str, str],
        *,
        control_mode: str = "joint_pos",
    ):
        self.path, self.config, self.session, self.cameras = path, config, session, cameras
        self.control_mode = control_mode
        self.recorder: Any = None
        self.log: Any = None
        self.final_observation: Any = None
        self.frame_count = 0

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Start exactly one durable recording for this Router run."""
        from colosseum_client.recording import TrialRecorder

        if self.recorder is not None or epoch != 0:
            raise ValueError("A Colosseum run must contain exactly one Inspect trial")
        if self.config.recording:
            self.recorder = TrialRecorder(self.path, list(self.cameras))

    def observation(self, native: Any) -> Any:
        """Split measured YAM state; never manufacture missing joint or gripper data."""
        state = np.asarray(native.state.get("joint_pos"), dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Recording requires measured 14-D YAM joint_pos state")
        if np.any(state[[6, 13]] < 0) or np.any(state[[6, 13]] > 1):
            raise ValueError("YAM gripper state must be normalized 0 closed, 1 open")
        images = {name: native.images[source] for name, source in self.cameras.items()}
        return SimpleNamespace(
            images=images,
            joints=state[[*range(6), *range(7, 13)]],
            gripper=state[[6, 13]],
            cartesian_position=np.empty(0),
        )

    def log_step(self, t: int, observation: Any, action: Any, result: Any) -> None:
        """Record post-guardrail commands with the actual pre-action observation."""
        if t != self.frame_count:
            raise ValueError("Inspect control steps are not contiguous")
        recorded_action = action.data
        if self.control_mode == "eef_abs_pose":
            recorded_action = np.asarray(result.info.get(_JOINT_COMMAND_INFO_KEY), dtype=np.float64)
            if recorded_action.shape != (14,) or not np.isfinite(recorded_action).all():
                raise ValueError("EEF step has no valid post-IK joint command; cannot record it")
            if np.any(recorded_action[[6, 13]] < 0) or np.any(recorded_action[[6, 13]] > 1):
                raise ValueError("Post-IK joint command grippers must be normalized in [0, 1]")
        if self.recorder is not None:
            self.recorder.add(
                self.observation(observation),
                recorded_action,
                captured_at=observation.extra["_colosseum_received_at"],
            )
        self.final_observation = result.observation
        self.session.event("step", "step", step=t)
        self.frame_count += 1

    def on_trial_end(self, record: Any) -> None:
        """Preserve the final observation and mark failed recordings incomplete."""
        from colosseum_client.evaluation import save

        if self.recorder is not None:
            if record.status == "success" and self.final_observation is not None:
                self.recorder.add(
                    self.observation(self.final_observation),
                    captured_at=self.final_observation.extra["_colosseum_received_at"],
                )
            self.recorder.close()
            if record.status != "success":
                save(
                    self.path / "recording-status.json",
                    {"complete": False, "frames": self.recorder.frames},
                )

    def on_eval_end(self, log: Any) -> None:
        """Remember the canonical Inspect log for export provenance and resume."""
        from colosseum_client.evaluation import save

        self.log = log
        save(self.path / "inspect-result.json", log.to_dict())

    def close(self) -> None:
        """Drain the writer on every exceptional exit without claiming completion."""
        if self.recorder is not None:
            self.recorder.close()
            if self.log is None or self.log.status != "success":
                from colosseum_client.evaluation import save

                save(
                    self.path / "recording-status.json",
                    {"complete": False, "frames": self.recorder.frames},
                )


def native_args(
    options: argparse.Namespace, task: dict[str, Any], path: Path, *, control_hz: int = 30
) -> Any:
    """Use the standard run parser while binding task, identity and horizon to Router."""
    from inspect_robots.cli import _check_shared_run_conflicts, _parse_kvs, build_parser

    tail = list(options.inspect_args)
    if tail[:1] == ["--"]:
        tail.pop(0)
    args = build_parser().parse_args(["run", *tail])
    if args.task or args.auto_task or args.sim:
        raise ValueError(
            "Bridge uses the Router task; --task, --auto-task and --sim are unsupported"
        )
    if args.policy not in {None, "agent"}:
        raise ValueError("Native bridge currently supports --policy agent only")
    if args.instruction is not None and args.instruction != task["instruction"]:
        raise ValueError("Native instruction must match the Router assignment")
    if args.epochs not in {None, 1}:
        raise ValueError("Each Colosseum run must use --epochs 1")
    if args.max_steps is not None and args.max_steps != task["max_steps"]:
        raise ValueError("Native --max-steps must match the Router task horizon")
    args.instruction, args.max_steps, args.epochs = task["instruction"], task["max_steps"], 1
    args.policy = "agent"
    args.embodiment = args.embodiment or "yam_arms"
    args.scorer = args.scorer or "episode_length"
    args.grader = args.grader or "none"  # Colosseum collects the actual human scores.
    args.log_dir = str(path / "inspect")
    if args.store_frames is None:
        args.store_frames = True
    if args.rerun is None:
        args.rerun = False
    if args.rerun_save is None:
        args.rerun_save = False
    if args.max_action_delta is None:
        args.max_action_delta = 0.05
    if args.fail_on_error is None:
        args.fail_on_error = 1
    eef = args.embodiment == "yam_eef"
    from inspect_robots.defaults import load_defaults

    env = dict(os.environ)
    if args.config:
        env["INSPECT_ROBOTS_CONFIG"] = str(Path(args.config).expanduser().resolve())
    configured = load_defaults(env)
    if configured.policy_args_owner == "agent" and "wire" in configured.policy_args:
        args.policy_args = [f"wire={configured.policy_args['wire']}", *(args.policy_args or [])]
    inherited: dict[str, Any] = {}
    if eef and configured.embodiment_args_owner in {"yam_arms", "yam_eef"}:
        inherited = configured.embodiment_args
    defaults = {
        **inherited,
        "auto_start": False,
        "unattended": False,
        "control_hz": control_hz,
        "control_interface": "eef_pos" if eef else "joints",
        "joints_are_delta": False,
    }
    overrides = _parse_kvs(args.embodiment_args)
    interface = "eef_pos" if eef else "joints"
    if overrides.get("control_interface", interface) != interface:
        raise ValueError(f"{args.embodiment} requires control_interface={interface}")
    if overrides.get("joints_are_delta", False):
        raise ValueError("Router catalog requires absolute joint commands")
    if overrides.get("control_hz", control_hz) != control_hz:
        raise ValueError("Native control_hz must match Client control_hz")
    args.embodiment_args = [f"{k}={v}" for k, v in {**defaults, **overrides}.items()]
    _check_shared_run_conflicts(args)
    return args


def execute_trial(
    config: Any,
    assignment: dict[str, Any],
    trial: dict[str, Any],
    path: Path,
    options: argparse.Namespace,
    catalog: list[dict[str, Any]],
    *,
    session_factory: Any = None,
) -> None:
    """Run the real CLI loop, then finish only after successful hardware cleanup."""
    from colosseum_client.evaluation import save

    from inspect_robots.cli import _cmd_run, _parse_kvs

    hz = client_control_hz(config)
    args = native_args(options, assignment["task"], path, control_hz=hz)
    control_mode = "eef_abs_pose" if args.embodiment == "yam_eef" else "joint_pos"
    if args.no_prompt and not config.test:
        raise ValueError("Physical bridge evaluations require native operator prompts")
    if args.config:
        os.environ["INSPECT_ROBOTS_CONFIG"] = str(Path(args.config).expanduser().resolve())
    local_selection = session_factory is not None
    session = (session_factory or RouterSession)(config, assignment, trial)
    sink: RecordingSink | None = None
    agent: Any = None
    try:
        identity = session.prepare()
        descriptors = path.parent / (
            "selected-models.json" if local_selection else "router-models.json"
        )
        assigned = json.loads(descriptors.read_text()) if descriptors.exists() else {}
        recorded = public_model(identity)
        parsed_url = urlsplit(recorded.get("url", ""))
        recorded["url"] = urlunsplit(
            (parsed_url.scheme, parsed_url.hostname or "", parsed_url.path, "", "")
        )
        assigned[trial["id"]] = {"side": trial["side"], "model": recorded}
        save(descriptors, assigned)
        validate_assigned_llm(identity)
        matches = [model for model in catalog if public_model(model) == public_model(identity)]
        if len(matches) != 1:
            if identity.get("name", "").startswith("claude-") and config.llm_account_logins:
                raise ValueError(
                    "Router assigned a Claude model that does not match the account-login "
                    "catalog. Register/enable the exact official Anthropic URL, model name "
                    "and revision on Router; Claude Code login cannot replace a relay key."
                )
            raise ValueError("Router model has no unique matching entry in runtime config")
        model = matches[0]
        agent = make_agent(model, config, _parse_kvs(args.policy_args))
        cameras = {name: _CAMERAS[name] for name in config.cameras}
        if set(assignment["task"]["cameras"]) - set(cameras):
            raise ValueError("Client recording cameras do not cover the Router task")
        save(path / "policy.json", {"run_id": trial["id"], "model": identity})
        save(
            path / "recording-context.json",
            {
                "robot_type": "yam",
                "fps": hz,
                "test": config.test,
                "action_space": "joint_position",
                "execution_enabled": True,
                "camera_map": cameras,
                "source": "inspect-robots CLI",
                "model_selection": "local" if local_selection else "router",
                "native_llm_auth": (
                    "claude-code"
                    if identity["name"].startswith("claude-")
                    and config.llm_account_logins.get("anthropic")
                    else "api-key"
                ),
                "native_embodiment": args.embodiment,
                "native_control_mode": control_mode,
                "capture_timestamp_semantics": (
                    "monotonic native observation receipt before inference"
                ),
                "action_recording": (
                    "sent normalized joint commands after EEF IK and joint clipping"
                    if control_mode == "eef_abs_pose"
                    else "executed post-guardrail commands"
                ),
                "joint_command_info_key": (
                    _JOINT_COMMAND_INFO_KEY if control_mode == "eef_abs_pose" else None
                ),
            },
        )
        sink = RecordingSink(path, config, session, cameras, control_mode=control_mode)
        policy = ReportingPolicy(agent, session, config, cameras, control_mode=control_mode)
        code = _cmd_run(
            args,
            policy_override=policy,
            extra_sinks=[sink],
            embodiment_wrapper=CaptureClockEmbodiment,
        )
        if code != 0 or sink.log is None or sink.log.status != "success" or sink.frame_count == 0:
            raise RuntimeError(
                "Inspect trial failed or had no executed steps; result was not submitted"
            )
        # Preserve the settings actually used, rather than the current config at upload time.
        context_path = path / "recording-context.json"
        context = json.loads(context_path.read_text())
        context["native_execution"] = {
            "eval": sink.log.to_dict()["eval"],
            "embodiment_args": {
                key: value
                for key, value in _parse_kvs(args.embodiment_args).items()
                if key
                in {
                    "eef_orientation",
                    "control_hz",
                    "control_interface",
                    "joints_are_delta",
                    "auto_start",
                    "unattended",
                }
            },
            "action_space": {
                "semantics": (
                    dataclasses.asdict(agent.info.action_space.semantics)
                    if agent.info.action_space.semantics is not None
                    else None
                ),
                "low": np.asarray(agent.info.action_space.low).tolist(),
                "high": np.asarray(agent.info.action_space.high).tolist(),
            },
            "guardrails": {
                "disabled": bool(args.disable_guardrails),
                "max_action_delta": args.max_action_delta,
            },
        }
        save(context_path, context)
        save(path / "bridge-trial.json", {"native_complete": True, "router_finished": False})
        session.finish()
        save(
            path / "bridge-trial.json",
            {
                "native_complete": True,
                "router_finished": not local_selection,
                "local_finished": local_selection,
            },
        )
    finally:
        try:
            if sink is not None:
                sink.close()
        finally:
            try:
                session.close()
            finally:
                client = getattr(agent, "_client", None)
                if client is not None:
                    client.close()


def validate_assignment(config: Any, assignment: dict[str, Any]) -> None:
    """Validate peer data before paths, hardware or result publication."""
    safe_id(assignment["id"])
    if (
        assignment.get("inference_mode") != "local"
        or assignment.get("robot_id", "yam") != "yam"
        or bool(assignment.get("test", False)) != config.test
        or bool(assignment.get("recording", True)) != config.recording
    ):
        raise ValueError("Assignment mode, robot, test or recording flag differs from config")
    if assignment["track"] not in {"open", "fine-tuning"}:
        raise ValueError("Unknown assignment track")
    task = assignment["task"]
    if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        raise ValueError("Assignment has no instruction")
    if type(task.get("max_steps")) is not int or task["max_steps"] <= 0:
        raise ValueError("Assignment has an invalid step horizon")
    if set(task.get("cameras", [])) - set(config.cameras):
        raise ValueError("Assignment requires cameras missing from client config")
    trials = assignment["runs"]
    expected = {"A", "B"} if assignment["track"] == "open" else {"trial"}
    if len(trials) != len(expected) or {trial["side"] for trial in trials} != expected:
        raise ValueError("Unexpected assignment run sides")
    if len({safe_id(trial["id"]) for trial in trials}) != len(trials):
        raise ValueError("Duplicate run IDs")


def sync_excel(manifest: Path, local: dict[str, Any]) -> bool:
    """Keep Excel current while retaining failed exports for a local-only retry."""
    from colosseum_client.evaluation import save

    target = local.get("excel_path")
    if not target:
        return True
    try:
        from inspect_robots.colosseum_excel import update_workbook

        update_workbook(target, [manifest])
    except Exception as exc:
        if not local.get("excel_pending"):
            print(
                f"Excel update deferred: {exc}. Scores remain saved; "
                "retry with --export-excel-only.",
                file=sys.stderr,
            )
        local["excel_pending"] = True
        local["excel_error"] = type(exc).__name__
        save(manifest, local)
        return False
    local["excel_pending"] = False
    local.pop("excel_error", None)
    save(manifest, local)
    return True


def export_excel_only(config: Any, options: argparse.Namespace) -> int:
    """Refresh one or all saved evaluations without cloud calls or a robot rerun."""
    from colosseum_client.evaluation import save

    from inspect_robots.colosseum_excel import validate_target

    if options.abort or options.dry_run:
        raise ValueError("--export-excel-only cannot combine with --abort or --dry-run")
    root = Path(config.evaluation_dir).resolve()
    manifests = (
        [root / safe_id(options.resume) / "manifest.json"]
        if options.resume
        else sorted(root.glob("*/manifest.json"))
    )
    target = str(validate_target(options.excel)) if options.excel else None
    if not manifests:
        raise ValueError("No saved Colosseum evaluations found for Excel export")
    success = True
    count = 0
    for manifest in manifests:
        local = json.loads(manifest.read_text())
        if target:
            local["excel_path"] = target
            save(manifest, local)
        if not local.get("excel_path"):
            raise ValueError("Use --excel FILE.xlsx to select the output workbook")
        if not local.get("outcomes"):
            continue
        success = sync_excel(manifest, local) and success
        count += 1
    if success:
        print(f"Excel updated from {count} saved evaluation(s); no Router or hardware contacted.")
    return 0 if success else 1


def process_assignment(
    config: Any, api: Any, assignment: dict[str, Any], options: Any, catalog: list[dict[str, Any]]
) -> None:
    """Persist scores and exports before submission; retries never rerun finished trials."""
    from colosseum_client.evaluation import progress, save, yes_no
    from colosseum_client.lerobot_export import export_trial

    validate_assignment(config, assignment)
    folder = Path(config.evaluation_dir).resolve() / assignment["id"]
    manifest = folder / "manifest.json"
    local: dict[str, Any] = (
        json.loads(manifest.read_text())
        if manifest.exists()
        else {
            "bridge": "inspect-robots-v1",
            "assignment": assignment,
            "outcomes": {},
            "uploaded": [],
        }
    )
    if local.get("bridge") != "inspect-robots-v1":
        raise ValueError("Existing evaluation was not created by this bridge")
    if local.get("record_only") and not local.get("result") and not options.record_only:
        raise ValueError("This recording awaits scoring in --review")
    if options.submit_only and (
        not local.get("result")
        or any(
            run["state"] != "finished" or run["side"] not in local.get("outcomes", {})
            for run in assignment["runs"]
        )
    ):
        raise ValueError("--submit-only requires all runs finished and scores saved locally")
    if options.defer_submit:
        local["submission_deferred"] = True
    if options.record_only:
        if not config.recording:
            raise ValueError("--record-only requires recording: true")
        if local.get("submitted") or assignment["state"] == "completed":
            raise ValueError("Cannot record an already submitted evaluation")
        local.update(record_only=True, review="pending", submission_deferred=True)
    if options.excel:
        from inspect_robots.colosseum_excel import validate_target

        local["excel_path"] = str(validate_target(options.excel))
    if local.get("excel_path"):
        from colosseum_client.evaluation import api_url

        local["excel_router"] = api_url(config)
    previous = local["assignment"]
    if (
        previous["id"] != assignment["id"]
        or previous["task"] != assignment["task"]
        or previous["track"] != assignment["track"]
        or [(r["id"], r["side"]) for r in previous["runs"]]
        != [(r["id"], r["side"]) for r in assignment["runs"]]
    ):
        raise ValueError("Saved assignment task or run identity differs from Router")
    save(manifest, local)
    task, track = assignment["task"], assignment["track"]
    completed = assignment["state"] == "completed"
    if assignment["state"] not in {"pending", "completed"}:
        raise ValueError("Assignment is closed; no trial will be executed")
    for trial in assignment["runs"]:
        path = folder / trial["id"]
        marker = path / "bridge-trial.json"
        if trial["state"] == "assigned" and not completed:
            if path.exists():
                raise RuntimeError(
                    "Existing trial artifacts found; refusing to rerun or overwrite them"
                )
            if not catalog:
                raise ValueError("--runtime-config is required for unfinished native trials")
            print(f"\n{trial['side']}: restore the initial scene before the native Inspect prompt.")
            execute_trial(config, assignment, trial, path, options, catalog)
        elif trial["state"] != "finished":
            raise RuntimeError(
                "Trial is interrupted or still running; resume cannot execute it again"
            )
        if not marker.exists() or not json.loads(marker.read_text()).get("native_complete"):
            raise ValueError("Missing completed native trial provenance")
        # A lost finish acknowledgment is recoverable only if Router already says finished.
        save(marker, {"native_complete": True, "router_finished": True})
        trial["state"] = "finished"
        local["assignment"] = assignment
        context = json.loads((path / "recording-context.json").read_text())
        if context.get("native_execution"):
            local.setdefault("execution", {})[trial["id"]] = context["native_execution"]
        save(manifest, local)
        if options.record_only:
            continue
        side = trial["side"]
        if side not in local["outcomes"]:
            if completed:
                raise ValueError("Completed evaluation is missing its saved score")
            partial = progress(side)
            success = partial == 1.0 if track == "open" else yes_no(f"{side} succeeded?")
            local["outcomes"][side] = {"success": success, "partial_success": partial}
            save(manifest, local)
            sync_excel(manifest, local)
        if config.recording:
            context = json.loads((path / "recording-context.json").read_text())
            from inspect_robots.colosseum_evidence import refresh_outcome

            refresh_outcome(path, local["outcomes"][side])
            export_trial(
                path,
                list(context["camera_map"]),
                fps=context["fps"],
                robot_type="yam",
                task=task["instruction"],
                metadata={
                    "assignment_id": assignment["id"],
                    "run_id": trial["id"],
                    "side": side,
                    "track": track,
                    "test": config.test,
                    "institution": assignment.get("institution", config.institution),
                    "scene": assignment.get("scene", config.scene),
                    "evaluator": assignment.get("evaluator", config.evaluator),
                    "outcome": local["outcomes"][side],
                    "action_space": "joint_position",
                    "execution_enabled": True,
                    "source": "inspect-robots CLI",
                    "recording_context": context,
                },
            )
    if options.record_only:
        print("Runs recorded locally; no scores, datasets or final result submitted.")
        print(f"Review: inspect-robots colosseum --client-config {options.client_config} --review")
        return
    if "result" not in local:
        preference = None
        if track == "open":
            while preference not in {"A", "B", "tie"}:
                raw = input("Preference (A/B/tie): ").strip()
                preference = "tie" if raw.lower() == "tie" else raw.upper()
        local["result"] = {
            "outcomes": local["outcomes"],
            "preference": preference,
            "feedback": input("Feedback (optional): ").strip() if track == "open" else "",
        }
        save(manifest, local)
    sync_excel(manifest, local)
    if local.get("submission_deferred") and not options.submit_only:
        print(
            f"Evaluation {assignment['id']} saved locally for review; "
            "final result and datasets have not been submitted by this command."
        )
        print(f"After review, use --submit-only --resume {assignment['id']} to publish.")
        return
    if not completed:
        api.request("POST", f"/assignments/{assignment['id']}/result", local["result"])
    local["submitted"] = True
    local["submission_deferred"] = False
    save(manifest, local)
    sync_excel(manifest, local)
    print(f"Evaluation {assignment['id']} submitted.")
    if config.recording and not config.skip_upload:
        for trial in assignment["runs"]:
            api.upload_dataset(trial["id"], folder / trial["id"] / "lerobot")
            local.setdefault("datasets_uploaded", {})[trial["id"]] = True
            save(manifest, local)
            sync_excel(manifest, local)
        print("LeRobot datasets uploaded and verified.")
    elif config.recording:
        print("Dataset upload deferred; saved locally. Scores have been submitted.")
    if local.get("excel_path") and not local.get("excel_pending"):
        print(f"Excel updated: {local['excel_path']} (Colosseum Evals)")


def run(options: argparse.Namespace) -> int:
    """Dispatch one assignment, local validation, abort, or durable resume."""
    from inspect_robots.cli import _parse_kvs

    api: Any = None
    assignment: dict[str, Any] | None = None
    try:
        load_dependencies()
        from colosseum_client.evaluation import EvalAPI, select_task
        from colosseum_client.robot_config import RobotClientConfig

        if options.review:
            if (
                options.resume
                or options.abort
                or options.submit_only
                or options.record_only
                or options.dry_run
                or options.export_excel_only
                or options.inspect_args
            ):
                raise ValueError(
                    "--review cannot combine with recording, submission or run options"
                )
            from inspect_robots.colosseum_review import serve

            return serve(
                options.client_config, options.review_port, open_browser=options.review_open
            )
        if options.record_only and (
            options.submit_only or options.abort or options.export_excel_only
        ):
            raise ValueError("--record-only cannot combine with submission, abort or Excel export")
        if options.submit_only and (
            not options.resume
            or options.defer_submit
            or options.abort
            or options.dry_run
            or options.export_excel_only
        ):
            raise ValueError(
                "--submit-only requires --resume and cannot combine with "
                "--defer-submit, --abort, --dry-run or --export-excel-only"
            )
        if options.defer_submit and (options.abort or options.export_excel_only):
            raise ValueError("--defer-submit cannot combine with --abort or --export-excel-only")
        config = RobotClientConfig.from_yaml(options.client_config)
        config = dataclasses.replace(
            config,
            **{
                name: getattr(options, name)
                for name in ("instruction", "scene", "evaluator", "track")
                if getattr(options, name) is not None
            },
            skip_upload=config.skip_upload or options.skip_upload,
        )
        if options.record_only and not config.recording:
            raise ValueError("--record-only requires recording: true")
        if options.export_excel_only:
            return export_excel_only(config, options)
        local_id = options.resume or options.abort
        local_manifest = (
            Path(config.evaluation_dir) / safe_id(local_id) / "manifest.json" if local_id else None
        )
        saved_local = (
            local_manifest is not None
            and local_manifest.is_file()
            and json.loads(local_manifest.read_text()).get("bridge") == "inspect-robots-local-v1"
        )
        if options.local_pair or saved_local:
            from inspect_robots.colosseum_local import run_local_pair

            return run_local_pair(config, options)
        if options.model_a or options.model_b:
            raise ValueError("--model-a/--model-b require --local-pair")
        if options.excel:
            from inspect_robots.colosseum_excel import validate_target

            validate_target(options.excel)
        if config.robot_type != "yam" or set(config.cameras) - set(_CAMERAS):
            raise ValueError("Native bridge requires robot_type: yam and standard camera names")
        hz = client_control_hz(config)
        catalog = (
            read_catalog(options.runtime_config, control_hz=hz) if options.runtime_config else []
        )
        if catalog:
            validate_catalog_keys(config, catalog)
        if options.abort:
            if (
                options.resume
                or options.dry_run
                or not options.reason
                or not options.reason.strip()
            ):
                raise ValueError(
                    "--abort requires --reason and cannot combine with --resume/--dry-run"
                )
            api = EvalAPI(config)
            api.request(
                "POST", f"/assignments/{safe_id(options.abort)}/abort", {"reason": options.reason}
            )
            manifest = Path(config.evaluation_dir).resolve() / options.abort / "manifest.json"
            if manifest.exists():
                from colosseum_client.evaluation import save

                saved = json.loads(manifest.read_text())
                saved["aborted"] = True
                save(manifest, saved)
            return 0
        if not options.resume and not catalog:
            raise ValueError("--runtime-config is required for a new evaluation")
        if options.resume:
            safe_id(options.resume)
        root = Path(config.evaluation_dir).resolve()
        # Validate native flags before requesting an assignment or opening a robot.
        preview = native_args(
            options,
            {
                "instruction": config.instruction or "validation only",
                "max_steps": config.max_trial_steps,
            },
            root / "preview",
            control_hz=hz,
        )
        invalid = set(_parse_kvs(preview.policy_args)) - _AGENT_OPTIONS
        if invalid:
            raise ValueError(f"Unsupported native -P options: {', '.join(sorted(invalid))}")
        speed = float(_parse_kvs(preview.policy_args).get("max_speed_frac", 0.05))
        if not math.isfinite(speed) or not 0 < speed <= 1:
            raise ValueError("Native speed must be finite and in (0, 1]")
        if options.dry_run:
            print(
                "Local bridge configuration validated; "
                "no Router, API inference or hardware contacted."
            )
            print("Models: " + ", ".join(model["name"] for model in catalog))
            print(
                "LLM authentication: "
                + ", ".join(
                    f"{provider}={config.llm_account_logins.get(provider, 'api-key')}"
                    for provider in config.llm_providers
                )
            )
            print(f"Local data: {root}")
            control = (
                "EEF → IK → joint execution"
                if preview.embodiment == "yam_eef"
                else "absolute joints"
            )
            print(f"Native control: YAM {control}, {hz} Hz; cameras: " + str(_CAMERAS))
            return 0
        if not options.resume:
            for path in root.glob("*/manifest.json"):
                saved = json.loads(path.read_text())
                if (
                    saved.get("bridge") == "inspect-robots-v1"
                    and not saved.get("submitted")
                    and not saved.get("aborted")
                ):
                    raise ValueError(f"Resume or abort pending evaluation {path.parent.name} first")
        api = EvalAPI(config)
        if options.resume:
            assignment = api.request("GET", f"/assignments/{options.resume}")
        else:
            track = config.track or "open"
            instruction = config.instruction or (
                input("Instruction: ").strip() if track == "open" else ""
            )
            if track == "open" and not instruction:
                raise ValueError("Instruction cannot be empty")
            scene = config.scene or (input("Scene / setup: ").strip() if track == "open" else "")
            body = {
                "robot_id": "yam",
                "track": track,
                "instruction": instruction,
                "scene": scene,
                "evaluator": config.evaluator,
                "institution": config.institution.strip(),
                "inference_mode": "local",
                "test": config.test,
                "recording": config.recording,
                "max_steps": config.max_trial_steps,
                "cameras": list(config.cameras),
            }
            if track == "fine-tuning":
                body["task_id"] = select_task(api, "yam", config.test)
            assignment = api.request("POST", "/next", body)
        print(
            f"Colosseum evaluation: {assignment['id']}; use --resume {assignment['id']} to retry."
        )
        process_assignment(config, api, assignment, options, catalog)
        manifest = root / safe_id(assignment["id"]) / "manifest.json"
        if json.loads(manifest.read_text()).get("excel_pending"):
            return 1
        return 0
    except KeyboardInterrupt:
        print(
            "Interrupted. Local data retained; no successful trial is inferred from interruption.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"Colosseum bridge: {exc}", file=sys.stderr)
        if options.review:
            print("Review server could not start; saved recordings retained.", file=sys.stderr)
        elif assignment is None and not (
            options.resume or options.abort or options.export_excel_only or options.local_pair
        ):
            print(
                "No evaluation assignment was obtained and no robot was started. "
                "Correct the error and retry the original command.",
                file=sys.stderr,
            )
        else:
            print(
                "Saved results/recordings are retained. "
                "Resume with the same configs and evaluation ID.",
                file=sys.stderr,
            )
        return 1
    finally:
        if api is not None:
            api.close()
