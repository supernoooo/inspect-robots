"""Claude Code login, vision, failures, and eval integration without model/hardware calls."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.task import Task
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._capture import WireCapture
from inspect_robots_agent._claude_code import ClaudeCodeClient
from inspect_robots_agent._png import encode_png, png_data_url
from inspect_robots_agent._tools import build_toolset

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_SCRIPT = r"""
import json, os, sys, time
from pathlib import Path
if sys.argv[-2:] == ['auth', 'status']:
    print(json.dumps({'loggedIn': os.environ.get('FAKE_LOGIN') != 'false',
                      'authMethod': os.environ.get('FAKE_AUTH_METHOD', 'claude.ai')}))
    sys.exit(0)
time.sleep(float(os.environ.get('FAKE_SLEEP', '0')))
payload = json.loads(sys.stdin.read())
trace = Path(os.environ['FAKE_TRACE'])
entry = {'argv': sys.argv[1:], 'payload': payload,
         'api_key': os.environ.get('ANTHROPIC_API_KEY'),
         'auth_token': os.environ.get('ANTHROPIC_AUTH_TOKEN'),
         'openai_key': os.environ.get('OPENAI_API_KEY')}
with trace.open('a') as stream:
    stream.write(json.dumps(entry) + '\n')
replies = json.loads(Path(os.environ['FAKE_REPLIES']).read_text())
index = len(trace.read_text().splitlines()) - 1
reply = replies[min(index, len(replies) - 1)]
print(json.dumps({'type': 'system', 'subtype': 'init'}))
print(os.environ.get('FAKE_STDOUT') or json.dumps(reply))
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
"""


def _fake_cli(tmp_path: Path, replies: list[Any]) -> tuple[Path, dict[str, str], Path]:
    executable = tmp_path / "fake claude"
    executable.write_text(f"#!{sys.executable}\n" + _SCRIPT)
    executable.chmod(0o755)
    response_path = tmp_path / "replies.json"
    response_path.write_text(json.dumps(replies))
    trace = tmp_path / "trace.jsonl"
    environment = {
        **os.environ,
        "FAKE_REPLIES": str(response_path),
        "FAKE_TRACE": str(trace),
        "ANTHROPIC_API_KEY": "anthropic-test-key",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-test-token",
        "OPENAI_API_KEY": "gpt-test-key",
    }
    return executable, environment, trace


def _decision(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "result",
        "is_error": False,
        "structured_output": {
            "content": "The scene is visible.",
            "tool_calls": [{"name": name, "arguments": arguments}],
        },
        "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3},
    }


def _info() -> EmbodimentInfo:
    return EmbodimentInfo(
        name="single-arm-test",
        action_space=Box(
            shape=(6,),
            low=np.array([-180.0] * 5 + [0.0]),
            high=np.array([180.0] * 5 + [100.0]),
            semantics=ActionSemantics("joint_pos", dim_labels=_JOINTS, max_step=(1.0,) * 6),
        ),
        observation_space=ObservationSpace(
            cameras=(CameraSpec(name="front", height=2, width=2, channels=3),),
            state=StateSpec(fields=(StateField(key="joint_pos", shape=(6,)),)),
        ),
        control_hz=10.0,
        is_simulated=True,
    )


def _tools() -> list[dict[str, Any]]:
    info = _info()
    return build_toolset(info.action_space, info.observation_space, info.control_hz, 0.05).schemas()


def test_cli_receives_camera_history_and_keeps_gpt_key(tmp_path: Path) -> None:
    executable, environment, trace = _fake_cli(tmp_path, [_decision("done", {"summary": "done"})])
    capture = WireCapture()
    capture.begin_trial(str(tmp_path / "logs"), "run", "trial")
    client = ClaudeCodeClient("sonnet", command=str(executable), env=environment, capture=capture)
    image = np.full((2, 2, 3), 99, dtype=np.uint8)
    messages = [
        {"role": "system", "content": "Use these robot tools."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "previous"}]},
        {"role": "tool", "content": "Reached target.", "tool_call_id": "previous"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "camera front"},
                {"type": "image_url", "image_url": {"url": png_data_url(image)}},
            ],
        },
    ]
    message = client.complete(messages, _tools(), reasoning_effort="low")
    assert message.tool_calls[0].name == "done"
    assert message.usage == {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3}
    request = json.loads(trace.read_text())
    blocks = request["payload"]["message"]["content"]
    (frame,) = [block for block in blocks if block["type"] == "image"]
    assert base64.b64decode(frame["source"]["data"]) == encode_png(image)
    assert frame["source"]["media_type"] == "image/png"
    text = "\n".join(block["text"] for block in blocks if block["type"] == "text")
    assert "Reached target." in text and "previous" in text and "camera front" in text
    args = request["argv"]
    assert args[args.index("--tools") + 1] == ""
    assert json.loads(args[args.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert args[args.index("--effort") + 1] == "low"
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--bare" not in args and "--dangerously-skip-permissions" not in args
    schema = json.loads(args[args.index("--json-schema") + 1])
    choices = schema["properties"]["tool_calls"]["items"]["anyOf"]
    assert {choice["properties"]["name"]["const"] for choice in choices} == {
        tool["function"]["name"] for tool in _tools()
    }
    assert request["api_key"] is None and request["auth_token"] is None
    assert request["openai_key"] == environment["OPENAI_API_KEY"] == "gpt-test-key"
    assert environment["ANTHROPIC_API_KEY"] == "anthropic-test-key"
    capture.end_trial()
    row = json.loads((tmp_path / "logs/wire/run/trial/calls.jsonl").read_text())
    assert row["status"] == 0 and row["endpoint"] == "claude-code://local/print"
    assert row["response"]["structured_output"]["tool_calls"][0]["name"] == "done"
    assert "image/png;base64,$blob:" in json.dumps(row["request"])
    client.close()


@pytest.mark.parametrize(
    "reply",
    [
        {},
        [],
        {"type": "result", "is_error": True, "result": "Subscription usage limit reached"},
        {"structured_output": {"content": "", "tool_calls": []}},
        {"structured_output": {"content": "", "tool_calls": [{"name": "shell", "arguments": {}}]}},
        {"structured_output": {"content": "", "tool_calls": [{"name": "done", "arguments": []}]}},
    ],
)
def test_bad_cli_result_yields_no_action(tmp_path: Path, reply: Any) -> None:
    executable, environment, _ = _fake_cli(tmp_path, [reply])
    client = ClaudeCodeClient("sonnet", command=str(executable), env=environment)
    with pytest.raises(RuntimeError):
        client.complete([{"role": "user", "content": "observe"}], _tools())


@pytest.mark.parametrize(
    "setting,value", [("FAKE_LOGIN", "false"), ("FAKE_AUTH_METHOD", "api_key")]
)
def test_missing_or_api_login_fails_before_eval(tmp_path: Path, setting: str, value: str) -> None:
    executable, environment, trace = _fake_cli(tmp_path, [{}])
    environment[setting] = value
    with pytest.raises(ConfigError, match="claude auth login"):
        ClaudeCodeClient("sonnet", command=str(executable), env=environment)
    assert not trace.exists()


def test_timeout_and_non_json_output_fail(tmp_path: Path) -> None:
    executable, environment, _ = _fake_cli(tmp_path, [{}])
    environment["FAKE_SLEEP"] = "2"
    client = ClaudeCodeClient("sonnet", command=str(executable), env=environment, timeout_s=0.3)
    with pytest.raises(RuntimeError, match="timed out"):
        client.complete([{"role": "user", "content": "observe"}], _tools())
    environment.pop("FAKE_SLEEP")
    environment["FAKE_STDOUT"] = "Not JSON"
    client = ClaudeCodeClient("sonnet", command=str(executable), env=environment)
    with pytest.raises(RuntimeError, match="parse Claude Code"):
        client.complete([{"role": "user", "content": "observe"}], _tools())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.2},
        {"effort": "xhigh"},
        {"base_url": "https://api.example"},
        {"api_key_env": "OPENAI_API_KEY"},
        {"transport": httpx.MockTransport(lambda _: None)},
    ],
)
def test_cli_rejects_unsupported_api_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        LLMAgentPolicy(wire="claude-code", model="sonnet", env={}, **kwargs)


class _SingleArm:
    def __init__(self) -> None:
        self.info = _info()
        self.q = np.zeros(6)

    def _observation(self) -> Observation:
        return Observation(
            state={"joint_pos": self.q.copy()},
            images={"front": np.full((2, 2, 3), 99, dtype=np.uint8)},
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        return self._observation()

    def step(self, action: Action) -> StepResult:
        self.q = np.asarray(action.data).copy()
        return StepResult(observation=self._observation())

    def close(self) -> None:
        pass


def test_single_arm_eval_retains_actions_transcript_usage_and_hindsight(tmp_path: Path) -> None:
    executable, environment, trace = _fake_cli(
        tmp_path,
        [
            _decision(
                "move_joints",
                {
                    "targets": {"shoulder_pan": 0.5},
                    "note": "The target is close; move shoulder_pan slightly.",
                },
            ),
            _decision(
                "done", {"summary": "Reached target", "hindsight": "Use small joint targets."}
            ),
        ],
    )
    policy = LLMAgentPolicy(
        wire="claude-code",
        model="anthropic/sonnet",
        env=environment,
        claude_command=str(executable),
        max_llm_calls=5,
        max_speed_frac=0.05,
    )
    arm = _SingleArm()
    task = Task(
        name="single-arm",
        scenes=[Scene(id="trial", instruction="Move half a degree")],
        scorer=success_at_end(),
        max_steps=10,
    )
    (log,) = ir_eval(task, policy, arm, log_dir=str(tmp_path / "eval"))
    assert log.status == "success"
    assert log.eval.policy_config["wire"] == "claude-code"
    assert log.eval.policy_config["model"] == "sonnet"
    assert arm.q[0] == pytest.approx(0.5)
    assert np.all(arm.q[1:] == 0.0)
    assert len(trace.read_text().splitlines()) == 2
    serialized = json.dumps(log.to_dict())
    assert "move_joints" in serialized and "hindsight" in serialized and "llm_usage" in serialized
    assert "Use small joint targets." in serialized


def test_openai_api_key_still_drives_responses_wire() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "function_call", "call_id": "gpt", "name": "done", "arguments": "{}"}
                ]
            },
        )

    policy = LLMAgentPolicy(
        model="openai/gpt-test",
        wire="responses",
        env={"OPENAI_API_KEY": "gpt-test-key"},
        transport=httpx.MockTransport(respond),
        wire_capture=False,
    )
    response = policy._client.complete([{"role": "user", "content": "observe"}], _tools())
    assert response.tool_calls[0].name == "done"
    assert requests[0].headers["Authorization"] == "Bearer gpt-test-key"
    assert requests[0].url.host == "api.openai.com"
    policy._client.close()
