"""CaP-X over the Claude Code subscription CLI, using a fake executable."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation
from inspect_robots_capx import CapxPolicy, CapxPolicyConfig

# The subscription-login client is optional in inspect-robots-agent builds.
pytest.importorskip("inspect_robots_agent._claude_code")

_SCRIPT = r"""
import json, os, sys
from pathlib import Path
if sys.argv[-2:] == ['auth', 'status']:
    print(json.dumps({'loggedIn': True, 'authMethod': 'claude.ai'}))
    sys.exit(0)
payload = json.loads(sys.stdin.read())
trace = Path(os.environ['FAKE_TRACE'])
with trace.open('a') as stream:
    stream.write(json.dumps({'argv': sys.argv[1:], 'payload': payload,
                             'api_key': os.environ.get('ANTHROPIC_API_KEY')}) + '\n')
print(json.dumps({'type': 'result', 'is_error': False, 'result': os.environ['FAKE_TEXT'],
                  'usage': {'input_tokens': 3, 'output_tokens': 4}}))
"""


def _fake_cli(tmp_path: Path, text: str) -> tuple[Path, dict[str, str], Path]:
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\n" + _SCRIPT)
    executable.chmod(0o755)
    trace = tmp_path / "trace.jsonl"
    env = {
        **os.environ,
        "FAKE_TRACE": str(trace),
        "FAKE_TEXT": text,
        "ANTHROPIC_API_KEY": "must-not-reach-the-cli",
    }
    return executable, env, trace


def _info() -> EmbodimentInfo:
    return EmbodimentInfo(
        name="joint-test",
        action_space=Box(
            shape=(3,),
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            semantics=ActionSemantics(
                "joint_pos", gripper="continuous", dim_labels=("j0", "j1", "gripper")
            ),
        ),
        observation_space=ObservationSpace(
            cameras=(CameraSpec("front", 2, 2),),
            state=StateSpec(fields=(StateField("joint_pos", (3,)),)),
        ),
        control_hz=10.0,
        is_simulated=True,
    )


def test_claude_code_wire_runs_code_turns_through_the_cli(tmp_path: Path) -> None:
    executable, env, trace = _fake_cli(
        tmp_path, "import numpy as np\nmove_to_joints(np.array([0.2, 0.0]))"
    )
    policy = CapxPolicy(
        model="anthropic/claude-opus-5-5",
        wire="claude-code",
        effort="medium",
        claude_command=str(executable),
        claude_timeout_s=30.0,
        env=env,
    )
    try:
        policy.bind(_info())
        policy.reset(Scene(id="s", instruction="pick"))
        chunk = policy.act(
            Observation(
                images={"front": np.zeros((2, 2, 3), dtype=np.uint8)},
                state={"joint_pos": np.array([0.0, 0.0, 1.0])},
            )
        )
    finally:
        policy.close()

    assert np.array_equal(chunk.actions[-1].data, np.array([0.2, 0.0, 1.0]))
    assert isinstance(policy.config, CapxPolicyConfig)
    assert policy.config.model == "claude-opus-5-5"
    assert policy.config.base_url == "claude-code://local"
    assert policy.config.claude_command == str(executable.resolve())
    assert policy.config.claude_timeout_s == 30.0
    call = json.loads(trace.read_text().splitlines()[0])
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--effort") + 1] == "medium"
    assert "--json-schema" not in argv
    assert "You generate Python code" in argv[argv.index("--system-prompt") + 1]
    assert call["api_key"] is None
    assert any(block["type"] == "image" for block in call["payload"]["message"]["content"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_url": "http://x.test/v1"}, "official CLI login"),
        ({"temperature": 0.2}, "temperature"),
        ({"effort": "xhigh"}, "low, medium, high, or max"),
        ({"model": None}, "needs a model"),
    ],
)
def test_claude_code_wire_rejects_api_only_options(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    executable, env, _ = _fake_cli(tmp_path, "FINISH")
    options: dict[str, Any] = {
        "model": "anthropic/claude-opus-5-5",
        "claude_command": str(executable),
        **kwargs,
    }
    with pytest.raises(ConfigError, match=message):
        CapxPolicy(
            wire="claude-code",
            env={k: v for k, v in env.items() if k != "INSPECT_ROBOTS_MODEL"},
            **options,
        )


def test_claude_options_require_the_claude_code_wire() -> None:
    with pytest.raises(ConfigError, match="require -P wire=claude-code"):
        CapxPolicy(model="m", base_url="http://x.test/v1", claude_timeout_s=60.0, env={})
