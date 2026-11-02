"""Call the unmodified Claude Code CLI using the operator's own login.

Only observations and proposed tool calls cross this boundary. Claude Code's
built-in tools and MCP servers are disabled; the existing agent executes the
returned proposal through its normal toolset and rollout approvers. Login and
token refresh stay inside the official binary; credential files are never read.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from inspect_robots.errors import ConfigError

from ._capture import WireCapture
from ._llm import AssistantMessage, ToolCall

# These select API billing or another provider ahead of a subscription login.
_API_AUTH_ENV = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    }
)
_IMAGE_URL = re.compile(r"data:(image/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/=]+)")
_ENDPOINT = "claude-code://local/print"
_SETTINGS = json.dumps({"disableAllHooks": True, "forceLoginMethod": "claudeai"})


class ClaudeCodeClient:
    """Blocking CLI adapter with a finite timeout and no direct actuation."""

    def __init__(
        self,
        model: str,
        *,
        command: str | None = None,
        timeout_s: float = 120.0,
        env: dict[str, str] | None = None,
        capture: WireCapture | None = None,
    ) -> None:
        if command is not None and (not isinstance(command, str) or not command):
            raise ConfigError("claude_command must be a non-empty executable path or None")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ConfigError("claude_timeout_s must be finite and > 0")
        requested = command or "claude"
        executable = shutil.which(requested)
        if executable is None:
            raise ConfigError(
                "Claude Code CLI not found.\n"
                "fix: install the official CLI, run claude auth login, and add it to PATH; "
                "or pass -P claude_command=/absolute/path/to/claude"
            )
        self._command = str(Path(executable).resolve())
        self._model = model
        self._timeout_s = timeout_s
        inherited = dict(os.environ) if env is None else dict(env)
        self._env = {key: value for key, value in inherited.items() if key not in _API_AUTH_ENV}
        self._capture = capture
        self._validate_login()

    @property
    def command(self) -> str:
        """Return the executable recorded with the evaluation configuration."""
        return self._command

    def _validate_login(self) -> None:
        # Ask the official binary for status; never inspect credential files.
        try:
            with TemporaryDirectory(prefix="inspect-robots-claude-auth-") as directory:
                result = subprocess.run(
                    [self._command, "--setting-sources", "", "auth", "status"],
                    cwd=directory,
                    env=self._env,
                    capture_output=True,
                    text=True,
                    timeout=min(self._timeout_s, 30.0),
                    check=False,
                )
            status = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise ConfigError(
                "Could not check Claude Code login.\n"
                "fix: update the official CLI, then run claude auth login and claude auth status"
            ) from exc
        if (
            not isinstance(status, dict)
            or result.returncode != 0
            or status.get("loggedIn") is not True
        ):
            raise ConfigError(
                "Claude Code is not logged in.\n"
                "fix: run claude auth login with your own Claude subscription account"
            )
        if status.get("authMethod") not in {"claude.ai", "oauth_token"}:
            raise ConfigError(
                "Claude Code reports API/Console authentication instead of a subscription.\n"
                "fix: run claude auth login and choose your Claude account (without --console)"
            )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """Return one proposed Inspect Robots tool call, including usage."""
        if temperature is not None:
            raise ConfigError("temperature is not supported by the Claude Code CLI")
        system, content = _prompt(messages)
        schema = _decision_schema(tools)
        system += (
            "\n\nReturn one next robot-tool proposal using the supplied JSON schema. "
            "The tool descriptions are part of this contract; they are not CLI tools. "
            "Inspect Robots validates and executes your proposal and provides its result "
            "with the next observation. Do not execute tools yourself."
        )
        payload = (
            json.dumps(
                {
                    "type": "user",
                    "session_id": "",
                    "parent_tool_use_id": None,
                    "message": {"role": "user", "content": content},
                }
            )
            + "\n"
        )
        if len(payload.encode()) > 10 * 1024 * 1024:
            raise RuntimeError("Claude Code input exceeds 10 MB; use -P image_horizon=1")
        command = [
            self._command,
            "-p",
            "--model",
            self._model,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--json-schema",
            json.dumps(schema),
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--settings",
            _SETTINGS,
            "--disable-slash-commands",
            "--no-session-persistence",
            "--max-turns",
            "3",
        ]
        if reasoning_effort is not None:
            command.extend(["--effort", str(reasoning_effort)])
        request = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "reasoning_effort": reasoning_effort,
            "output_schema": schema,
        }
        start = time.time()
        result: subprocess.CompletedProcess[str] | None = None
        response: dict[str, Any] | None = None
        error: str | None = None
        try:
            # A fresh CLI session receives canonical agent history, so image eviction
            # and per-trial reset retain the same meaning as on the API backends.
            with TemporaryDirectory(prefix="inspect-robots-claude-call-") as directory:
                result = subprocess.run(
                    command,
                    input=payload,
                    cwd=directory,
                    env=self._env,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                    check=False,
                )
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if isinstance(event, dict) and event.get("type") == "result":
                    response = event
            if response is None:
                raise RuntimeError("Claude Code returned no result event: " + result.stderr[:1000])
            if result.returncode != 0 or response.get("is_error"):
                detail = response.get("result") or response.get("subtype") or result.stderr
                raise RuntimeError(f"Claude Code failed: {str(detail)[:1000]}")
            return _parse_decision(response, tools)
        except subprocess.TimeoutExpired as exc:
            error = f"Claude Code timed out after {self._timeout_s:g} seconds"
            raise RuntimeError(error) from exc
        except (OSError, ValueError) as exc:
            detail = result.stderr[:500] if result is not None else str(exc)
            error = f"Could not run/parse Claude Code: {detail}"
            raise RuntimeError(error) from exc
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            raise
        finally:
            if self._capture is not None:
                self._capture.record(
                    attempt=0,
                    endpoint=_ENDPOINT,
                    request=request,
                    status=result.returncode if result is not None else None,
                    response_text=(
                        json.dumps(response)
                        if response is not None
                        else result.stdout
                        if result is not None
                        else None
                    ),
                    error=error,
                    t_start=start,
                    duration_s=time.time() - start,
                )

    def close(self) -> None:
        """Release resources; each completed call already closes its child process."""


def _prompt(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Translate labeled conversation history and inline camera frames."""
    system: list[str] = []
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Conversation history in order, followed by the current state:",
        }
    ]
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role == "system":
            if not isinstance(content, str):
                raise RuntimeError("Claude Code requires a text system prompt")
            system.append(content)
            continue
        blocks.append({"type": "text", "text": f"\n[{role}]"})
        parts: list[dict[str, Any]] = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        )
        for part in parts:
            if part["type"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                match = _IMAGE_URL.fullmatch(part["image_url"]["url"])
                if match is None:
                    raise RuntimeError(
                        "Claude Code requires an inline PNG/JPEG/GIF/WebP camera image"
                    )
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": match[1],
                            "data": match[2],
                        },
                    }
                )
            else:
                raise RuntimeError(f"Unsupported Claude Code content part: {part['type']}")
        if message.get("tool_calls"):
            blocks.append(
                {"type": "text", "text": json.dumps({"tool_calls": message["tool_calls"]})}
            )
        if message.get("tool_call_id"):
            blocks.append({"type": "text", "text": f"Tool result for {message['tool_call_id']}"})
    return "\n\n".join(system), blocks


def _decision_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Constrain a proposal to the embodiment's existing named robot tools."""
    choices = []
    for tool in tools:
        function = tool["function"]
        choices.append(
            {
                "type": "object",
                "additionalProperties": False,
                "description": function.get("description", ""),
                "properties": {
                    "name": {"const": function["name"], "type": "string"},
                    "arguments": function["parameters"],
                },
                "required": ["name", "arguments"],
            }
        )
    if not choices:
        raise RuntimeError("Claude Code needs a bound robot toolset")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {"anyOf": choices},
            },
        },
        "required": ["content", "tool_calls"],
    }


def _parse_decision(response: dict[str, Any], tools: list[dict[str, Any]]) -> AssistantMessage:
    decision = response.get("structured_output")
    if not isinstance(decision, dict):
        raise RuntimeError("Claude Code returned no structured_output; update the CLI and retry")
    content = decision.get("content")
    calls = decision.get("tool_calls")
    if not isinstance(content, str) or not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError("Claude Code must return text and exactly one robot-tool proposal")
    call = calls[0]
    allowed = {tool["function"]["name"] for tool in tools}
    if not isinstance(call, dict) or call.get("name") not in allowed:
        raise RuntimeError("Claude Code returned an unknown robot tool")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        raise RuntimeError("Claude Code robot-tool arguments must be a JSON object")
    usage = response.get("usage") or {}
    counts = (
        {
            key: value
            for key, value in usage.items()
            if key
            in {
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            }
            and isinstance(value, int)
            and not isinstance(value, bool)
        }
        if isinstance(usage, dict)
        else {}
    )
    return AssistantMessage(
        content=content or None,
        tool_calls=(
            ToolCall(
                id=f"claude_{uuid.uuid4().hex}", name=call["name"], arguments=json.dumps(arguments)
            ),
        ),
        usage=counts or None,
    )
