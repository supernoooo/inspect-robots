"""Strict, independent client for MolmoAct2's bimanual YAM /act protocol."""

from __future__ import annotations

import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import json_numpy
import numpy as np
import numpy.typing as npt

from inspect_robots import Action, ActionChunk
from inspect_robots_jev.yam_contract import ACTION_DIM, CAMERA_NAMES, DIM_LABELS, action_space

from .contract import DecodedInput

REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")


class MolmoError(Exception):
    """A stable failure code; server response bodies and URLs are never exposed."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, eq=False)
class MolmoResult:
    chunk: ActionChunk
    checkpoint: str
    revision: str
    latency_s: float
    server_latency_ms: float | None


class MolmoActClient:
    """GET identity once, then POST one inference for each accepted observation."""

    def __init__(self, url: str = "http://127.0.0.1:8202", *, timeout_s: float = 120.0,
                 max_response_bytes: int = 8_000_000) -> None:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                parsed.path not in ("", "/")):
            raise ValueError("molmo_url must be an HTTP(S) server origin without credentials")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or max_response_bytes < 1:
            raise ValueError("invalid Molmo transport limits")
        self.url = url.rstrip("/") + "/act"
        self.timeout_s = timeout_s
        self.max_response_bytes = max_response_bytes
        self._identity: tuple[str, str] | None = None

    @property
    def identity(self) -> tuple[str, str] | None:
        return self._identity

    def _request(self, method: str, payload: bytes | None, limit: int) -> object:
        request = urllib.request.Request(
            self.url, data=payload, method=method,
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if response.status != 200:
                    raise MolmoError("http_status")
                raw = response.read(limit + 1)
            if len(raw) > limit:
                raise MolmoError("response_too_large")
        except urllib.error.HTTPError:
            raise MolmoError("http_status") from None
        except (TimeoutError, socket.timeout):
            raise MolmoError("timeout") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise MolmoError("timeout") from None
            raise MolmoError("service_error") from None
        try:
            return json_numpy.loads(raw)
        except (UnicodeError, ValueError, TypeError):
            raise MolmoError("invalid_json") from None

    def _check_identity(self) -> tuple[str, str]:
        if self._identity is not None:
            return self._identity
        value = self._request("GET", None, 16_384)
        if not isinstance(value, dict):
            raise MolmoError("invalid_identity")
        checkpoint = value.get("checkpoint")
        revision = value.get("revision")
        if (value.get("status") != "ok" or value.get("repo_id") != REPO_ID or
                value.get("norm_tag") != NORM_TAG or value.get("num_cameras") != 3 or
                value.get("state_dim") != ACTION_DIM or checkpoint != REPO_ID or
                not isinstance(revision, str) or not _REVISION.fullmatch(revision)):
            raise MolmoError("invalid_identity")
        order = value.get("action_order")
        if order is not None and order != list(DIM_LABELS):
            raise MolmoError("action_order_mismatch")
        self._identity = checkpoint, revision
        return self._identity

    def infer(self, decoded: DecodedInput) -> MolmoResult:
        """Validate the *whole* returned chunk before any prefix may be used."""
        checkpoint, revision = self._check_identity()
        payload = {name: decoded.images[name] for name in CAMERA_NAMES}
        payload.update(instruction=decoded.instruction,
                       state=decoded.joint_pos.astype(np.float32),
                       timestamp=decoded.state_time, num_steps=10)
        started = time.monotonic()
        try:
            body = json_numpy.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            raise MolmoError("invalid_request") from None
        value = self._request("POST", body, self.max_response_bytes)
        elapsed = time.monotonic() - started
        if not isinstance(value, dict) or "actions" not in value:
            raise MolmoError("missing_actions")
        for key, expected in (("repo_id", REPO_ID), ("checkpoint", checkpoint),
                              ("revision", revision), ("norm_tag", NORM_TAG)):
            if key in value and value[key] != expected:
                raise MolmoError("identity_mismatch")
        if "action_order" in value and value["action_order"] != list(DIM_LABELS):
            raise MolmoError("action_order_mismatch")
        try:
            raw_actions = np.asarray(value["actions"])
            if not np.issubdtype(raw_actions.dtype, np.integer) and not np.issubdtype(
                raw_actions.dtype, np.floating
            ):
                raise MolmoError("invalid_shape")
            actions = raw_actions.astype(np.float64)
        except MolmoError:
            raise
        except (TypeError, ValueError, OverflowError):
            raise MolmoError("invalid_shape") from None
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
            raise MolmoError("invalid_shape")
        if not np.isfinite(actions).all():
            raise MolmoError("nonfinite_actions")
        if np.any(actions[:, [6, 13]] < 0) or np.any(actions[:, [6, 13]] > 1):
            raise MolmoError("invalid_gripper")
        space = action_space()
        assert space.low is not None and space.high is not None
        if np.any(actions < space.low) or np.any(actions > space.high):
            raise MolmoError("joint_limit")
        dt_ms = value.get("dt_ms")
        if dt_ms is not None and (type(dt_ms) not in (int, float) or
                                  not math.isfinite(dt_ms) or dt_ms < 0):
            raise MolmoError("invalid_latency")
        chunk = ActionChunk(actions=[Action(data=row.copy()) for row in actions],
                            inference_latency_s=elapsed)
        return MolmoResult(chunk, checkpoint, revision, elapsed,
                           float(dt_ms) if dt_ms is not None else None)
