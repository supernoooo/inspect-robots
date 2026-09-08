"""HTTP policy adapter for MolmoAct2-LIBERO's generic ``/act`` server."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from inspect_robots import Action, ActionChunk, Observation, PolicyConfig, PolicyInfo, Scene
from inspect_robots_libero.contract import (
    ACTION_DIM,
    CAMERA_NAMES,
    STATE_DIM,
    STATE_KEY,
    action_space,
    observation_space,
)

PostFn = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def _default_post(  # pragma: no cover - exercised only against a live model server
    url: str, payload: Mapping[str, Any], timeout_s: float
) -> Mapping[str, Any]:
    import json_numpy
    import requests

    response = requests.post(
        url,
        data=json_numpy.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    decoded: Mapping[str, Any] = json_numpy.loads(response.content)
    return decoded


class MolmoAct2LiberoPolicy:
    """Send LIBERO observations to a MolmoAct2-LIBERO ``/act`` server."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "requests": "uv pip install inspect-robots-libero",
        "json_numpy": "uv pip install inspect-robots-libero",
    }

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:8204",
        endpoint: str = "/act",
        timeout_s: float = 120.0,
        cam_height: int = 256,
        cam_width: int = 256,
        action_horizon: int = 10,
        n_action_steps: int = 10,
        norm_tag: str = "libero",
        post_fn: PostFn | None = None,
    ) -> None:
        if timeout_s <= 0 or not np.isfinite(timeout_s):
            raise ValueError("timeout_s must be finite and > 0")
        if cam_height < 1 or cam_width < 1:
            raise ValueError("cam_height and cam_width must be >= 1")
        if action_horizon < 1 or n_action_steps < 1:
            raise ValueError("action_horizon and n_action_steps must be >= 1")
        self._server_url = server_url.rstrip("/")
        self._endpoint = "/" + endpoint.lstrip("/")
        self._timeout_s = timeout_s
        self._cam_height = cam_height
        self._cam_width = cam_width
        self._n_action_steps = n_action_steps
        self._norm_tag = norm_tag
        self._post_fn = post_fn if post_fn is not None else _default_post
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name="molmoact2-libero",
            action_space=action_space(),
            observation_space=observation_space(cam_height, cam_width),
        )
        self.config = PolicyConfig(action_horizon=action_horizon)

    @property
    def server_url(self) -> str:
        """Expose the server address for Inspect Robots connection-error hints."""
        return self._server_url

    @property
    def server_metadata_url(self) -> str:
        """Use the generic MolmoAct2 server's read-only health endpoint."""
        return self._server_url + "/health"

    @property
    def remedy(self) -> str:
        """Return the server startup hint used when a connection fails."""
        return (
            "start experiments/scripts/serve_policy.py with checkpoint "
            "allenai/MolmoAct2-LIBERO, --image_keys libero, and --norm_tag libero"
        )

    def reset(self, scene: Scene) -> None:
        """Store the scene instruction and reset local inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Request and validate a non-empty 7-D action chunk."""
        images: dict[str, npt.NDArray[Any]] = {}
        expected_image_shape = (self._cam_height, self._cam_width, 3)
        for camera in CAMERA_NAMES:
            if camera not in observation.images:
                raise ValueError(f"observation missing camera {camera!r}")
            image = np.asarray(observation.images[camera])
            if image.shape != expected_image_shape:
                raise ValueError(
                    f"camera {camera!r} has shape {image.shape}, expected {expected_image_shape}"
                )
            images[camera] = image
        if STATE_KEY not in observation.state:
            raise ValueError(f"observation missing state key {STATE_KEY!r}")
        state = np.asarray(observation.state[STATE_KEY], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state has shape {state.shape}, expected ({STATE_DIM},)")
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values")
        payload: dict[str, Any] = {
            **images,
            "state": state,
            "instruction": observation.instruction or self._instruction or "",
            "norm_tag": self._norm_tag,
            "n_action_steps": self._n_action_steps,
        }
        start = time.perf_counter()
        response = self._post_fn(self._server_url + self._endpoint, payload, self._timeout_s)
        elapsed = time.perf_counter() - start
        if "actions" not in response:
            raise ValueError("/act response missing 'actions'")
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"/act returned actions of shape {actions.shape}; expected (N, {ACTION_DIM})"
            )
        if actions.shape[0] == 0:
            raise ValueError("/act returned an empty action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("/act returned non-finite actions")
        meta: dict[str, Any] = {}
        if "latency_ms" in response:
            latency_ms = float(response["latency_ms"])
            if latency_ms < 0:
                raise ValueError(f"/act returned negative latency_ms: {latency_ms!r}")
            meta["server_latency_ms"] = latency_ms
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            inference_latency_s=elapsed,
            meta=meta,
        )


class LiberoNoopPolicy:
    """Deterministic no-motion policy for environment and camera smoke tests."""

    def __init__(self, *, cam_height: int = 256, cam_width: int = 256) -> None:
        if cam_height < 1 or cam_width < 1:
            raise ValueError("cam_height and cam_width must be >= 1")
        self.info = PolicyInfo(
            name="libero-noop",
            action_space=action_space(),
            observation_space=observation_space(cam_height, cam_width),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Scene) -> None:
        """Begin a smoke-test trial; this policy has no state to clear."""

    def act(self, observation: Observation) -> ActionChunk:
        """Hold the end effector still and leave the gripper open."""
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[-1] = -1.0
        return ActionChunk(actions=[Action(data=action)])
