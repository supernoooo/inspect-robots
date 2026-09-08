"""Inspect Robots embodiment for the bimanual YAM task in Isaac Lab.

Isaac Sim, Isaac Lab, Gymnasium, and Torch are imported lazily. Importing the
plugin and inspecting its static contract therefore works outside an Isaac Lab
environment; only :meth:`IsaacSim2YamEmbodiment.reset` launches the simulator.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from inspect_robots import EmbodimentInfo, Observation, Scene, StepResult
from inspect_robots_isaacsim_2yam.asset import PreparedMjcf, prepare_isaac_mjcf
from inspect_robots_isaacsim_2yam.contract import ACTION_DIM, CAMERA_NAMES, action_space
from inspect_robots_isaacsim_2yam.contract import observation_space as yam_observation_space

if TYPE_CHECKING:
    from inspect_robots import Action

ISAAC_TASK_ID = "Isaac-Put-Everything-In-Box-2YAM-Direct-v0"
_CAPABILITIES = frozenset({"seedable", "resettable", "privileged_success", "renderable"})
_ACTIVE_APP: Any | None = None


def _missing_isaac(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "Isaac Sim / Isaac Lab is not importable in this environment "
        f"({exc}). Run inspect-robots from the Isaac Lab Python environment and install "
        "this plugin there. The static contract can be inspected without Isaac."
    )


def _candidate_asset_paths() -> tuple[Path, ...]:
    explicit = os.environ.get("MOLMOACT2_YAM_MJCF")
    root = os.environ.get("MOLMOACT2_ROOT")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if root:
        candidates.append(
            Path(root).expanduser()
            / "sim_eval/assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml"
        )
    return tuple(candidates)


class IsaacSim2YamEmbodiment:
    """Run MolmoAct2's bimanual YAM wire contract in an independent Isaac Lab task.

    Parameters:
        asset_path: Absolute path to MolmoAct2's
            ``bimanual_yam_linear_flattened.xml`` asset. It can instead be set by
            ``MOLMOACT2_YAM_MJCF`` or inferred from ``MOLMOACT2_ROOT``.
        headless: Launch Isaac Sim without a GUI. Cameras remain enabled.
        device: Isaac Lab simulation and Torch device.
        cam_height: Camera height expected by the policy. The YAM checkpoint uses 360.
        cam_width: Camera width expected by the policy. The YAM checkpoint uses 640.
        control_hz: Environment control rate. Physics runs at 120 Hz with decimation.
    """

    def __init__(
        self,
        *,
        asset_path: str | None = None,
        headless: bool = True,
        device: str = "cuda:0",
        cam_height: int = 360,
        cam_width: int = 640,
        control_hz: float = 30.0,
        spawn_noise: float = 0.02,
    ) -> None:
        if cam_height < 1 or cam_width < 1:
            raise ValueError("cam_height and cam_width must be >= 1")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if not np.isfinite(spawn_noise) or spawn_noise < 0:
            raise ValueError("spawn_noise must be finite and >= 0")
        self.asset_path = asset_path
        self.headless = headless
        self.device = device
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.control_hz = control_hz
        self.spawn_noise = spawn_noise
        self.info = EmbodimentInfo(
            name="isaacsim-2yam",
            action_space=action_space(),
            observation_space=yam_observation_space(cam_height, cam_width),
            control_hz=control_hz,
            is_simulated=True,
            capabilities=_CAPABILITIES,
            supported_setups=frozenset({"isaacsim-2yam-put-everything-in-box"}),
            docs=(
                "Bimanual YAM. Action order is left j0..j5, left gripper, right "
                "j0..j5, right gripper. Joint targets are absolute radians. "
                "Gripper 0 is closed and 1 is open."
            ),
        )
        self._app: Any | None = None
        self._env: Any | None = None
        self._torch: Any | None = None
        self._instruction: str | None = None
        self._prepared_asset: PreparedMjcf | None = None

    def _resolve_asset_path(self) -> Path:
        candidates: tuple[Path, ...] = (
            (Path(self.asset_path).expanduser(),) if self.asset_path else ()
        )
        for candidate in (*candidates, *_candidate_asset_paths()):
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        raise RuntimeError(
            "YAM MJCF asset was not found. Download MolmoAct2 simulation assets with "
            "`uv run python sim_eval/scripts/download_assets.py`, then pass "
            "`-E asset_path=/absolute/path/to/bimanual_yam_linear_flattened.xml` or set "
            "MOLMOACT2_ROOT."
        )

    def _ensure_app(self) -> Any:
        """Launch the process-wide Isaac SimulationApp with cameras enabled."""
        global _ACTIVE_APP
        if self._app is not None:
            return self._app
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on external Isaac install
            raise _missing_isaac(exc) from exc
        self._torch = torch
        if _ACTIVE_APP is not None:
            self._app = _ACTIVE_APP
            return self._app
        try:
            from isaaclab.app import AppLauncher
        except ImportError as exc:  # pragma: no cover - depends on external Isaac install
            raise _missing_isaac(exc) from exc
        self._app = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
        _ACTIVE_APP = self._app
        return self._app

    def _ensure_env(self) -> Any:
        """Create the bundled Isaac Lab direct environment on first use."""
        if self._env is not None:
            return self._env
        asset_path = self._resolve_asset_path()
        self._prepared_asset = prepare_isaac_mjcf(asset_path)
        self._ensure_app()
        try:
            import gymnasium as gym
            from isaacsim.core.utils.extensions import enable_extension

            from inspect_robots_isaacsim_2yam.isaac_env import make_env_cfg, register_env
        except ImportError as exc:
            raise _missing_isaac(exc) from exc
        # Isaac Lab's rendering-only experience does not start the MJCF importer.
        # The converter command must be registered before the articulation spawns.
        enable_extension("isaacsim.asset.importer.mjcf")
        register_env()
        cfg = make_env_cfg(
            asset_path=str(self._prepared_asset.path),
            device=self.device,
            height=self.cam_height,
            width=self.cam_width,
            control_hz=self.control_hz,
            spawn_noise=self.spawn_noise,
        )
        self._env = gym.make(ISAAC_TASK_ID, cfg=cfg, render_mode="rgb_array")
        return self._env

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset physics and preserve the language instruction for the episode."""
        env = self._ensure_env()
        self._instruction = scene.instruction
        obs, _info = env.reset(seed=seed if seed is not None else scene.init_seed)
        return self._to_observation(obs, self._instruction)

    def step(self, action: Action) -> StepResult:
        """Apply one 14-D absolute joint target and translate the Isaac transition."""
        vector = np.asarray(action.data, dtype=np.float32)
        if vector.shape != (ACTION_DIM,):
            raise ValueError(f"expected action shape ({ACTION_DIM},), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("action contains non-finite values")
        env = self._ensure_env()
        torch = self._torch
        assert torch is not None
        tensor = torch.as_tensor(vector, device=self.device).reshape(1, -1)
        obs, reward, terminated, truncated, info = env.step(tensor)
        success = bool(_scalar(info.get("success", False))) if isinstance(info, Mapping) else False
        term = bool(_scalar(terminated))
        return StepResult(
            observation=self._to_observation(obs, self._instruction),
            reward=float(_scalar(reward)),
            terminated=term,
            termination_reason="success" if term and success else ("failure" if term else None),
            truncated=bool(_scalar(truncated)),
            info={"success": success},
        )

    def close(self) -> None:
        """Close the environment and the process-wide Isaac application."""
        global _ACTIVE_APP
        if self._env is not None:
            self._env.close()
            self._env = None
        # SimulationApp.close() can terminate Kit's Python shutdown sequence,
        # so release the private asset copy before closing the application.
        if self._prepared_asset is not None:
            self._prepared_asset.cleanup()
            self._prepared_asset = None
        if self._app is not None:
            self._app.close()
            if _ACTIVE_APP is self._app:
                _ACTIVE_APP = None
            self._app = None
        self._torch = None

    def __enter__(self) -> IsaacSim2YamEmbodiment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _to_observation(self, raw: Any, instruction: str | None) -> Observation:
        group = raw.get("policy", raw) if isinstance(raw, Mapping) else raw
        images: dict[str, npt.NDArray[np.uint8]] = {}
        state: dict[str, npt.NDArray[np.float64]] = {}
        if isinstance(group, Mapping):
            for camera in CAMERA_NAMES:
                if camera in group:
                    images[camera] = _to_image(group[camera])
            if "joint_pos" in group:
                state["joint_pos"] = _to_float_array(group["joint_pos"])
        return Observation(images=images, state=state, instruction=instruction)


def _np(value: Any) -> npt.NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    array = _np(value).reshape(-1)
    return float(array[0]) if array.size else 0.0


def _to_float_array(value: Any) -> npt.NDArray[np.float64]:
    array = _np(value).astype(np.float64)
    return array[0] if array.ndim > 1 and array.shape[0] == 1 else array


def _to_image(value: Any) -> npt.NDArray[np.uint8]:
    array = _np(value)
    if array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return array.astype(np.uint8)
