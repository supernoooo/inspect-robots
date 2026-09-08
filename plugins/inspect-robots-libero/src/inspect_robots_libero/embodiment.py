"""LIBERO single-arm Franka embodiment for Inspect Robots."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from inspect_robots import EmbodimentInfo, Observation, Scene, StepResult
from inspect_robots_libero.contract import (
    ACTION_DIM,
    CAMERA_NAMES,
    RAW_CAMERA_NAMES,
    STATE_KEY,
    action_space,
    observation_space,
    robot_state,
)

if TYPE_CHECKING:
    from inspect_robots import Action

_CAPABILITIES = frozenset({"seedable", "resettable", "privileged_success", "renderable"})
_LIBERO_ROOT_ENV = "INSPECT_ROBOTS_LIBERO_ROOT"


def _prepare_headless_runtime() -> None:
    """Select robust defaults before robosuite imports MuJoCo and Numba."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    cache_root = Path(tempfile.gettempdir()) / f"inspect-robots-libero-{os.getuid()}"
    for variable, child in (
        ("NUMBA_CACHE_DIR", "numba"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        if variable in os.environ:
            continue
        path = cache_root / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


def _requested_package_root(value: str) -> tuple[Path, Path]:
    """Resolve a LIBERO checkout/package root and its Python import root."""
    root = Path(value).expanduser().resolve()
    candidates = (
        (root / "libero" / "libero", root),
        (root / "libero", root.parent),
        (root, root.parent.parent),
    )
    for package_root, import_root in candidates:
        if (package_root / "__init__.py").is_file() and all(
            (package_root / child).is_dir()
            for child in ("assets", "bddl_files", "init_files")
        ):
            return package_root, import_root
    raise RuntimeError(
        f"{_LIBERO_ROOT_ENV}={value!r} is not a LIBERO checkout or package root; "
        "expected assets/, bddl_files/, and init_files/"
    )


def _local_path_resolver(package_root: Path) -> Any:
    """Build a get_libero_path replacement pinned to one installed checkout."""
    paths = {
        "benchmark_root": package_root,
        "bddl_files": package_root / "bddl_files",
        "init_states": package_root / "init_files",
        "datasets": package_root.parent / "datasets",
        "assets": package_root / "assets",
    }

    def get_path(key: str) -> str:
        if key not in paths:
            raise KeyError(f"unknown LIBERO path key {key!r}; available: {sorted(paths)}")
        return str(paths[key])

    return get_path


def _missing_libero(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "LIBERO is not importable in this environment "
        f"({exc}). Install the official LIBERO repository and its benchmark assets "
        "in the Python environment that runs inspect-robots."
    )


def _libero_api() -> tuple[Any, Any, Any]:
    _prepare_headless_runtime()
    requested_root = os.environ.get(_LIBERO_ROOT_ENV)
    requested_package: Path | None = None
    if requested_root:
        requested_package, import_root = _requested_package_root(requested_root)
        import_root_text = str(import_root)
        if import_root_text not in sys.path:
            sys.path.insert(0, import_root_text)
    try:
        package = import_module("libero.libero")
        get_libero_path = package.get_libero_path
        package_file = getattr(package, "__file__", None)
        package_root = requested_package
        if package_root is None and isinstance(package_file, str):
            package_root = Path(package_file).resolve().parent
        if package_root is not None:
            get_libero_path = _local_path_resolver(package_root)
            # Import benchmark/envs only after replacing the resolver: both
            # modules bind get_libero_path at import time.
            package.get_libero_path = get_libero_path
        benchmark = getattr(package, "benchmark", None)
        if benchmark is None:
            benchmark = import_module("libero.libero.benchmark")
        benchmark.get_libero_path = get_libero_path
        envs = import_module("libero.libero.envs")
        OffScreenRenderEnv = envs.OffScreenRenderEnv
    except ImportError as exc:  # pragma: no cover - depends on external LIBERO install
        raise _missing_libero(exc) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def load_suite(name: str) -> Any:
    """Instantiate a named LIBERO benchmark suite with an actionable validation error."""
    benchmark, _get_libero_path, _env_type = _libero_api()
    suites = benchmark.get_benchmark_dict()
    if name not in suites:
        raise ValueError(f"unknown LIBERO suite {name!r}; available: {sorted(suites)}")
    suite = suites[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"LIBERO suite {name!r} has no tasks")
    return suite


class LiberoEmbodiment:
    """Wrap official LIBERO tasks as an Inspect Robots simulator embodiment.

    A task scene may select ``libero_suite``, ``libero_task_id``, and
    ``libero_init_state_id`` through metadata. Without those fields, the
    constructor defaults are used, which also supports ad-hoc instructions.
    """

    def __init__(
        self,
        *,
        suite: str = "libero_goal",
        task_id: int = 0,
        init_state_id: int = 0,
        cam_height: int = 256,
        cam_width: int = 256,
        control_hz: float = 20.0,
        num_steps_wait: int = 10,
        use_init_states: bool = True,
    ) -> None:
        if task_id < 0 or init_state_id < 0:
            raise ValueError("task_id and init_state_id must be >= 0")
        if cam_height < 1 or cam_width < 1:
            raise ValueError("cam_height and cam_width must be >= 1")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if num_steps_wait < 0:
            raise ValueError("num_steps_wait must be >= 0")
        self.suite = suite
        self.task_id = task_id
        self.init_state_id = init_state_id
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.control_hz = control_hz
        self.num_steps_wait = num_steps_wait
        self.use_init_states = use_init_states
        self.info = EmbodimentInfo(
            name="libero",
            action_space=action_space(),
            observation_space=observation_space(cam_height, cam_width),
            control_hz=control_hz,
            is_simulated=True,
            capabilities=_CAPABILITIES,
            docs=(
                "Single-arm Franka in LIBERO. Actions are normalized relative end-effector "
                "position and axis-angle deltas followed by a continuous gripper command."
            ),
        )
        self._env: Any | None = None
        self._task_suite: Any | None = None
        self._active_key: tuple[str, int] | None = None
        self._instruction: str | None = None
        self._init_states: dict[tuple[str, int], Any] = {}

    def _ensure_env(self, suite_name: str, task_id: int) -> Any:
        key = (suite_name, task_id)
        if self._env is not None and self._active_key == key:
            return self._env
        if self._env is not None:
            self._env.close()
            self._env = None
        _benchmark, get_libero_path, env_type = _libero_api()
        task_suite = load_suite(suite_name)
        if task_id >= len(task_suite.tasks):
            raise ValueError(
                f"LIBERO task_id {task_id} is out of range for {suite_name!r}; "
                f"expected 0..{len(task_suite.tasks) - 1}"
            )
        task = task_suite.get_task(task_id)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self._env = env_type(
            bddl_file_name=bddl_path,
            camera_heights=self.cam_height,
            camera_widths=self.cam_width,
        )
        self._task_suite = task_suite
        self._active_key = key
        return self._env

    def _load_init_states(self, suite_name: str, task_id: int) -> Any:
        key = (suite_name, task_id)
        if key in self._init_states:
            return self._init_states[key]
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - installed with LIBERO in live use
            raise _missing_libero(exc) from exc
        _benchmark, get_libero_path, _env_type = _libero_api()
        assert self._task_suite is not None
        task = self._task_suite.tasks[task_id]
        path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        if not path.is_file():
            raise RuntimeError(f"LIBERO initial-state file does not exist: {path}")
        states = torch.load(path, weights_only=False)  # nosec B614 - official LIBERO asset
        if len(states) == 0:
            raise RuntimeError(f"LIBERO initial-state file is empty: {path}")
        self._init_states[key] = states
        return states

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Select the task and official initial state encoded by ``scene``."""
        suite_name = str(scene.metadata.get("libero_suite", self.suite))
        task_id = int(scene.metadata.get("libero_task_id", self.task_id))
        init_state_id = int(scene.metadata.get("libero_init_state_id", self.init_state_id))
        if task_id < 0 or init_state_id < 0:
            raise ValueError("scene LIBERO task and initial-state ids must be >= 0")
        env = self._ensure_env(suite_name, task_id)
        effective_seed = seed if seed is not None else scene.init_seed
        env.seed(effective_seed)
        raw = env.reset()
        if self.use_init_states:
            states = self._load_init_states(suite_name, task_id)
            raw = env.set_init_state(states[init_state_id % len(states)])
        for _ in range(self.num_steps_wait):
            raw, _reward, _done, _info = env.step(
                np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            )
        for robot in env.robots:
            robot.controller.use_delta = True
        self._instruction = scene.instruction
        return self._to_observation(raw, self._instruction)

    def step(self, action: Action) -> StepResult:
        """Apply one LIBERO relative end-effector action without auto-resetting."""
        if self._env is None:
            raise RuntimeError("step() called before reset()")
        vector = np.asarray(action.data, dtype=np.float32)
        if vector.shape != (ACTION_DIM,):
            raise ValueError(f"expected action shape ({ACTION_DIM},), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("action contains non-finite values")
        raw, reward, done, raw_info = self._env.step(vector)
        success = bool(self._env.check_success())
        terminated = bool(done) or success
        info = dict(raw_info) if isinstance(raw_info, Mapping) else {}
        info.update(
            {
                "success": success,
                "libero_suite": self._active_key[0] if self._active_key else self.suite,
                "libero_task_id": self._active_key[1] if self._active_key else self.task_id,
            }
        )
        return StepResult(
            observation=self._to_observation(raw, self._instruction),
            reward=float(reward),
            terminated=terminated,
            termination_reason=(
                "success" if success else ("environment_done" if terminated else None)
            ),
            truncated=False,
            info=info,
        )

    def close(self) -> None:
        """Close the current MuJoCo environment; safe before reset and on repeat calls."""
        if self._env is not None:
            self._env.close()
            self._env = None
        self._task_suite = None
        self._active_key = None

    def __enter__(self) -> LiberoEmbodiment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _to_observation(self, raw: Mapping[str, Any], instruction: str | None) -> Observation:
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for output_name, raw_name in zip(CAMERA_NAMES, RAW_CAMERA_NAMES, strict=True):
            if raw_name not in raw:
                raise ValueError(f"LIBERO observation is missing camera {raw_name!r}")
            image = np.asarray(raw[raw_name])
            expected_shape = (self.cam_height, self.cam_width, 3)
            if image.shape != expected_shape:
                raise ValueError(
                    f"LIBERO camera {raw_name!r} has shape {image.shape}, "
                    f"expected {expected_shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(
                    f"LIBERO camera {raw_name!r} has dtype {image.dtype}, expected uint8"
                )
            # Official MolmoAct2 LIBERO preprocessing rotates raw LIBERO frames by 180 degrees.
            images[output_name] = np.ascontiguousarray(image[::-1, ::-1])
        return Observation(
            images=images,
            state={STATE_KEY: robot_state(dict(raw))},
            instruction=instruction,
        )
