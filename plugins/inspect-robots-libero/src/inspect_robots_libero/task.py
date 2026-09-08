"""Task factories for the five official LIBERO benchmark suites."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from inspect_robots import Scene, Task
from inspect_robots.scorer import success_at_end
from inspect_robots_libero.embodiment import load_suite

SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _parse_ids(value: Iterable[int] | str | None, total: int, name: str) -> list[int]:
    if value is None:
        return list(range(total))
    if isinstance(value, str):
        ids = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    else:
        ids = sorted({int(item) for item in value})
    if not ids:
        raise ValueError(f"{name} must select at least one id")
    for item in ids:
        if item < 0 or item >= total:
            raise ValueError(f"{name} id {item} is out of range 0..{total - 1}")
    return ids


def libero_suite_task(
    suite_name: str,
    *,
    task_ids: Iterable[int] | str | None = None,
    init_state_ids: Iterable[int] | str | None = None,
    episodes_per_task: int = 1,
    seed: int = 0,
    max_steps: int | None = None,
) -> Task:
    """Build an Inspect Robots task from official LIBERO task descriptions."""
    if suite_name not in SUITE_MAX_STEPS:
        raise ValueError(
            f"unsupported LIBERO suite {suite_name!r}; expected {sorted(SUITE_MAX_STEPS)}"
        )
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be >= 1")
    suite = load_suite(suite_name)
    selected_tasks = _parse_ids(task_ids, len(suite.tasks), "task")
    if init_state_ids is None:
        selected_states = list(range(episodes_per_task))
    else:
        # Official task files normally contain 50 states. Use a generous upper
        # validation bound here; exact modulo selection is handled at reset.
        selected_states = _parse_ids(init_state_ids, 10_000, "initial-state")
    scenes: list[Scene] = []
    scene_index = 0
    for task_id in selected_tasks:
        task = suite.get_task(task_id)
        for init_state_id in selected_states:
            scenes.append(
                Scene(
                    id=f"{suite_name}-task-{task_id:03d}-init-{init_state_id:03d}",
                    instruction=str(task.language),
                    init_seed=seed + scene_index,
                    metadata={
                        "libero_suite": suite_name,
                        "libero_task_id": task_id,
                        "libero_init_state_id": init_state_id,
                    },
                )
            )
            scene_index += 1
    return Task(
        name=suite_name.replace("_", "-"),
        scenes=scenes,
        scorer=success_at_end(),
        max_steps=max_steps if max_steps is not None else SUITE_MAX_STEPS[suite_name],
        metadata={"benchmark": "LIBERO", "suite": suite_name},
    )


def libero_spatial(**kwargs: Any) -> Task:
    """Return the LIBERO-Spatial task suite."""
    return libero_suite_task("libero_spatial", **kwargs)


def libero_object(**kwargs: Any) -> Task:
    """Return the LIBERO-Object task suite."""
    return libero_suite_task("libero_object", **kwargs)


def libero_goal(**kwargs: Any) -> Task:
    """Return the LIBERO-Goal task suite."""
    return libero_suite_task("libero_goal", **kwargs)


def libero_10(**kwargs: Any) -> Task:
    """Return the LIBERO-10 long-horizon task suite."""
    return libero_suite_task("libero_10", **kwargs)


def libero_90(**kwargs: Any) -> Task:
    """Return the LIBERO-90 task suite."""
    return libero_suite_task("libero_90", **kwargs)
