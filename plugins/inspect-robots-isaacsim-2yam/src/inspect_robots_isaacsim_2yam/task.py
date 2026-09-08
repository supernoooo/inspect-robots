"""Inspect Robots task registration for the bundled Isaac bimanual YAM scene."""

from __future__ import annotations

from inspect_robots import Scene, Task
from inspect_robots.scorer import success_at_end


def put_everything_in_box(*, episodes: int = 10, seed: int = 0) -> Task:
    """Evaluate placing the left and right objects into the central open box."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    scenes = [
        Scene(
            id=f"isaacsim-2yam-put-everything-in-box-{index:03d}",
            instruction="put everything into the box",
            init_seed=seed + index,
            metadata={"environment": "isaacsim-2yam", "episode_index": index},
        )
        for index in range(episodes)
    ]
    return Task(
        name="isaacsim-2yam-put-everything-in-box",
        scenes=scenes,
        scorer=success_at_end(),
        max_steps=400,
    )
