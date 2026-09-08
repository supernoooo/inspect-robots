"""Bimanual YAM Isaac Lab plugin for Inspect Robots.

Installing this package registers embodiment ``isaacsim-2yam`` and task
``isaacsim-2yam-put-everything-in-box`` without modifying the existing Franka
``isaacsim`` plugin.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_isaacsim_2yam.embodiment import IsaacSim2YamEmbodiment
from inspect_robots_isaacsim_2yam.task import put_everything_in_box

__all__ = [
    "IsaacSim2YamEmbodiment",
    "isaacsim_2yam_embodiment",
    "put_everything_in_box",
]

__version__ = "0.1.0"


def isaacsim_2yam_embodiment(**kwargs: Any) -> IsaacSim2YamEmbodiment:
    """Build the ``isaacsim-2yam`` embodiment from CLI-forwarded keyword arguments."""
    return IsaacSim2YamEmbodiment(**kwargs)
