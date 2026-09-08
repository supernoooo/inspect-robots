"""LIBERO environment, tasks, and MolmoAct2 policy plugin for Inspect Robots."""

from __future__ import annotations

from typing import Any

from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.policy import LiberoNoopPolicy, MolmoAct2LiberoPolicy
from inspect_robots_libero.task import (
    libero_10,
    libero_90,
    libero_goal,
    libero_object,
    libero_spatial,
    libero_suite_task,
)

__all__ = [
    "LiberoEmbodiment",
    "LiberoNoopPolicy",
    "MolmoAct2LiberoPolicy",
    "libero_10",
    "libero_90",
    "libero_embodiment",
    "libero_goal",
    "libero_noop_policy",
    "libero_object",
    "libero_spatial",
    "libero_suite_task",
    "molmoact2_libero_policy",
]

__version__ = "0.1.0"


def libero_embodiment(**kwargs: Any) -> LiberoEmbodiment:
    """Build the registry's ``libero`` embodiment from CLI keyword arguments."""
    return LiberoEmbodiment(**kwargs)


def molmoact2_libero_policy(**kwargs: Any) -> MolmoAct2LiberoPolicy:
    """Build the registry's ``molmoact2-libero`` HTTP policy adapter."""
    return MolmoAct2LiberoPolicy(**kwargs)


def libero_noop_policy(**kwargs: Any) -> LiberoNoopPolicy:
    """Build the registry's model-free LIBERO smoke-test policy."""
    return LiberoNoopPolicy(**kwargs)
