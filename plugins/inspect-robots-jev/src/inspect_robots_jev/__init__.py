"""Safe Jev policy entry points for the bimanual YAM embodiment."""

from __future__ import annotations

from inspect_robots_jev.contract import DecodedInput, InputError, decode_observation, hold
from inspect_robots_jev.candidates import Candidate, CandidateGenerator, CandidateSet, Stage
from inspect_robots_jev.motion import MotionLimits, MotionPlanner, MotionResult
from inspect_robots_jev.agent_candidates import (
    AgentCandidateValidator, FilteredProposal, ValidatedActionCandidate,
    ValidatedCandidateSet,
)
from inspect_robots_jev.jev_choice import ChoiceError, ChoiceOption, ChoiceResult, JevChoiceClient
from inspect_robots_jev.policy import JevDirectPolicy, JevHybridPolicy
from inspect_robots_jev.agent_policy import JevAgentPolicy

__all__ = [
    "AgentCandidateValidator",
    "DecodedInput",
    "Candidate",
    "CandidateGenerator",
    "CandidateSet",
    "ChoiceError",
    "ChoiceOption",
    "ChoiceResult",
    "InputError",
    "FilteredProposal",
    "JevDirectPolicy",
    "JevAgentPolicy",
    "JevHybridPolicy",
    "JevChoiceClient",
    "MotionLimits",
    "MotionPlanner",
    "MotionResult",
    "Stage",
    "ValidatedActionCandidate",
    "ValidatedCandidateSet",
    "decode_observation",
    "hold",
    "jev_direct_policy",
    "jev_agent_policy",
    "jev_hybrid_policy",
]

__version__ = "0.1.0"


def jev_direct_policy(**kwargs: object) -> JevDirectPolicy:
    """Create the registered direct policy."""
    return JevDirectPolicy(**kwargs)


def jev_hybrid_policy(**kwargs: object) -> JevHybridPolicy:
    """Create the registered hybrid policy."""
    return JevHybridPolicy(**kwargs)


def jev_agent_policy(**kwargs: object) -> JevAgentPolicy:
    """Create the registered Agent/JEV policy."""
    return JevAgentPolicy(**kwargs)
