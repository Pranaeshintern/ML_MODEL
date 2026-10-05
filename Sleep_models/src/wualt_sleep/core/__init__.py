"""Deterministic core: state machine (§4), HMM (§7), session engine (§7).

Nothing here is learned in the MVP, so all of it is buildable and testable without a
dataset — which is why Phase 1 of the roadmap starts here.
"""

from .hmm import PassthroughSmoother, ViterbiResult, ViterbiSmoother
from .session import SessionEngine, compute_breakdown, compute_waso
from .state_machine import MutableOnsetContext, SleepStateMachine, Transition

__all__ = [
    "SleepStateMachine",
    "Transition",
    "MutableOnsetContext",
    "ViterbiSmoother",
    "ViterbiResult",
    "PassthroughSmoother",
    "SessionEngine",
    "compute_breakdown",
    "compute_waso",
]
