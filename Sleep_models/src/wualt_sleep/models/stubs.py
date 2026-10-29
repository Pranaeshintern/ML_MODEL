"""Null components that make the skeleton walk.

These exist so the runtime is testable before any model is trained. They are the
reference implementations of each protocol — if a real component's signature drifts
from these, the contract tests catch it.

Never ship these. `ConstantStageClassifier` returns a fixed distribution; it will
happily produce a full SessionRecord that means nothing.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import OnsetContext
from ..schema import LabelledRow, Row
from ..types import Stage

__all__ = ["NeverOnsetDetector", "ScriptedOnsetDetector", "ConstantStageClassifier"]


class NeverOnsetDetector:
    """P(onset) = 0 always. The session never starts."""

    version = "stub_never_v0"

    def predict_proba(self, row: Row, context: OnsetContext) -> float:
        return 0.0

    def fit(self, data: list[LabelledRow]) -> None:
        pass


class ScriptedOnsetDetector:
    """Replays a fixed probability sequence.

    The state-machine tests are built on this: it makes §4's transition table
    directly assertable without involving a model.
    """

    version = "stub_scripted_v0"

    def __init__(self, probabilities: list[float], *, default: float = 0.0) -> None:
        self._probs = list(probabilities)
        self._default = default
        self._i = 0

    def predict_proba(self, row: Row, context: OnsetContext) -> float:
        p = self._probs[self._i] if self._i < len(self._probs) else self._default
        self._i += 1
        return p

    def fit(self, data: list[LabelledRow]) -> None:
        pass


class ConstantStageClassifier:
    """Returns the same distribution for every row."""

    version = "stub_constant_v0"
    n_classes = 4

    def __init__(self, distribution: list[float] | None = None) -> None:
        dist = np.asarray(distribution or [0.1, 0.6, 0.2, 0.1], dtype=np.float64)
        if dist.shape != (Stage.n_classes(),):
            raise ValueError("distribution must have 4 entries")
        self._dist = dist / dist.sum()

    def predict_proba(self, rows: list[Row]) -> np.ndarray:
        return np.tile(self._dist, (len(rows), 1))

    def fit(self, data: list[LabelledRow]) -> None:
        pass


class ScriptedStageClassifier:
    """Emits a scripted stage sequence as near-one-hot probabilities.

    Used to test that Layer 3 actually changes the answer — feed it an implausible
    sequence (Deep -> Wake -> Deep) and assert Viterbi repairs it.
    """

    version = "stub_scripted_stage_v0"
    n_classes = 4

    def __init__(self, stages: list[Stage], *, confidence: float = 0.9) -> None:
        self._stages = stages
        self._conf = confidence

    def predict_proba(self, rows: list[Row]) -> np.ndarray:
        out = np.full((len(rows), Stage.n_classes()), (1 - self._conf) / 3, dtype=np.float64)
        for i in range(len(rows)):
            stage = self._stages[i] if i < len(self._stages) else Stage.LIGHT
            out[i, stage.value] = self._conf
        return out

    def fit(self, data: list[LabelledRow]) -> None:
        pass
