"""§6 — Layer 2, 4-class sleep-stage classification.

MVP        XGBoost + rolling-window features
Production TCN (per-row MLP encoder -> 2 dilated causal blocks -> 4-class head)

§6's own comparison table says the Transformer overfits at n=50 subjects. We have 80
subjects x 1 night ~= 2,000 stageable rows, which puts even the TCN out of reach —
§6 targets ~500+ subjects for production. The TCN class below is a placeholder for
the post-pilot phase, deliberately not implemented now.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..featuredict import FeatureDictionary
from ..schema import LabelledRow, Row

__all__ = ["XGBStageClassifier", "TCNStageClassifier", "RollingWindowSpec"]


@dataclass(frozen=True, slots=True)
class RollingWindowSpec:
    """Context the MVP classifier sees around the row being classified.

    Windows are TIME-based, not index-based. §1 says row cadence is sporadic before
    onset and dense after; an index-based window silently changes meaning when the
    spacing changes. Layer 2 only runs during ASLEEP (dense), but the training data
    is built by the same code that serves Layer 1, so the distinction must hold.
    """

    lookback_rows: int = 4
    lookahead_rows: int = 2  # non-causal — legal because staging is not real-time
    stats: tuple[str, ...] = ("mean", "std", "delta")


class XGBStageClassifier:
    """MVP staging model.

    Training config per §6/§8:
      - weighted cross-entropy, class weights 1/freq capped at 3x
      - label smoothing α=0.10
      - per-class isotonic calibration post-training
      - cycle-aware baseline subtraction applied BEFORE inference (§6)

    OPEN (D-002): hard majority labels vs stage-proportion targets. The `objective`
    switch below exists because that decision is not yet made — the §9 user metric is
    time-in-stage error, a proportion metric, and majority voting discards 29 of the
    30 epochs behind each row.
    """

    version = "xgb_v0"
    n_classes = 4

    def __init__(
        self,
        fd: FeatureDictionary,
        window: RollingWindowSpec | None = None,
        *,
        objective: str = "multiclass",  # "multiclass" | "proportion"
    ) -> None:
        if objective not in ("multiclass", "proportion"):
            raise ValueError(f"unknown objective {objective!r}")
        self.fd = fd
        self.window = window or RollingWindowSpec()
        self.objective = objective

    def predict_proba(self, rows: list[Row]) -> np.ndarray:
        raise NotImplementedError("Phase 4: XGBoost staging")

    def fit(self, data: list[LabelledRow]) -> None:
        raise NotImplementedError("Phase 4: XGBoost staging")


class TCNStageClassifier:
    """Production staging model (§6). Post-pilot, ~500+ subjects.

    Architecture, for when it is time:
        LayerNorm -> Linear(32->64) -> GELU -> Linear(64->128)      per-row encoder
        stack T=8 rows [B,8,128] -> transpose [B,128,8]
        Conv1D(128->128, k=3, d=1, causal) -> LN -> GELU -> Drop 0.1
        Conv1D(128->128, k=3, d=2, causal) -> LN -> GELU -> Drop 0.1
        Linear(128->64) -> GELU -> Linear(64->4) -> softmax
    """

    version = "tcn_v0"
    n_classes = 4

    def predict_proba(self, rows: list[Row]) -> np.ndarray:
        raise NotImplementedError("Phase 7: needs ~500+ subjects (§6)")

    def fit(self, data: list[LabelledRow]) -> None:
        raise NotImplementedError("Phase 7: needs ~500+ subjects (§6)")
