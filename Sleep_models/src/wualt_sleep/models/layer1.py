"""§5 — Layer 1, sleep onset detection.

Ships in three versions (§5 table):
  MVP  rule-based multi-signal AND  (accel-still + HR below RHR + temp rising)
  V1   RF ~100 trees AND XGBoost ~100 rounds, head-to-head, best-on-LOSO ships
  V2   winner retrained on accumulated pilot ground truth

ROADMAP DEVIATION (see decisions D-003): V1's head-to-head is deferred past the
DREAMT phase. DREAMT contains ~80 onset transitions total — one per subject — which
will not support a learned onset model that generalises. DREAMT is used to tune and
ceiling-test the rule baseline; the bake-off harness is built now and runs when pilot
ground truth exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..featuredict import FeatureDictionary
from ..interfaces import OnsetContext
from ..schema import LabelledRow, Row

__all__ = ["RuleOnsetDetector", "RuleThresholds", "TreeOnsetDetector"]


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """Tuned under LOSO against the PSG-derived PREP->sleep boundary.

    Defaults are placeholders, NOT tuned values. Phase 4 replaces them.
    """

    still_fraction_min: float = 0.90
    hr_below_rhr_bpm: float = 2.0
    temp_slope_min: float = 0.10  # degC/hr, rising distal temp
    require_all: bool = True  # §5 specifies AND, not a vote


class RuleOnsetDetector:
    """MVP onset detector, and the permanent fallback for §4's 6-hour timeout.

    It is not throwaway: the escape hatch depends on it forever, so it needs to be
    genuinely tuned, not just present.

    Returns a graded probability rather than a hard fire, because the state machine
    gates on 0.5 and 0.7 (§4). A detector that only returns 0.0/1.0 collapses
    CANDIDATE_ONSET into a single-row trigger and defeats the sustained-evidence
    design.
    """

    version = "rule_v0"

    def __init__(self, fd: FeatureDictionary, thresholds: RuleThresholds | None = None) -> None:
        self.fd = fd
        self.thresholds = thresholds or RuleThresholds()

    def predict_proba(self, row: Row, context: OnsetContext) -> float:
        raise NotImplementedError("Phase 4: rule baseline; see §5 and D-008 for grading")

    def fit(self, data: list[LabelledRow]) -> None:
        """Threshold search under LOSO. Not gradient fitting — a grid or coordinate
        sweep maximising onset-within-±15-min (§9)."""
        raise NotImplementedError("Phase 4: LOSO threshold tuning")


class TreeOnsetDetector:
    """V1 — RF or XGBoost behind one interface, for the §5 head-to-head.

    Both arms share the feature set, both get focal loss (γ=2.0), label smoothing
    (α=0.05) and isotonic calibration (§8); the stronger on LOSO ships. Keeping them
    behind one class makes the bake-off a config change rather than two scripts —
    §10 says the same shootout repeats post-pilot.

    Blocked on pilot data (D-003).
    """

    version = "tree_v0"

    def __init__(self, fd: FeatureDictionary, *, backend: str = "xgboost") -> None:
        if backend not in ("xgboost", "random_forest"):
            raise ValueError(f"unknown backend {backend!r}")
        self.fd = fd
        self.backend = backend

    def predict_proba(self, row: Row, context: OnsetContext) -> float:
        raise NotImplementedError("Phase 7: blocked on pilot onset labels (D-003)")

    def fit(self, data: list[LabelledRow]) -> None:
        raise NotImplementedError("Phase 7: blocked on pilot onset labels (D-003)")
