"""Component protocols — the pluggable seams of the pipeline.

Everything the roadmap swaps out over time sits behind one of these:

  OnsetDetector    rule baseline (MVP)  -> RF / XGBoost bake-off (post-pilot, §5)
  StageClassifier  XGBoost (MVP)        -> TCN (production, §6)
  WakeDetector     none                 -> 30-s accelerometer channel (D-015)
  Smoother         fixed HMM matrix     -> REJECTED, see D-015 and §7.5
  Calibrator       isotonic             -> REJECTED, §7.6

Structural typing (Protocol) rather than inheritance: a component only has to match
the shape, so a sklearn estimator or an ONNX session can be adapted without wrapping
it in our class hierarchy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .schema import LabelledRow, Row
from .types import Stage

__all__ = [
    "OnsetDetector",
    "StageClassifier",
    "WakeDetector",
    "WasoBandEstimator",
    "Smoother",
    "Calibrator",
    "FeatureExtractor",
    "BaselineProvider",
    "OnsetContext",
    "SmoothingResult",
]


class OnsetContext(Protocol):
    """State-machine memory that Layer 1 needs but a single row cannot carry (§5).

    `sustained_still_rows` is explicitly listed as a Layer 1 feature, and it is by
    definition cross-row. Passing it as context keeps `Row` a pure aggregation of
    sensor data rather than a mutable carrier of pipeline state.
    """

    @property
    def rows_since_bedtime(self) -> int: ...
    @property
    def sustained_still_rows(self) -> int: ...
    @property
    def baseline_confidence(self) -> float: ...


@runtime_checkable
class OnsetDetector(Protocol):
    """Layer 1 (§5). Runs during AWAKE and CANDIDATE_ONSET only.

    Returns a probability, not a hard fire — §3 requires the probability be retained,
    because the state machine gates on absolute thresholds (0.5 / 0.7). An
    uncalibrated detector makes those thresholds meaningless, so any implementation
    that isn't intrinsically calibrated must be wrapped in a `Calibrator`.
    """

    version: str

    def predict_proba(self, row: Row, context: OnsetContext) -> float:
        """P(sleep_started_in_this_row) in [0, 1]."""
        ...

    def fit(self, data: list[LabelledRow]) -> None: ...


@runtime_checkable
class StageClassifier(Protocol):
    """Layer 2 (§6). Runs during ASLEEP.

    Takes a SEQUENCE, not a single row: the XGBoost MVP uses rolling-window features
    and the production TCN uses a T=8 row context window. A single-row signature would
    have to be rewritten for the production model.
    """

    version: str
    n_classes: int

    def predict_proba(self, rows: list[Row]) -> np.ndarray:
        """Return shape (len(rows), 4) — columns ordered per `Stage.ordered()`."""
        ...

    def fit(self, data: list[LabelledRow]) -> None: ...


@runtime_checkable
class WasoBandEstimator(Protocol):
    """Layer W's night-level half. Runs once, at finalisation.

    Separate from `WakeDetector` because they answer different questions from the same
    probabilities: the detector says WHERE wake is, at 30 s, and this says HOW MUCH there
    was across the night. The minute figure derived from the timeline is not defensible
    on its own -- 12.3 min of error against a true 21.4 on the healthy cohort, barely
    better than printing the median -- so the band is what a user is shown.

    Returns None rather than a band when the night cannot be scored: too little
    accelerometer coverage, too short, or no sleep detected. A withheld band is a
    reportable state; an invented one is not.
    """

    version: str

    def estimate_band(self, epochs: np.ndarray, pwake: np.ndarray,
                      coverage: float) -> Any | None:
        """(E, F) epoch features + (E,) P(wake) -> an estimate with `band`,
        `range_min`, `range_max` and `confidence`, or None."""
        ...


class WakeDetector(Protocol):
    """The sleep/wake channel (D-015). Runs at 30 s on accelerometry alone.

    Wake is deliberately NOT the stage classifier's job. Post-onset wake bouts average
    2.8 minutes and a 15-minute majority vote needs 7.5 to flip, so the cardiac channel
    is structurally blind to most of them — measured, its WASO estimate is worse than
    predicting the population mean. The accelerometer runs continuously and can see
    them, so it owns this decision and the stage classifier emits only Light/Deep/REM.

    Returns a PROBABILITY per 30-second epoch, not a decision, because the cut is
    scorer- and prevalence-dependent: 0.70 on a class-balanced scorer and 0.40 on a
    calibrated one are the same operating point. See `models.merge`.
    """

    def predict_wake_proba(self, accel_epochs: np.ndarray) -> np.ndarray:
        """(E, F) 30-second accelerometer features -> (E,) P(wake)."""
        ...


class SmoothingResult(Protocol):
    """What Layer 3 returns (§7): the decoded path plus per-row posteriors.

    RETAINED FOR THE RECORD, NOT USED. Layer 3 was measured and rejected (§7.5, D-015):
    HMM smoothing improved every classification metric and degraded every derived one.
    The runtime no longer calls a smoother.
    """

    @property
    def path(self) -> list[Stage]: ...
    @property
    def posterior(self) -> np.ndarray: ...
    @property
    def log_likelihood(self) -> float: ...


@runtime_checkable
class Smoother(Protocol):
    """Layer 3 (§7). Deterministic in the MVP — not itself an ML model.

    Argmax over the Layer 2 softmax produces biologically implausible sequences
    (Deep -> Wake -> Deep -> Wake); Viterbi finds the globally most-likely sequence
    under the transition constraints.
    """

    version: str

    def decode(self, probabilities: np.ndarray, *, gaps: np.ndarray | None = None) -> SmoothingResult:
        """`gaps[i]` marks row i as missing; §7 requires Viterbi still run, using the
        transition prior alone for those rows."""
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Post-training probability calibration (§8): isotonic for both layers.

    Layer 2 calibrates per class; Layer 1 calibrates a single probability.
    """

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> None: ...
    def transform(self, probabilities: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Turns validated raw signal into the ordered feature vector for one window.

    Implementations must emit values in feature-dictionary order and use each
    feature's declared missing policy — never a silent impute (§1).
    """

    def extract(self, window: object) -> np.ndarray: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...


@runtime_checkable
class BaselineProvider(Protocol):
    """Personal baselines and the cold-start ramp (§6).

    On DREAMT this is necessarily a within-night provider (one night per subject), so
    the reference comes from the PREP period. A cross-night provider only becomes
    testable with pilot data.
    """

    def reference(self, subject_id: str, feature: str, cycle_phase: str | None) -> float | None: ...

    def confidence(self, subject_id: str) -> float:
        """0.0 on night 1, ramping to 1.0 by night 14 (§6)."""
        ...
