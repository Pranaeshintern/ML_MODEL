"""Data records that flow through the pipeline.

    64 Hz samples  ->  Epoch (30 s)  ->  Row (15 min)  ->  StageEstimate  ->  SessionRecord
                                    \\-> StageLabel / OnsetLabel  (training only)

`Row` is the unit of inference and the interface boundary described in §1: a 15-min
row with ~32 aggregated features. Everything upstream of it is our ETL; everything
downstream is the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .errors import SchemaError
from .featuredict import FeatureDictionary
from .types import Stage

__all__ = ["Epoch", "Row", "StageLabel", "OnsetLabel", "LabelledRow"]


@dataclass(slots=True)
class Epoch:
    """30-second feature record — the intermediate between raw 64 Hz and a Row.

    Kept as a distinct stage because PSG labels are natively 30 s, so this is the
    finest resolution at which our labels are honest.
    """

    subject_id: str
    start: datetime
    features: dict[str, float]
    psg_stage: str | None = None
    valid: bool = True
    quality: float = 1.0

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=30)


@dataclass(slots=True)
class Row:
    """A 15-min aggregated row — the unit of inference (§1).

    `values` is ordered per the feature dictionary. Order is validated on construction
    because a silent misalignment produces a model that trains fine and is wrong.

    NaNs are legal and meaningful: tree models consume them natively and the doc
    requires missing-data tolerance (§1). Do not impute here.
    """

    subject_id: str
    start: datetime
    values: np.ndarray
    feature_version: int
    n_valid_epochs: int = 30
    quality: float = 1.0

    def validate(self, fd: FeatureDictionary, *, scope: str = "shared") -> None:
        expected = len(fd.for_scope(scope))  # type: ignore[arg-type]
        if self.values.ndim != 1 or self.values.shape[0] != expected:
            raise SchemaError(
                f"row {self.subject_id}@{self.start.isoformat()}: expected {expected} "
                f"features for scope {scope!r}, got shape {self.values.shape}"
            )
        if self.values.dtype != np.float32 and self.values.dtype != np.float64:
            raise SchemaError(f"row values must be float32/float64, got {self.values.dtype}")
        if self.feature_version != fd.version:
            raise SchemaError(
                f"row was built with feature dictionary v{self.feature_version} but "
                f"v{fd.version} is loaded — rebuild the rows or pin the spec"
            )

    def get(self, name: str, fd: FeatureDictionary) -> float:
        return float(self.values[fd.index_of(name)])

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=15)

    @property
    def is_complete(self) -> bool:
        return self.n_valid_epochs == 30


@dataclass(slots=True)
class StageLabel:
    """Ground-truth stage for one 15-min row.

    Carries BOTH representations deliberately. The hard-label-vs-proportion decision
    is still open (see `docs/decisions.md` D-002): majority voting discards 29 of 30
    epochs, and the user-facing metric in §9 is time-in-stage error, which is a
    proportion metric. Keeping `distribution` costs nothing and defers the choice.
    """

    majority: Stage
    distribution: np.ndarray  # shape (4,), sums to 1.0 over valid epochs
    n_valid_epochs: int
    contains_pre_sleep: bool = False

    def __post_init__(self) -> None:
        if self.distribution.shape != (Stage.n_classes(),):
            raise SchemaError(f"stage distribution must be shape (4,), got {self.distribution.shape}")
        total = float(self.distribution.sum())
        if self.n_valid_epochs > 0 and not np.isclose(total, 1.0, atol=1e-6):
            raise SchemaError(f"stage distribution must sum to 1.0, got {total:.6f}")

    @property
    def purity(self) -> float:
        """Fraction of valid epochs in the majority stage.

        The Phase-0 audit reports the distribution of this value. It quantifies exactly
        how much information hard labelling throws away, and it is the evidence base
        for D-002.
        """
        return float(self.distribution.max())

    @property
    def is_mixed(self) -> bool:
        return self.purity < 0.8


@dataclass(slots=True)
class OnsetLabel:
    """Ground truth for Layer 1 on one row.

    On DREAMT this is PSG-derived: the boundary between the trailing PREP/WAKE run and
    the first sustained sleep epoch. That is stronger ground truth than the in-app
    button §5 plans for — but there are only ~80 such transitions in the whole dataset,
    one per subject. See `docs/decisions.md` D-003.
    """

    is_onset_row: bool
    seconds_to_onset: float | None = None
    source: str = "psg"


@dataclass(slots=True)
class LabelledRow:
    """A Row paired with its supervision. Training only; never crosses the runtime."""

    row: Row
    stage: StageLabel | None = None
    onset: OnsetLabel | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def subject_id(self) -> str:
        return self.row.subject_id
