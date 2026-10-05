"""The shipped artifacts, behind the runtime's protocols.

`SleepPipeline` is written against `interfaces`, which is what lets stubs drive the
tests. This module is the other side of those seams: one adapter per protocol, each
holding a loaded artifact and doing the per-call shaping that artifact needs.

It exists because the shaping is not derivable from the row, and three of the four are
easy to get silently wrong:

  layer 1  131 inputs, not 26 — backward lags, first differences and a running still
           counter, assembled ACROSS CALLS from rows already seen. A wrong width
           raises; the right width in the wrong ORDER returns a confident wrong
           probability, and the state machine gates on absolute values (0.5 / 0.7).
  layer 2  column 0 of the returned (T, 4) is the Wake logit, and the artifact is
           `no_wake: true` — that logit was never supervised, so it is not P(wake) and
           must not be read as one. `merge_hypnogram` drops it; the wake channel owns
           the sleep/wake call.
  layer W  the band is a NIGHT-level vector built from the epoch features and P(wake),
           with each activity quantile divided by that night's own median so two
           devices with different count scales land in the same band. Nothing in the
           epoch array says so, and getting it wrong moves the band without erroring.

Inputs are selected from the feature dictionary BY NAME, so the caller builds one row
vector in dictionary order and never maps columns itself. A model needing something the
dictionary cannot supply raises `FeatureContractError` at construction, not at the first
row of a night.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.accel_epochs import context_matrix
from ..featuredict import FeatureDictionary
from ..interfaces import OnsetContext
from ..models.loader import (FeatureContractError, LoadedModel, LoadedTreeModel,
                             load_onset_model, load_stage_model, predict_stages,
                             select_features)
from ..models.waso import WasoEstimate, load_waso_model, night_features
from ..schema import Row

__all__ = ["OnsetAdapter", "StageAdapter", "WakeAdapter", "BandAdapter",
           "ShippedModels", "load_shipped"]


class OnsetAdapter:
    """Layer 1. Streams one row at a time and keeps its own lag history.

    `sustained_still_rows` is taken from the state machine's context rather than
    recomputed here. It is the same quantity the training matrix carries — consecutive
    rows with `still_fraction >= 0.9`, reset by a moving or missing row — and the
    pipeline updates it before this is called, so the value already includes the current
    row, exactly as it did at fit time. Two copies of that counter is how they drift.
    """

    version = "layer1/xgb"

    def __init__(self, model: LoadedTreeModel, dict_names: Sequence[str]):
        self.model = model
        self.dict_names = list(dict_names)
        self.base = list(model.meta["base_features"])
        self.lags = int(model.meta.get("lags", 2))
        # Fail here, not on the first row of a real night.
        select_features(np.zeros(len(self.dict_names)), self.dict_names, self.base)
        expected = len(self.base) * (1 + 2 * self.lags) + 1
        if len(model.feature_order) != expected:
            raise FeatureContractError(
                f"{len(self.base)} base features and {self.lags} lags build "
                f"{expected} inputs, but the artifact declares "
                f"{len(model.feature_order)} — the metadata and this construction "
                "disagree, which would misalign every column")
        self._history: list[np.ndarray] = []

    def reset(self) -> None:
        """Start a new night. Lag history must not cross sessions — at fit time lags
        never crossed a sequence boundary, they were NaN there."""
        self._history = []

    def predict_proba(self, row: Row, context: OnsetContext) -> float:
        v = select_features(row.values, self.dict_names, self.base)
        self._history.append(v)
        blocks = [v]
        for lag in range(1, self.lags + 1):
            # NaN for history that does not exist yet. The trees split on missingness
            # directly, so this is the signal the model was fitted with — imputing zero
            # here would tell it the sensor read exactly zero.
            prev = (self._history[-1 - lag] if len(self._history) > lag
                    else np.full(v.shape, np.nan))
            blocks += [prev, v - prev]
        blocks.append(np.array([float(context.sustained_still_rows)]))
        x = np.concatenate(blocks).reshape(1, -1)
        return float(self.model.predict_proba(x)[0])

    def fit(self, data) -> None:                       # pragma: no cover - serving only
        raise NotImplementedError("training lives in pipelines/10_train_onset.py")


class StageAdapter:
    """Layer 2. Re-decodes the whole night on every call, because the pipeline does.

    Returns four columns because that is the artifact's output width, but column 0 is
    unsupervised under `no_wake` and carries no meaning. Callers other than
    `merge_hypnogram` must not read it as P(wake).
    """

    version = "layer2/tcn"

    def __init__(self, model: LoadedModel, dict_names: Sequence[str]):
        self.model = model
        self.dict_names = list(dict_names)
        select_features(np.zeros(len(self.dict_names)), self.dict_names,
                        model.feature_order)
        self.n_classes = len(model.meta.get("stage_order", ("wake", "light", "deep", "rem")))

    def predict_proba(self, rows: list[Row]) -> np.ndarray:
        x = np.vstack([select_features(r.values, self.dict_names,
                                      self.model.feature_order) for r in rows])
        return predict_stages(self.model, x)

    def fit(self, data) -> None:                       # pragma: no cover - serving only
        raise NotImplementedError("training lives in pipelines/11_train_stage.py")


class WakeAdapter:
    """Layer W's epoch half: P(wake) per 30 seconds, from accelerometry alone.

    The scorer is fitted on epoch features PLUS temporal context — rolling means over
    5 and 21 epochs and a self-normalised activity ratio — because a single still epoch
    looks identical asleep or lying awake. `context_matrix` builds those 26 inputs from
    the 13 stored per epoch; handing the scorer the 13 raw columns is a shape error, and
    building the context in a different order is not.
    """

    version = "waso/epoch"

    def __init__(self, bundle: dict):
        self.clf = bundle["epoch_scorer"]
        self.cols = list(bundle["meta"]["epoch_feature_order"])

    def predict_wake_proba(self, accel_epochs: np.ndarray) -> np.ndarray:
        ep = np.asarray(accel_epochs, dtype=np.float64)
        if ep.size == 0:
            return np.zeros(0)
        if ep.ndim != 2 or ep.shape[1] != len(self.cols):
            raise ValueError(f"expected (E, {len(self.cols)}) epoch features matching "
                             f"{self.cols}, got {ep.shape}")
        return self.clf.predict_proba(context_matrix(ep, self.cols))[:, 1]


class BandAdapter:
    """Layer W's night half: the WASO band, once, at finalisation.

    The band is what a user is shown. The minute figure the timeline implies is not
    defensible on its own — 12.3 min of error against a true mean of 21.4 on the healthy
    cohort, where predicting the median alone gives 12.7 — so it stays internal to the
    Sleep Score and this is what reaches a screen.

    The window is Layer 1's confirmed onset by construction: the pipeline's buffer
    begins at the confirming row, so every epoch handed here is already post-onset and
    no onset index is passed. The band heads were refitted against that same onset
    (`onset_source: layer1` in the artifact), which matters because the two onset
    definitions disagree enough to move the reported band on 15.9% of nights.

    Returns None when the night cannot be scored — coverage below 0.5 or under an hour
    of post-onset data. A withheld band is a reportable state; an invented one is not.
    """

    version = "waso/band"

    def __init__(self, bundle: dict):
        self.band = bundle["band_model"]
        meta = bundle["meta"]
        self.cols = list(meta["epoch_feature_order"])
        self.normalised = bool(meta.get("normalised", True))

    def estimate_band(self, epochs: np.ndarray, pwake: np.ndarray,
                      coverage: float = 1.0) -> WasoEstimate | None:
        ep = np.asarray(epochs, dtype=np.float64)
        pw = np.asarray(pwake, dtype=np.float64).ravel()
        if ep.size == 0 or pw.size == 0:
            return None
        if ep.ndim != 2 or ep.shape[1] != len(self.cols):
            raise ValueError(f"expected (E, {len(self.cols)}) epoch features matching "
                             f"{self.cols}, got {ep.shape}")
        n = min(len(ep), len(pw))
        feats = night_features(ep[:n], self.cols, pw[:n], normalised=self.normalised)
        return self.band.estimate(feats, n, coverage=float(coverage))


@dataclass(frozen=True)
class ShippedModels:
    """The four adapters, ready to pass to `SleepPipeline`."""

    onset: OnsetAdapter
    stage: StageAdapter
    wake: WakeAdapter
    band: BandAdapter


def load_shipped(fd: FeatureDictionary, root: Path | str = "artifacts",
                 *, scope: str = "shared") -> ShippedModels:
    """Load all three artifacts and bind them to the feature dictionary.

    One call, so a caller cannot load two of the three and silently serve a night
    without the wake channel — which under-reports wake by roughly 12 minutes.
    """
    root = Path(root)
    dict_names = [f.name for f in fd.for_scope(scope)]  # type: ignore[arg-type]
    l1 = load_onset_model(root / "layer1/model.pt")
    if not isinstance(l1, LoadedTreeModel):
        raise TypeError("the shipped layer 1 is a gradient-boosted bundle; found "
                        f"{type(l1).__name__} — check artifacts/layer1/model.joblib")
    l2 = load_stage_model(root / "layer2/model.pt")
    lw = load_waso_model(str(root / "waso/model.joblib"))
    return ShippedModels(onset=OnsetAdapter(l1, dict_names),
                         stage=StageAdapter(l2, dict_names),
                         wake=WakeAdapter(lw), band=BandAdapter(lw))
