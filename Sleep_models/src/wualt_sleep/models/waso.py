"""WASO from accelerometry alone — reported as a band, not a number.

WHY A BAND

Measured on 99 DREAMT nights: every minute-valued WASO estimator available lands within
7 minutes of simply predicting the population mean (constant MAE 41.8; best estimator
34.9), and correlation with truth tops out near r = 0.43. A number carrying 50% relative
error is worse than no number, because WASO is a quantity the user can partly check
against their own memory of the night.

Rank information survives where absolute information does not — Spearman reaches 0.55,
which is enough to order nights and therefore enough to say "quiet night" versus "you
were awake a fair amount". So this model predicts the BAND and never prints a
single-minute figure.

The four bands tile the whole range continuously, with boundaries at 20 / 45 / 90
minutes and no gaps:

    MINIMAL  up to 20 min  ->  SOME  20-45  ->  NOTABLE  45-90  ->  HIGH  over 90

Those are the ordinal model's actual decision thresholds, not a description of observed
values, and the range shown to a user is the band's own interval.

STRUCTURE

  epoch scorer   30-s accelerometer features -> P(wake) per epoch
  night model    ordinal cumulative-link over night-level summaries -> band

The ordinal head is three binary classifiers, P(WASO > 20), P(WASO > 45), P(WASO > 90),
made monotone by a running minimum. Ordinal rather than 4-way softmax because the
classes are ordered and a confusion between MINIMAL and HIGH is not the same mistake as
one between MINIMAL and SOME — the loss should know that.

DEVICE INVARIANCE

Two corpora with very different WASO distributions (DREAMT 71.7 min, BIDSleep 21.2 min)
and different hardware. Any feature that identifies the corpus lets the model skip the
signal and apply that corpus's prior — the same shortcut the 15-minute staging model was
found to be taking through `has_hrv`. Night features are therefore SELF-NORMALISED where
possible: activity is expressed relative to the night's own quantiles, so the device's
absolute scale divides out. `feature_names(normalised=False)` exposes the raw variant so
the difference can be measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["WasoBand", "WasoEstimate", "BAND_EDGES", "MIN_COVERAGE", "MIN_EPOCHS",
           "night_feature_names", "night_features", "WasoModel", "load_waso_model",
           "estimate_waso", "longest_covered_span"]

EPOCH_MIN = 0.5

# Boundaries in minutes. 30 min is the usual clinical threshold for disturbed sleep;
# these straddle it so the band the user is most likely to care about has an edge on
# either side of it rather than sitting in the middle of a bucket.
BAND_EDGES = (20.0, 45.0, 90.0)

# A night the accelerometer barely covered is a non-wear night, not a quiet one. Seven
# BIDSleep nights have motion overlapping under half the scored period, one as low as
# 0.8%; scored naively they look perfectly still and would be reported as the user's
# best night's sleep.
MIN_COVERAGE = 0.5
MIN_EPOCHS = 120                 # one hour of post-onset data


class WasoBand(Enum):
    """Ordered. `value` is the ordinal level, so comparisons are meaningful."""

    MINIMAL = 0
    SOME = 1
    NOTABLE = 2
    HIGH = 3

    @property
    def label(self) -> str:
        return {0: "minimal", 1: "some", 2: "notable", 3: "high"}[self.value]

    @property
    def message(self) -> str:
        """What the user is told. No minute figure — see the module docstring."""
        return {
            0: "You barely woke during the night.",
            1: "You woke a few times, which is normal.",
            2: "You were awake for a fair amount of the night.",
            3: "You spent a long stretch of the night awake.",
        }[self.value]

    @property
    def bounds(self) -> tuple[float, float]:
        """The band's WASO range in minutes, as (lower, upper].

        These ARE the decision boundaries — the ordinal heads are fitted on
        P(WASO > 20), P(WASO > 45), P(WASO > 90) — not a summary of observed values.
        The four bands tile [0, inf) with no gap and no overlap.
        """
        lo = BAND_EDGES[self.value - 1] if self.value > 0 else 0.0
        hi = BAND_EDGES[self.value] if self.value < len(BAND_EDGES) else float("inf")
        return lo, hi

    @property
    def range_label(self) -> str:
        """Human-readable bound. Intervals are (lower, upper], matching `of()` — a
        night at exactly 20 minutes is MINIMAL, not SOME."""
        lo, hi = self.bounds
        if self.value == 0:
            return f"up to {hi:.0f} min"
        if hi == float("inf"):
            return f"over {lo:.0f} min"
        return f"{lo:.0f}-{hi:.0f} min"

    @classmethod
    def of(cls, waso_min: float) -> "WasoBand":
        """Bands are half-open as (edge, next]: a night at exactly 20 min is MINIMAL.

        `side="left"` is load-bearing, not a detail. The ordinal heads are fitted on
        `waso > edge`, so a night sitting exactly on a boundary must fall on the lower
        side in both places. With `side="right"` the label said SOME while the head
        said MINIMAL, and WASO is quantised to half-minutes, so exact boundary values
        do occur.
        """
        return cls(int(np.searchsorted(BAND_EDGES, waso_min, side="left")))

    @classmethod
    def ordered(cls) -> list["WasoBand"]:
        return [cls(i) for i in range(4)]


@dataclass(frozen=True, slots=True)
class WasoEstimate:
    """What the model returns.

    `range_min`/`range_max` are the BAND'S OWN BOUNDARIES — the thresholds the ordinal
    heads are fitted on. They are not a summary of observed values.

    An earlier version reported the interquartile spread of true WASO among training
    nights assigned to each band (6-16, 24-36, 50-72, 103-159 min). That was wrong to
    surface: it left visible gaps between consecutive bands, so a night the model placed
    in MINIMAL and one it placed in SOME appeared to be separated by 8 unexplained
    minutes, when in fact the bands are contiguous at 20 / 45 / 90. The IQR is still
    computed for analysis — see `WasoModel.observed_iqr` — but it does not define a band.
    """

    band: WasoBand
    band_probs: np.ndarray
    range_min: float
    range_max: float
    confidence: str
    n_epochs_scored: int

    @property
    def message(self) -> str:
        return self.band.message

    @property
    def approx_minutes(self) -> str:
        return f"{self.band.range_label} awake"

    def describe(self) -> str:
        return f"{self.message} ({self.approx_minutes}, confidence {self.confidence})"


# ------------------------------------------------------------------ features -
_ACT = ("act_mean", "act_p95", "jerk_rms", "mag_std")
_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
_PROB_CUTS = (0.3, 0.5, 0.7, 0.9)
_POSTURE_CUTS = (10.0, 20.0, 40.0)

# Per-epoch features averaged into the night vector. `zcr`, `bp_lo` and `bp_mid` were
# here and were removed: measured contribution was nil at the epoch level and NEGATIVE
# at the night level (exact 0.561 with them, 0.581 without). Named as one constant so
# the names and the vector cannot drift apart — they are built from this list twice.
_NIGHT_TAIL = ("coverage",)


def night_feature_names(normalised: bool = True) -> list[str]:
    names = ["n_epochs_min", "pwake_mean", "pwake_std", "pwake_sum_min"]
    names += [f"pwake_frac_gt{c}" for c in _PROB_CUTS]
    names += ["pwake_runs", "pwake_longest_run_min", "still_longest_run_min"]
    tag = "rel" if normalised else "abs"
    for f in _ACT:
        names += [f"{f}_{tag}_q{int(q*100)}" for q in _QUANTILES]
    names += ["still_frac_mean", "n_moves_mean", "n_moves_per_hr"]
    names += ["posture_delta_mean"] + [f"posture_gt{int(c)}_per_hr" for c in _POSTURE_CUTS]
    names += ["posture_range_q90"] + [f"{f}_mean" for f in _NIGHT_TAIL]
    return names


def _safe(fn, v: np.ndarray, *a, default: float = 0.0) -> float:
    """Aggregate ignoring NaN, returning `default` when nothing is finite.

    An all-NaN night is not an error to swallow silently — it means the accelerometer
    produced nothing over that window. `MIN_COVERAGE` is what refuses those nights; this
    only stops the aggregation itself from emitting warnings and NaNs on the way there.
    """
    v = v[np.isfinite(v)]
    return default if v.size == 0 else float(fn(v, *a))


def _runs(mask: np.ndarray) -> tuple[int, int]:
    """(number of runs, longest run) over a boolean mask."""
    if mask.size == 0:
        return 0, 0
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if starts.size == 0:
        return 0, 0
    return int(starts.size), int((ends - starts).max())


def night_features(ep: np.ndarray, cols: list[str], pwake: np.ndarray,
                   normalised: bool = True) -> np.ndarray:
    """Post-onset epoch features (n, F) + P(wake) (n,) -> one night-level vector.

    Self-normalisation divides each activity quantile by the night's own median, so a
    device with twice the counts produces the same feature. The median is floored to
    avoid dividing by a night that was almost perfectly still.
    """
    idx = {c: i for i, c in enumerate(cols)}
    n = len(pwake)
    out: list[float] = [n * EPOCH_MIN, float(pwake.mean()), float(pwake.std()),
                        float(pwake.sum()) * EPOCH_MIN]
    out += [float((pwake > c).mean()) for c in _PROB_CUTS]
    runs, longest = _runs(pwake > 0.5)
    out += [float(runs), float(longest) * EPOCH_MIN]

    still = ep[:, idx["still_frac"]]
    out.append(float(_runs(still > 0.9)[1]) * EPOCH_MIN)

    for f in _ACT:
        v = ep[:, idx[f]]
        v = v[np.isfinite(v)]
        if v.size == 0:
            out += [0.0] * len(_QUANTILES)
            continue
        q = np.quantile(v, _QUANTILES)
        if normalised:
            base = max(float(np.median(v)), 1e-4)
            q = q / base
        out += [float(x) for x in q]

    hours = max(n * EPOCH_MIN / 60.0, 1e-6)
    out.append(_safe(np.mean, still))
    moves = ep[:, idx["n_moves"]]
    out += [_safe(np.mean, moves), float(np.nansum(moves)) / hours]

    pd_ = ep[:, idx["posture_delta"]]
    out.append(_safe(np.mean, pd_))
    out += [float(np.nansum(pd_ > c)) / hours for c in _POSTURE_CUTS]
    out.append(_safe(np.quantile, ep[:, idx["posture_range"]], 0.9))
    for f in _NIGHT_TAIL:
        out.append(_safe(np.mean, ep[:, idx[f]]))
    return np.nan_to_num(np.array(out, dtype=np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------- model -
class WasoModel:
    """Ordinal cumulative-link band model over night-level accelerometer summaries.

    Not a neural network on purpose. 353 nights is a very small training set for a
    night-level target, and the honest comparison here is against a constant predictor
    that already achieves MAE 41.8 — capacity is not what is missing.
    """

    def __init__(self, seed: int = 0, decision: str = "median",
                 balanced: bool = True) -> None:
        """`decision` picks the band from the predicted ordinal distribution.

        "median"  the largest band whose lower edge the model puts >50% mass above —
                  the median of the predicted distribution. This is the natural readout
                  of a cumulative-link model and it respects the ordering.
        "argmax"  the single most probable band. Looks reasonable and is not: on a
                  skewed target it collapses onto the mode, which measured here means
                  only 16% of genuinely-high-WASO nights were ever called high. A user
                  who was awake two hours being told "you woke a few times" is the one
                  failure this model exists to avoid.
        """
        self.seed = seed
        self.decision = decision
        self.balanced = balanced
        self.heads: list = []
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        # Diagnostic only; a band is defined by BAND_EDGES, not by this.
        self.observed_iqr: dict[int, tuple[float, float]] = {}
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, waso_min: np.ndarray,
            feature_names: list[str] | None = None) -> "WasoModel":
        from sklearn.ensemble import GradientBoostingClassifier

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        Z = (X - self.mu) / self.sd

        self.heads = []
        for edge in BAND_EDGES:
            y = (waso_min > edge).astype(int)
            if y.min() == y.max():                  # degenerate fold — constant head
                self.heads.append(float(y.mean()))
                continue
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                random_state=self.seed)
            # The top edge has ~9% positives. Without reweighting that head learns to
            # say "no" and the HIGH band becomes unreachable.
            sw = None
            if self.balanced:
                freq = np.bincount(y, minlength=2).astype(float)
                sw = (len(y) / (2.0 * np.maximum(freq, 1)))[y]
            self.heads.append(clf.fit(Z, y, sample_weight=sw))

        # Interquartile spread of TRUE WASO among training nights assigned to each band.
        # DIAGNOSTIC ONLY — it is not what a band means and must not be shown to a user.
        # Surfacing it produced apparent gaps between contiguous bands (6-16 then 24-36),
        # which is an artefact of quartiles, not a property of the model. A band is
        # defined by BAND_EDGES; this says how the truth happened to distribute inside
        # one, which is useful for spotting a band that is drifting off its interval.
        pred = self.predict_bands(X)
        for b in range(4):
            sel = waso_min[pred == b]
            self.observed_iqr[b] = ((float(np.percentile(sel, 25)),
                                     float(np.percentile(sel, 75)))
                                    if sel.size >= 5 else WasoBand(b).bounds)
        return self

    def cumulative(self, X: np.ndarray) -> np.ndarray:
        """(n, 3) monotone P(WASO > edge) for each band edge."""
        Z = (X - self.mu) / self.sd
        cum = np.empty((len(Z), len(BAND_EDGES)))
        for j, h in enumerate(self.heads):
            cum[:, j] = h if isinstance(h, float) else h.predict_proba(Z)[:, 1]
        # A cumulative link must be non-increasing: P(>90) cannot exceed P(>45).
        # Independently fitted heads can violate that, so enforce it rather than hope.
        return np.minimum.accumulate(cum, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, 4) band probabilities."""
        cum = self.cumulative(X)
        p = np.empty((len(cum), 4))
        p[:, 0] = 1.0 - cum[:, 0]
        for j in range(1, len(BAND_EDGES)):
            p[:, j] = cum[:, j - 1] - cum[:, j]
        p[:, -1] = cum[:, -1]
        return np.clip(p, 1e-9, 1.0) / np.clip(p, 1e-9, 1.0).sum(1, keepdims=True)

    def predict_bands(self, X: np.ndarray) -> np.ndarray:
        if self.decision == "argmax":
            return self.predict_proba(X).argmax(1)
        return (self.cumulative(X) >= 0.5).sum(1)

    def estimate(self, x: np.ndarray, n_epochs: int,
                 coverage: float = 1.0) -> WasoEstimate | None:
        """None when the accelerometer did not cover enough of the night.

        Returning None rather than a band is the same call as `sleep_score` returning
        None: a ring that was not worn has no WASO, and a plausible-looking band would
        reach the user with nobody downstream aware it was invented.
        """
        if coverage < MIN_COVERAGE or n_epochs < MIN_EPOCHS:
            return None
        p = self.predict_proba(x.reshape(1, -1))[0]
        b = int(self.predict_bands(x.reshape(1, -1))[0])
        lo, hi = WasoBand(b).bounds
        conf = "high" if p[b] >= 0.60 else "medium" if p[b] >= 0.40 else "low"
        return WasoEstimate(band=WasoBand(b), band_probs=p, range_min=lo, range_max=hi,
                            confidence=conf, n_epochs_scored=n_epochs)


# ------------------------------------------------------------------ runtime -
def load_waso_model(path: str = "artifacts/waso/model.joblib") -> dict:
    """Load the two-stage bundle plus its metadata.

    Same contract as `models/loader.py`: the weights alone are not a model. The
    `.meta.json` beside them fixes the epoch feature order, the context transform and
    the band edges, and loading refuses without it — present the features in a
    different order and the model returns a confident wrong band rather than an error.
    """
    import json
    from pathlib import Path

    import joblib

    p = Path(path)
    meta_path = p.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing — the model is unusable without the feature order "
            "and band edges")
    bundle = joblib.load(p)
    bundle["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return bundle


def longest_covered_span(t: np.ndarray, max_gap_s: float = 300.0
                         ) -> tuple[float, float]:
    """Largest stretch of `t` with no gap longer than `max_gap_s`.

    A recording file is not a night. BIDSleep's Bidslab08/1 spans 31.3 hours around a
    single 16.4-hour hole — the device off the wrist between two wearing periods — while
    the scored night is 6.9 hours. Spanning the whole file drags mean coverage to 0.475
    and the night is refused for a reason that is an artefact of framing, not of data.
    A live device gets its window from the session; an offline caller gets this.
    """
    t = np.sort(np.asarray(t, dtype=np.float64).ravel())
    if t.size < 2:
        return (float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 0.0)
    brk = np.flatnonzero(np.diff(t) > max_gap_s)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [t.size - 1]])
    best = int(np.argmax(t[ends] - t[starts]))
    return float(t[starts[best]]), float(t[ends[best]])


def estimate_waso(bundle: dict, t: np.ndarray, x: np.ndarray, y: np.ndarray,
                  z: np.ndarray, *, normalised: bool | None = None,
                  t_start: float | None = None, t_end: float | None = None,
                  onset_epoch: int | None = None) -> WasoEstimate | None:
    """One night of raw tri-axial accelerometry -> a WASO band.

        est = estimate_waso(load_waso_model(), t, x, y, z)
        print(est.describe() if est else "not enough motion data")

    `t` is seconds (any epoch), `x/y/z` are in g. Sample rate is inferred and the
    signal is resampled onto the model's grid, so a 50 Hz and a 64 Hz device produce
    the same features. Pass `t_start`/`t_end` to bound the night explicitly; without
    them the longest continuously-worn stretch is used.

    Returns None when the night cannot be scored — too little accelerometer coverage,
    too short, or no sleep ever detected. A band is withheld rather than guessed.
    """
    from ..data.accel_epochs import (ACCEL_EPOCH_FEATURES, context_matrix,
                                     context_names, epoch_features, resample_uniform)

    meta = bundle.get("meta", {})
    cols = list(ACCEL_EPOCH_FEATURES)
    if list(meta.get("epoch_feature_order") or cols) != cols:
        raise ValueError(
            "epoch feature order in the metadata does not match the installed "
            "extractor — this model predates the current feature dictionary")
    # The context transform is just as contractual as the raw dictionary: dropping one
    # rolling window silently shifts every column the scorer reads, and the weights
    # would still load. Checked here rather than trusted.
    if list(meta.get("context_feature_order") or context_names(cols)) != context_names(cols):
        raise ValueError(
            "context feature order in the metadata does not match the installed "
            "transform — retrain, or check out the code this model was built with")
    if list(meta.get("night_feature_order") or night_feature_names(True)) != \
            night_feature_names(bool(meta.get("normalised", True))):
        raise ValueError(
            "night feature order in the metadata does not match the installed "
            "feature set — retrain")
    if normalised is None:
        normalised = bool(meta.get("normalised", True))

    t = np.asarray(t, dtype=np.float64).ravel()
    if t.size < 2:
        return None
    # Bound the night. Callers that know the session window pass it; otherwise take the
    # longest continuously-worn stretch rather than the whole file.
    if t_start is None or t_end is None:
        auto_start, auto_end = longest_covered_span(t)
        t_start = auto_start if t_start is None else t_start
        t_end = auto_end if t_end is None else t_end
    gx, gy, gz, cov = resample_uniform(t, np.asarray(x, float).ravel(),
                                       np.asarray(y, float).ravel(),
                                       np.asarray(z, float).ravel(),
                                       float(t_start), float(t_end))
    ep = epoch_features(gx, gy, gz, cov)
    if ep.shape[0] < MIN_EPOCHS:
        return None

    pwake = bundle["epoch_scorer"].predict_proba(context_matrix(ep, cols))[:, 1]

    # ONE ONSET PER SYSTEM. Pass `onset_epoch` and this uses Layer 1's confirmed onset,
    # so the record's onset_time and the window WASO is measured over describe the same
    # night. THE SHIPPED BAND HEADS WERE FITTED THAT WAY (`onset_source: layer1` in the
    # artifact), so passing it is the matched path and the motion fallback below is the
    # mismatched one -- it remains only for callers with no Layer 1 at all, such as
    # offline scoring of a corpus whose onsets were never predicted.
    #
    # SENSITIVITY, measured over 346 nights: moving the window from the motion-derived
    # onset to the TRUE onset changes the reported band on 15.9% of nights. The two
    # onsets differ by a median 6.5 min and a p90 of 42.8 min, so the window is not a
    # detail; whichever onset the heads were fitted on is the one they must be served.
    # The runtime never comes through here -- its buffer starts at the confirming row, so
    # every epoch it holds is already post-onset (see `runtime.adapters.BandAdapter`).
    if onset_epoch is not None:
        on = int(np.clip(onset_epoch, 0, max(len(pwake) - MIN_EPOCHS, 0)))
    else:
        on, run = None, 0
        for i, v in enumerate(pwake < 0.5):
            run = run + 1 if v else 0
            if run >= 6:
                on = i - 5
                break
        if on is None:
            return None                 # never detected sleep — no WASO to report

    coverage = _safe(np.mean, ep[on:, cols.index("coverage")])
    feats = night_features(ep[on:], cols, pwake[on:], normalised=normalised)
    return bundle["band_model"].estimate(feats, len(pwake) - on, coverage=coverage)
