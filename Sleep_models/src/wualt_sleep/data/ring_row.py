"""Ring data -> one 15-minute `Row`, plus the (30, 13) accelerometer block for that row.

THIS IS THE RING-SIDE REFERENCE FOR THE NATIVE PORT. The app builds the same numbers in
Swift/Kotlin and checks them against this module (see `examples/ring_row_vectors.json`).

WHY IT EXISTS
The training rows were built by the corpus adapters (`data/dreamt.py`, `data/bidsleep.py`),
which are not in this package. Without a ring-side builder the app had to leave 19 of the
32 row values as NaN. Every definition below is tagged in `STATUS`:

  confirmed    stated in this package (features.yaml, hrv.py, motion.py, accel_epochs.py)
               or forced by the shipped artifacts' training statistics
  provisional  the best reconstruction available. Replace with the training definition
               when the corpus adapters are published; the row layout does not change.

WHICH MODEL READS WHAT (from the artifacts' .meta.json)
  Layer 1 (onset)    26 base features: no time features at all
  Layer 2 (staging)  28 features: no temperature features at all
  The four `layer1`-scope dictionary entries are NOT part of the row the runtime takes
  (`push_row` validates the 32 `shared` features); `sustained_still_rows` is kept by the
  pipeline itself.

INPUT UNITS
  NN intervals  seconds, EVERY interval the ring reported for the burst, in order, with
                nothing removed. `clean_nn` does the rejecting and needs the gaps.
  accelerometer g (divide mg by 1000), timestamps in seconds on the same clock as
                `RowInput.start.timestamp()`. Any native rate; it is resampled to 50 Hz.
                The models were trained on 50-100 Hz and 64 Hz devices. A native rate
                below ~25 Hz cannot reproduce the 3-10 Hz band and has not been
                validated (see INTEGRATION.md).
  temperature   degC, one value per reading = mean of the usable skin sensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .accel_epochs import EPOCH_S, epoch_features, resample_uniform, time_coverage
from .hrv import MIN_BEATS, NNSeries, beat_coverage, clean_nn, hrv_features
from .motion import motion_features_on_grid

__all__ = ["ROW_FEATURES", "STATUS", "USED_BY", "MISSING_FILL", "ROW_S", "BURST_S",
           "REFERENCE_ROWS", "BASELINE_RAMP_NIGHTS", "RowInput", "NightState",
           "hr_features", "temp_features", "time_features", "baseline_confidence",
           "build_row_values", "accel_epochs_for_row", "check_against"]

ROW_S = 900.0
BURST_S = 90.0                # features.yaml `ppg_burst_seconds`; checked by check_against
REFERENCE_ROWS = 2            # provisional within-night reference: the session's first rows
BASELINE_RAMP_NIGHTS = 13     # (n - 1) / 13 is 0 on night 1 and 1.0 on night 14
MIN_TEMP_READINGS_SLOPE = 3

#: The 32 `shared` features, in dictionary order. `check_against` fails if
#: config/features.yaml ever disagrees.
ROW_FEATURES = (
    "hr_mean", "hr_min", "hr_std", "hr_slope", "hr_delta_vs_rhr",
    "rmssd", "sdnn", "pnn50", "poincare_sd1", "poincare_sd2", "sd1_sd2_ratio",
    "hrv_sample_entropy",
    "activity_count_mean", "activity_count_max", "still_fraction", "acc_magnitude_std",
    "posture_change_count", "movement_burst_count",
    "temp_mean", "temp_slope", "temp_std", "temp_delta_vs_base",
    "clock_hour_sin", "clock_hour_cos", "minutes_in_record", "minutes_since_onset",
    "cycle_phase_sin", "cycle_phase_cos",
    "sqi_ppg", "hrv_artifact_rate", "sqi_acc",
    "baseline_confidence",
)

_P, _C = "provisional", "confirmed"
STATUS = {
    "hr_mean": _C,            # 90 s burst only; mean of per-beat HR (60/NN), <0.3 bpm from 60/mean
    "hr_min": _P,             # per-beat minimum fits the training stats; not stated
    "hr_std": _C,             # per-beat HR; ddof changes it by 0.6% at ~90 beats
    "hr_slope": _P,           # WITHIN the burst is forced by the training sd (275 bpm/hr);
                              # least squares is the reconstruction
    "hr_delta_vs_rhr": _P,    # reference definition is in the missing adapters
    **{n: _C for n in ("rmssd", "sdnn", "pnn50", "poincare_sd1", "poincare_sd2",
                       "sd1_sd2_ratio", "hrv_sample_entropy")},
    **{n: _C for n in ("activity_count_mean", "activity_count_max", "still_fraction",
                       "acc_magnitude_std", "posture_change_count",
                       "movement_burst_count")},
    "temp_mean": _P, "temp_slope": _P, "temp_std": _P, "temp_delta_vs_base": _P,
    "clock_hour_sin": _P, "clock_hour_cos": _P,
    "minutes_in_record": _P,
    "minutes_since_onset": _P,
    "cycle_phase_sin": _C, "cycle_phase_cos": _C,   # formula is in features.yaml's note
    "sqi_ppg": _C, "hrv_artifact_rate": _C, "sqi_acc": _C,
    "baseline_confidence": _P,
}

_L1_EXCLUDED = {"clock_hour_sin", "clock_hour_cos", "minutes_in_record",
                "minutes_since_onset", "cycle_phase_sin", "cycle_phase_cos"}
_L2_EXCLUDED = {"temp_mean", "temp_slope", "temp_std", "temp_delta_vs_base"}
USED_BY = {n: tuple(layer for layer, excl in (("layer1", _L1_EXCLUDED),
                                              ("layer2", _L2_EXCLUDED)) if n not in excl)
           for n in ROW_FEATURES}

#: features.yaml `missing` policies that are numbers. Everything else stays NaN.
MISSING_FILL = {
    "posture_change_count": 0.0,
    "movement_burst_count": 0.0,
    "sqi_ppg": 0.0,
    "hrv_artifact_rate": 1.0,
    "sqi_acc": 0.0,
    "baseline_confidence": 0.0,
}


@dataclass(slots=True)
class RowInput:
    """Everything the ring delivered for one 15-minute row."""

    start: datetime                       # row start, timezone-aware LOCAL time
    nn_s: np.ndarray                      # every reported interval in the burst, seconds
    accel_t: np.ndarray                   # seconds
    accel_g: np.ndarray                   # (n, 3), g
    temp_t: np.ndarray                    # seconds
    temp_c: np.ndarray                    # degC, mean of usable sensors per reading
    burst_s: float = BURST_S


@dataclass(slots=True)
class NightState:
    """What a row needs that is not in the row. One per session; the app owns it.

    `onset_at` is `OnsetEvent.confirmed_at`, set by the app when the pipeline emits the
    event. Rows that start before it carry NaN for the onset-relative features.
    """

    session_start: datetime
    nights_before: int = 0                # nights this user already recorded (0 on night 1)
    onset_at: datetime | None = None
    hr_reference_bpm: float | None = None     # personal baseline; None -> within-night
    temp_reference_c: float | None = None     # personal baseline; None -> within-night
    _hr_ref: list[float] = field(default_factory=list)
    _temp_ref: list[float] = field(default_factory=list)

    def _observe(self, hr_mean: float, temp_mean: float) -> None:
        if np.isfinite(hr_mean) and len(self._hr_ref) < REFERENCE_ROWS:
            self._hr_ref.append(float(hr_mean))
        if np.isfinite(temp_mean) and len(self._temp_ref) < REFERENCE_ROWS:
            self._temp_ref.append(float(temp_mean))

    @property
    def hr_reference(self) -> float:
        if self.hr_reference_bpm is not None:
            return float(self.hr_reference_bpm)
        return float(np.mean(self._hr_ref)) if self._hr_ref else float("nan")

    @property
    def temp_reference(self) -> float:
        if self.temp_reference_c is not None:
            return float(self.temp_reference_c)
        return float(np.mean(self._temp_ref)) if self._temp_ref else float("nan")


# ------------------------------------------------------------------ cardiac ---

def hr_features(nn: NNSeries, *, min_beats: int = MIN_BEATS) -> dict[str, float]:
    """hr_mean, hr_min, hr_std, hr_slope from the cleaned beats of ONE burst.

    Per-beat HR is 60 / NN over the beats `clean_nn` accepted. The slope is a least-squares
    fit of per-beat HR against beat time, in bpm per hour (features.yaml unit). Below
    `min_beats` valid beats every value is NaN, the same floor the HRV features use.
    """
    out = dict.fromkeys(("hr_mean", "hr_min", "hr_std", "hr_slope"), float("nan"))
    if nn.n_valid < min_beats:
        return out
    t_beat = np.cumsum(nn.intervals)[nn.valid]          # seconds from burst start
    hr = 60.0 / nn.valid_intervals()
    out["hr_mean"] = float(np.mean(hr))
    out["hr_min"] = float(np.min(hr))
    out["hr_std"] = float(np.std(hr, ddof=1))
    out["hr_slope"] = float(np.polyfit(t_beat, hr, 1)[0] * 3600.0)
    return out


# -------------------------------------------------------------- temperature ---

def temp_features(t: np.ndarray, temp_c: np.ndarray, t0: float, t1: float
                  ) -> dict[str, float]:
    """temp_mean / temp_std / temp_slope over the readings inside [t0, t1).

    Layer 1 only. The training sensor (one wrist sensor) sampled far faster than the ring's
    one reading a minute, so temp_std in particular will not match training exactly.
    """
    out = dict.fromkeys(("temp_mean", "temp_slope", "temp_std"), float("nan"))
    t = np.asarray(t, dtype=np.float64).ravel()
    v = np.asarray(temp_c, dtype=np.float64).ravel()
    n = min(t.size, v.size)
    t, v = t[:n], v[:n]
    keep = np.isfinite(t) & np.isfinite(v) & (t >= t0) & (t < t1)
    t, v = t[keep], v[keep]
    if v.size == 0:
        return out
    out["temp_mean"] = float(np.mean(v))
    if v.size >= 2:
        out["temp_std"] = float(np.std(v, ddof=1))
    if v.size >= MIN_TEMP_READINGS_SLOPE and np.ptp(t) > 0:
        out["temp_slope"] = float(np.polyfit(t - t0, v, 1)[0] * 3600.0)
    return out


# --------------------------------------------------------------------- time ---

def time_features(start: datetime, night: NightState) -> dict[str, float]:
    """Clock, record and onset-relative features. Layer 2 only.

    clock: local fractional hour at the row START, sin/cos(2*pi*h/24).
    minutes_in_record: minutes from the session start to the row start. In training this
        was the lab recording start; the yaml calls it a protocol artifact.
    minutes_since_onset: minutes from `OnsetEvent.confirmed_at` to the row start; NaN for
        rows that start before it (including the confirming row itself).
    cycle phase: sin/cos(2*pi*(minutes_since_onset mod 90)/90), NaN with it.
    """
    h = start.hour + start.minute / 60.0 + start.second / 3600.0
    ang = 2.0 * math.pi * h / 24.0
    out = {
        "clock_hour_sin": math.sin(ang),
        "clock_hour_cos": math.cos(ang),
        "minutes_in_record": (start - night.session_start).total_seconds() / 60.0,
        "minutes_since_onset": float("nan"),
        "cycle_phase_sin": float("nan"),
        "cycle_phase_cos": float("nan"),
    }
    if night.onset_at is not None and start >= night.onset_at:
        m = (start - night.onset_at).total_seconds() / 60.0
        ph = 2.0 * math.pi * (m % 90.0) / 90.0
        out.update(minutes_since_onset=m, cycle_phase_sin=math.sin(ph),
                   cycle_phase_cos=math.cos(ph))
    return out


def baseline_confidence(nights_before: int) -> float:
    """0 on night 1, 1.0 from night 14: min(1, (n - 1) / 13) with n = nights_before + 1.

    PROVISIONAL, and read INTEGRATION.md before shipping it: the shipped models saw a mean
    of 0.12 (sd 0.13) in training, so 1.0 is far outside what Layer 2 was fitted on.
    """
    return float(min(1.0, max(0, int(nights_before)) / BASELINE_RAMP_NIGHTS))


# ---------------------------------------------------------------------- row ---

def build_row_values(inp: RowInput, night: NightState) -> np.ndarray:
    """One row's 32 values, in dictionary order, as float64. NaN is legal; do not impute.

    Updates `night` (the within-night references), so call it once per row, in order.
    """
    t0 = inp.start.timestamp()
    t1 = t0 + ROW_S
    v: dict[str, float] = dict.fromkeys(ROW_FEATURES, float("nan"))

    nn = clean_nn(np.asarray(inp.nn_s, dtype=np.float64))
    v.update(hr_features(nn))
    v.update(hrv_features(nn))
    v["sqi_ppg"] = beat_coverage(nn, inp.burst_s)
    v["hrv_artifact_rate"] = nn.artifact_rate

    t = np.asarray(inp.accel_t, dtype=np.float64).ravel()
    a = np.asarray(inp.accel_g, dtype=np.float64).reshape(-1, 3)
    n = min(t.size, len(a))
    t, a = t[:n], a[:n]
    ok = np.isfinite(t) & np.isfinite(a).all(axis=1)
    in_row = ok & (t >= t0) & (t < t1)
    if in_row.sum() >= 2:
        v.update(motion_features_on_grid(t[ok], a[ok, 0], a[ok, 1], a[ok, 2], t0, t1))
    v["sqi_acc"] = time_coverage(np.sort(t[ok]), t0, t1) if ok.any() else 0.0

    v.update(temp_features(inp.temp_t, inp.temp_c, t0, t1))

    night._observe(v["hr_mean"], v["temp_mean"])
    v["hr_delta_vs_rhr"] = v["hr_mean"] - night.hr_reference
    v["temp_delta_vs_base"] = v["temp_mean"] - night.temp_reference

    v.update(time_features(inp.start, night))
    v["baseline_confidence"] = baseline_confidence(night.nights_before)

    for name, fill in MISSING_FILL.items():
        if not np.isfinite(v[name]):
            v[name] = fill
    return np.array([v[name] for name in ROW_FEATURES], dtype=np.float64)


def accel_epochs_for_row(t: np.ndarray, accel_g: np.ndarray, start: datetime
                         ) -> np.ndarray | None:
    """The (30, 13) epoch block `push_row(..., accel_epochs=...)` takes, or None.

    Computed on the 50 Hz grid with ONE extra epoch of look-back that is then dropped,
    so the first epoch's `posture_delta` compares against the previous epoch exactly as it
    did when training computed epochs over the whole night. Returns None when the row has
    fewer than two real samples, which is the "no motion" case the pipeline expects.
    """
    t0 = start.timestamp()
    t = np.asarray(t, dtype=np.float64).ravel()
    a = np.asarray(accel_g, dtype=np.float64).reshape(-1, 3)
    n = min(t.size, len(a))
    t, a = t[:n], a[:n]
    ok = np.isfinite(t) & np.isfinite(a).all(axis=1)
    if ((t >= t0) & (t < t0 + ROW_S) & ok).sum() < 2:
        return None
    lo = t0 - EPOCH_S
    keep = ok & (t >= lo - 1.0) & (t < t0 + ROW_S + 1.0)
    gx, gy, gz, cov = resample_uniform(t[keep], a[keep, 0], a[keep, 1], a[keep, 2],
                                       lo, t0 + ROW_S)
    ep = epoch_features(gx, gy, gz, cov)
    n_ep = int(round(ROW_S / EPOCH_S))
    return ep[1:1 + n_ep] if len(ep) >= 1 + n_ep else None


def check_against(fd) -> None:
    """Fail loudly if config/features.yaml and this module disagree."""
    names = tuple(s.name for s in fd.for_scope("shared"))
    if names != ROW_FEATURES:
        raise ValueError("ring_row.ROW_FEATURES no longer matches features.yaml (shared "
                         "scope) - update this module before building rows")
    if float(fd.ppg_burst_seconds) != BURST_S:
        raise ValueError(f"features.yaml ppg_burst_seconds={fd.ppg_burst_seconds}, "
                         f"ring_row assumes {BURST_S}")
    for spec in fd.for_scope("shared"):
        fill = spec.missing_fill
        if fill is not None and not math.isnan(fill) and MISSING_FILL.get(spec.name) != fill:
            raise ValueError(f"{spec.name}: missing policy {fill} not mirrored in MISSING_FILL")
