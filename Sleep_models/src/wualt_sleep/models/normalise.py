"""Per-night robust normalisation (D-014).

Absolute HR, HRV and skin temperature vary enormously between people — genetics, age,
fitness, ambient temperature — while the quantity that tracks sleep stage is the
DEVIATION from that person's own level on that night. Expressing each signal relative
to its own night is standard in the published ring literature; Oura's OSSA 2.0 paper
calls per-night normalisation critical, on exactly that reasoning.

WHY MEDIAN/IQR AND NOT THE PUBLISHED 5-95 PERCENTILE
Oura normalises over ~960 thirty-second epochs per night. A 15-minute row grid gives
about 32 rows, and the 5th percentile of 32 samples is the second-smallest value — it
is min/max in disguise, and one bad PPG burst sets it. The interquartile range is the
same idea estimated from the middle half, which survives a sample this small.

WHICH FEATURES, AND WHY NOT ALL OF THEM
Three kinds of feature must be left alone, and getting this list wrong is the main way
the transform makes things worse:

  already relative   `hr_delta_vs_rhr`, `temp_delta_vs_base` are personal-baseline
                     features already. Normalising them again is incoherent.
  already scale-free ratios, correlations and fractions — `sd1_sd2_ratio`,
                     `hr_autocorr_lag1`, `still_fraction` — carry meaning in absolute
                     terms and have no per-person offset to remove.
  meaning is absolute time and signal quality. Re-centring `clock_hour_sin` destroys
                     the clock; re-centring `sqi_ppg` would make a uniformly bad night
                     look average, which is the opposite of what quality is for.

Derivatives (`hr_slope`, `temp_slope`) are excluded too: they are already centred near
zero, so subtracting a median mostly injects noise.

NO LEAKAGE BY CONSTRUCTION
Each night is transformed using only its own rows, so this is fold-independent and can
be applied once to the whole table before any split. It cannot move information between
subjects.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PER_NIGHT_FEATURES", "MIN_ROWS", "per_night_transform"]

#: Features whose per-person absolute level is a nuisance rather than a signal.
#: ORDER IS IRRELEVANT — membership is what matters — but the SET is contractual:
#: a checkpoint trained with this set must be served with it, so it travels in
#: `.meta.json` and `loader.py` refuses a mismatch.
PER_NIGHT_FEATURES = (
    # heart rate level and spread; not hr_slope (a derivative) and not
    # hr_delta_vs_rhr (already relative to the person's resting HR)
    "hr_mean", "hr_min", "hr_std",
    # HRV amplitudes in ms; not sd1_sd2_ratio (a ratio) and not
    # hrv_sample_entropy (already a scale-free complexity measure)
    "rmssd", "sdnn", "pnn50", "poincare_sd1", "poincare_sd2",
    # motion amplitude — this is also what makes a 64 Hz wrist unit and a 50 Hz one
    # comparable; not still_fraction (a fraction) and not the burst/posture COUNTS,
    # which are mostly small integers whose IQR collapses to zero
    "activity_count_mean", "activity_count_max", "acc_magnitude_std",
    # skin temperature level and spread; not temp_slope, not temp_delta_vs_base
    "temp_mean", "temp_std",
    # BIDSleep's HR-variability analogues; not hr_autocorr_lag1 (a correlation)
    # and not hr_coverage (a quality measure)
    "hr_succ_diff_rms", "hr_range_iqr",
)

#: Below this many finite values in a night, the median and IQR are not estimable and
#: the feature is left untouched. In practice this only fires for a feature the corpus
#: does not have at all — BIDSleep's `rmssd` is entirely NaN — where the availability
#: mask already zeroes the column.
MIN_ROWS = 4

_SCALE_FLOOR = 1e-6


def per_night_transform(x: np.ndarray, cols: list[str],
                        features: tuple[str, ...] = PER_NIGHT_FEATURES) -> np.ndarray:
    """One night's (T, F) feature matrix -> the same shape, per-night normalised.

    For each selected column: subtract that night's median, divide by its interquartile
    range. NaNs are preserved, not filled — the availability mask downstream needs to
    keep telling "sensor absent" apart from "value zero".
    """
    out = np.array(x, dtype=np.float64, copy=True)
    idx = {c: i for i, c in enumerate(cols)}
    for name in features:
        j = idx.get(name)
        if j is None:
            continue
        col = out[:, j]
        finite = np.isfinite(col)
        if finite.sum() < MIN_ROWS:
            continue                      # not estimable — leave the column as it is
        v = col[finite]
        med = float(np.median(v))
        q25, q75 = np.percentile(v, [25.0, 75.0])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale < _SCALE_FLOOR:
            # A night that barely varied. Centre it, but do not divide by noise —
            # the fold-level standardisation that follows will set the scale.
            out[finite, j] = v - med
            continue
        out[finite, j] = (v - med) / scale
    return out
