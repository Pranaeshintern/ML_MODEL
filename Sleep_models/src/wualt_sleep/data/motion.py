"""Motion features from tri-axial accelerometry.

Dataset-independent: arrays in, features out. Accelerometry is assumed CONTINUOUS
across the row — unlike PPG, accel is cheap enough to leave running, and §5's onset
rule plus WASO detection both depend on motion between PPG bursts.

If the ring turns out to duty-cycle accel too, `sample_rate_hz` and the window length
still describe whatever is delivered; only the interpretation of `still_fraction`
changes.

PREFER `motion_features_on_grid`. Activity counts are an integrated absolute difference,
so their value scales with the sample rate, and the sub-window length is computed as
`round(rate x sub_window_s)` -- both wrong the moment the delivered rate is not the rate
passed in. BIDSleep's watch switches between ~50 and ~100 Hz within a night against a
rate estimated from its first 2,000 samples, so those rows were neither comparable to
DREAMT's 64 Hz nor internally consistent. Resampling onto the fixed grid first removes
both problems and is the same treatment the 30-second WASO features already get.
"""

from __future__ import annotations

import numpy as np

from .accel_epochs import GRID_HZ, resample_uniform

__all__ = ["MOTION_FEATURE_NAMES", "motion_features", "motion_features_on_grid",
           "activity_counts"]

MOTION_FEATURE_NAMES = (
    "activity_count_mean",
    "activity_count_max",
    "still_fraction",
    "acc_magnitude_std",
    "posture_change_count",
    "movement_burst_count",
)

SUB_WINDOW_S = 30.0     # activity counts are conventionally computed per epoch
STILL_THRESHOLD = 0.02  # g of dynamic acceleration; below this the wrist is at rest
BURST_THRESHOLD = 0.15  # g; a clear movement event
POSTURE_THRESHOLD = 0.30  # g of change in the gravity vector => limb reorientation


def activity_counts(magnitude: np.ndarray, sample_rate_hz: float, sub_window_s: float) -> np.ndarray:
    """Per-sub-window integrated absolute dynamic acceleration.

    Gravity is removed by differencing rather than by a high-pass filter: a filter
    needs a settled state and behaves badly on the short, gappy segments that wear
    artifacts produce, whereas differencing is local and degrades gracefully.
    """
    n = int(round(sample_rate_hz * sub_window_s))
    if n <= 1 or magnitude.size < n:
        return np.empty(0)
    usable = (magnitude.size // n) * n
    blocks = magnitude[:usable].reshape(-1, n)
    return np.abs(np.diff(blocks, axis=1)).sum(axis=1)


def motion_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    sample_rate_hz: float,
    sub_window_s: float = SUB_WINDOW_S,
) -> dict[str, float]:
    nan = dict.fromkeys(MOTION_FEATURE_NAMES, float("nan"))
    x, y, z = (np.asarray(a, dtype=np.float64).ravel() for a in (x, y, z))
    n = min(x.size, y.size, z.size)
    if n < sample_rate_hz * sub_window_s:
        return nan
    x, y, z = x[:n], y[:n], z[:n]

    magnitude = np.sqrt(x**2 + y**2 + z**2)
    if not np.isfinite(magnitude).any():
        return nan

    counts = activity_counts(magnitude, sample_rate_hz, sub_window_s)
    if counts.size == 0:
        return nan

    # Normalise counts to per-second so the value does not silently change meaning
    # when sample_rate_hz or sub_window_s differ between datasets.
    counts = counts / sub_window_s

    # Dynamic component only — subtracting the median removes the ~1 g gravity term
    # without assuming a particular orientation.
    dynamic = np.abs(magnitude - np.median(magnitude))

    features = {
        "activity_count_mean": float(np.mean(counts)),
        "activity_count_max": float(np.max(counts)),
        "still_fraction": float(np.mean(dynamic < STILL_THRESHOLD)),
        "acc_magnitude_std": float(np.std(magnitude)),
        "movement_burst_count": float(np.sum(counts > BURST_THRESHOLD)),
        "posture_change_count": _posture_changes(x, y, z, sample_rate_hz, sub_window_s),
    }
    return features


def _posture_changes(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, sub_window_s: float
) -> float:
    """Count sub-window-to-sub-window shifts in the mean gravity vector.

    Distinct from `movement_burst_count`: a burst is transient motion that ends where
    it started (a twitch), a posture change leaves the limb somewhere new. Only the
    latter reliably marks an arousal, so they must not be collapsed into one feature.
    """
    n = int(round(fs * sub_window_s))
    if n <= 1 or x.size < 2 * n:
        return 0.0
    usable = (x.size // n) * n
    means = np.stack(
        [a[:usable].reshape(-1, n).mean(axis=1) for a in (x, y, z)], axis=1
    )
    shifts = np.linalg.norm(np.diff(means, axis=0), axis=1)
    return float(np.sum(shifts > POSTURE_THRESHOLD))


def motion_features_on_grid(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t0: float,
    t1: float,
    *,
    grid_hz: float = GRID_HZ,
    sub_window_s: float = SUB_WINDOW_S,
) -> dict[str, float]:
    """`motion_features` on a fixed grid: device rate in, one common rate out.

    The window [t0, t1) is interpolated onto `grid_hz` before any feature is computed,
    so a 50 Hz watch and a 64 Hz wrist unit produce the same numbers for the same motion
    and a rate change mid-night cannot alter what a feature means.

    Non-finite samples are dropped before interpolation, matching `sqi_acc`, which is
    computed from valid timestamps only. Gaps are interpolated ACROSS rather than
    dropped -- dropping would shorten the window and shift every sub-window after the
    gap. A long dropout therefore reads as smooth, low-activity signal, which is what
    `sqi_acc` is there to expose.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    ax, ay, az = (np.asarray(a, dtype=np.float64).ravel() for a in (x, y, z))
    n = min(t.size, ax.size, ay.size, az.size)
    t, ax, ay, az = t[:n], ax[:n], ay[:n], az[:n]
    ok = np.isfinite(t) & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    if ok.sum() < 2:
        return dict.fromkeys(MOTION_FEATURE_NAMES, float("nan"))
    gx, gy, gz, _ = resample_uniform(t[ok], ax[ok], ay[ok], az[ok], t0, t1, grid_hz)
    return motion_features(gx, gy, gz, sample_rate_hz=grid_hz,
                           sub_window_s=sub_window_s)
