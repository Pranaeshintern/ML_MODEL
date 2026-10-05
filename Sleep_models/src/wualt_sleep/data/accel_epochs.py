"""30-second accelerometer epoch features, device-agnostic.

The WASO channel runs at 30 seconds because that is where wake bouts live: 94.5% of
post-onset wake bouts are shorter than the 7.5 minutes a 15-minute majority vote needs,
but 57-63% of WASO MINUTES sit in bouts of 5 minutes or more, which this resolution can
see. WASO duration is recoverable at 30 s in a way awakening COUNT is not.

WHY EVERYTHING IS RESAMPLED TO A COMMON GRID
DREAMT is 64 Hz (Empatica E4, ACC in 1/64 g) and BIDSleep is ~50 Hz (in g). Activity
counts are conventionally an integrated absolute difference, and that quantity scales
with sample rate — pooling the two corpora on raw counts silently compares a 64 Hz
number against a 50 Hz one. Both are therefore interpolated onto a fixed GRID_HZ grid
before any feature is computed, so a feature means the same thing in both corpora. This
is the same class of confound as the unmatched burst sweep in the duty-cycle work, which
produced a fake plateau and nearly drove a wrong hardware recommendation.

WHAT IS DELIBERATELY NOT HERE
No heart rate, no temperature. The ring duty-cycles PPG to one 90-second burst per
15 minutes, so cardiac data cannot localise a wake bout to a 30-second epoch — it can
only confirm a total. This module is the accelerometer-only path, and keeping it
sensor-pure means it survives whatever the PPG duty cycle turns out to be.

FEATURE ORDER IS CONTRACTUAL. Append only; saved checkpoints depend on it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ACCEL_EPOCH_FEATURES", "GRID_HZ", "EPOCH_S", "resample_uniform",
           "time_coverage",
           "epoch_features", "CTX_FEATURES", "CTX_WINDOWS", "context_matrix",
           "context_names"]

#: The common grid. THE SHIPPED WASO ARTIFACT WAS BUILT AT THIS RATE (`grid_hz: 50.0` in
#: artifacts/waso/model.meta.json), and the epoch features are NOT rate-independent —
#: activity counts scale with sample rate — so scoring a night resampled to anything else
#: hands the model counts on a different scale and it returns a confident wrong band.
#: 50 Hz is BIDSleep's native rate, which leaves only DREAMT's 64 Hz interpolated; the
#: earlier value here was 32 Hz, which decimated both and no longer matches what shipped.
#: Changing this number invalidates the artifact.
GRID_HZ = 50.0
EPOCH_S = 30.0
SUB_S = 1.0             # sub-window for stillness and burst counting

STILL_G = 0.02          # g of within-second dynamic range below which the limb is at rest
BURST_G = 0.05          # g; a clear movement event
BANDS = ((0.1, 0.5), (0.5, 3.0), (3.0, 10.0))

ACCEL_EPOCH_FEATURES = (
    "act_mean",           # mean |d(magnitude)|/dt over the epoch, g/s
    "act_p95",            # 95th percentile of the same — peak movement intensity
    "jerk_rms",           # rms of d(magnitude)/dt, g/s
    "mag_std",            # std of acceleration magnitude, g
    "still_frac",         # fraction of 1-s sub-windows below STILL_G
    "n_moves",            # count of 1-s sub-windows above BURST_G
    "posture_delta",      # degrees between this epoch's gravity vector and the previous
    "posture_range",      # max degrees any 1-s sub-window deviates from the epoch mean
    "zcr",                # zero crossings per second of the demeaned magnitude
    "bp_lo",              # fraction of spectral power in 0.1-0.5 Hz
    "bp_mid",             # 0.5-3 Hz  — the band voluntary movement occupies
    "bp_hi",              # 3-10 Hz
    "coverage",           # fraction of grid samples backed by a real sample within 1 s
)


def resample_uniform(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     t0: float, t1: float, grid_hz: float = GRID_HZ
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate tri-axial acceleration onto a uniform grid spanning [t0, t1).

    Returns (gx, gy, gz, covered). `covered` marks grid points that have a real sample
    within 1 second; long dropouts are interpolated across rather than dropped, because
    dropping shifts every epoch after the gap and silently misaligns the labels — the
    same failure that put predictions 1.5-2.5 hours from their epochs in the derived
    metrics path.
    """
    n = int(np.floor((t1 - t0) * grid_hz))
    grid = t0 + np.arange(n) / grid_hz
    if t.size < 2:
        nan = np.full(n, np.nan)
        return nan, nan.copy(), nan.copy(), np.zeros(n, bool)
    order = np.argsort(t)
    t = t[order]
    out = [np.interp(grid, t, a[order]) for a in (x, y, z)]
    return out[0], out[1], out[2], _covered(grid, t)


#: A grid point counts as observed when a real sample lies within this many seconds.
MAX_GAP_S = 1.0


def _covered(grid: np.ndarray, t_sorted: np.ndarray) -> np.ndarray:
    """True where a real sample lies within MAX_GAP_S of each grid point.

    The single definition of accelerometer coverage: `resample_uniform` uses it for
    the 30-second `coverage` feature and `time_coverage` for the row-level `sqi_acc`,
    so the two cannot drift apart. Needs at least two samples.
    """
    idx = np.clip(np.searchsorted(t_sorted, grid), 1, t_sorted.size - 1)
    gap = np.minimum(np.abs(grid - t_sorted[idx - 1]), np.abs(t_sorted[idx] - grid))
    return gap <= MAX_GAP_S


def time_coverage(t_sorted: np.ndarray, t0: float, t1: float,
                  grid_hz: float = GRID_HZ) -> float:
    """`sqi_acc`: share of the window [t0, t1) that has a real sample within 1 s.

    Measured in TIME, not in samples, so it means the same thing at any sampling rate:
    an unbroken window scores 1.0 whether the device ran at 25, 50 or 100 Hz, and a
    5-minute dropout in a 15-minute row scores about 0.67. It cannot exceed 1.0.

    It replaced a sample COUNT divided by an assumed rate. BIDSleep's watch switches
    between ~50 and ~100 Hz within a night, so that ratio reached 1.83 and mostly
    measured the sampling rate -- and only on BIDSleep, which made it a corpus
    identifier. DREAMT's version only checked for NaN and was 1.0 on every row.

    `t_sorted` must be ascending and hold only timestamps of valid (finite) samples;
    it may be the whole night -- samples just outside the window count within 1 s,
    exactly as they do on the 30-second grid.
    """
    n = int(np.floor((t1 - t0) * grid_hz))
    if n <= 0:
        return 0.0
    lo = int(np.searchsorted(t_sorted, t0 - MAX_GAP_S, side="left"))
    hi = int(np.searchsorted(t_sorted, t1 + MAX_GAP_S, side="right"))
    local = t_sorted[lo:hi]
    if local.size == 0:
        return 0.0
    if local.size == 1:
        local = np.array([local[0], local[0]])
    grid = t0 + np.arange(n) / grid_hz
    return float(np.mean(_covered(grid, local)))


def epoch_features(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
                   covered: np.ndarray, grid_hz: float = GRID_HZ,
                   epoch_s: float = EPOCH_S) -> np.ndarray:
    """Uniformly-sampled tri-axial acceleration -> (n_epochs, len(ACCEL_EPOCH_FEATURES)).

    Vectorised over epochs; a per-epoch Python loop over 100k epochs is minutes of
    wall clock for no benefit.
    """
    per = int(round(grid_hz * epoch_s))
    sub = int(round(grid_hz * SUB_S))
    n_ep = int(gx.size // per)
    if n_ep == 0:
        return np.empty((0, len(ACCEL_EPOCH_FEATURES)))
    keep = n_ep * per

    X = gx[:keep].reshape(n_ep, per)
    Y = gy[:keep].reshape(n_ep, per)
    Z = gz[:keep].reshape(n_ep, per)
    cov = covered[:keep].reshape(n_ep, per).mean(1)

    mag = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    d = np.abs(np.diff(mag, axis=1)) * grid_hz          # g/s, rate-normalised

    act_mean = d.mean(1)
    act_p95 = np.percentile(d, 95, axis=1)
    jerk_rms = np.sqrt((d ** 2).mean(1))
    mag_std = mag.std(1)

    # ---- 1-second sub-windows: stillness and discrete movement events -------
    n_sub = per // sub
    sub_mag = mag[:, :n_sub * sub].reshape(n_ep, n_sub, sub)
    sub_range = sub_mag.max(2) - sub_mag.min(2)
    still_frac = (sub_range < STILL_G).mean(1)
    n_moves = (sub_range > BURST_G).sum(1).astype(float)

    # ---- posture: direction of the gravity vector ---------------------------
    # Magnitude says how hard the limb moved; direction says whether it ended up
    # somewhere else. Turning over and sitting up change direction, and that is
    # nearly orthogonal to activity intensity.
    g_ep = np.stack([X.mean(1), Y.mean(1), Z.mean(1)], 1)
    g_ep /= np.maximum(np.linalg.norm(g_ep, axis=1, keepdims=True), 1e-9)
    dot = np.clip((g_ep[1:] * g_ep[:-1]).sum(1), -1.0, 1.0)
    posture_delta = np.concatenate([[0.0], np.degrees(np.arccos(dot))])

    sub_g = np.stack([a[:, :n_sub * sub].reshape(n_ep, n_sub, sub).mean(2)
                      for a in (X, Y, Z)], 2)
    sub_g /= np.maximum(np.linalg.norm(sub_g, axis=2, keepdims=True), 1e-9)
    dot_in = np.clip((sub_g * g_ep[:, None, :]).sum(2), -1.0, 1.0)
    posture_range = np.degrees(np.arccos(dot_in)).max(1)

    # ---- frequency content --------------------------------------------------
    # Quiet wake is small, fast fidgeting; sleep is near-silent. A thresholded
    # still-fraction cannot tell those apart because both sit under the threshold,
    # so the spectrum is asked directly.
    centred = mag - mag.mean(1, keepdims=True)
    zcr = (np.diff(np.signbit(centred), axis=1).sum(1) / epoch_s).astype(float)

    spec = np.abs(np.fft.rfft(centred, axis=1)) ** 2
    freq = np.fft.rfftfreq(per, d=1.0 / grid_hz)
    total = spec.sum(1) + 1e-12
    bp = [spec[:, (freq >= lo) & (freq < hi)].sum(1) / total for lo, hi in BANDS]

    return np.column_stack([act_mean, act_p95, jerk_rms, mag_std, still_frac, n_moves,
                            posture_delta, posture_range, zcr, *bp, cov])


# ------------------------------------------------------- temporal context ---
# Kept beside the epoch features rather than in the training script: inference has to
# reproduce this transform exactly, and a runtime that imports from `pipelines/` to do
# so is a checkpoint nobody can load.

# `bp_mid` was here and has been removed. Measured twice: at the epoch level the
# spectral block bought nothing (AUC 0.862 with it, 0.863 without), and at the night
# level dropping it IMPROVED the band (exact 0.561 -> 0.581). The features are still
# EXTRACTED — the epoch dictionary is append-only and the parquet keeps them — they are
# simply not fed to a model. Keeping the extraction means the decision can be revisited
# without rebuilding 324k epochs.
CTX_FEATURES = ("act_mean", "still_frac", "n_moves", "posture_delta")
CTX_WINDOWS = (5, 21)          # 2.5 min and 10.5 min, centred


def _roll_mean(v: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(v, size=w, mode="nearest")


def _roll_max(v: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(v, size=w, mode="nearest")


def context_names(cols: list[str] | None = None) -> list[str]:
    names = list(cols or ACCEL_EPOCH_FEATURES)
    for f in CTX_FEATURES:
        for w in CTX_WINDOWS:
            names.append(f"{f}_mean{w}")
        names.append(f"{f}_max5")
    return names + ["act_mean_rel"]


def context_matrix(ep: np.ndarray, cols: list[str] | None = None) -> np.ndarray:
    """Per-epoch features plus temporal context. ORDER IS CONTRACTUAL.

    Wake is not identifiable from one 30-second epoch — a still epoch looks the same
    asleep or lying awake. What separates them is the neighbourhood: how much movement
    surrounds it, and how long the quiet stretch has run. Cole-Kripke used +/-5 min of
    context for the same reason.
    """
    cols = list(cols or ACCEL_EPOCH_FEATURES)
    idx = {c: i for i, c in enumerate(cols)}
    parts = [ep]
    for f in CTX_FEATURES:
        v = np.nan_to_num(ep[:, idx[f]].astype(np.float64), nan=0.0)
        for w in CTX_WINDOWS:
            parts.append(_roll_mean(v, w).reshape(-1, 1))
        parts.append(_roll_max(v, 5).reshape(-1, 1))
    # Self-normalised activity: absolute counts differ between a 64 Hz wrist unit and a
    # 50 Hz one, but a night's activity relative to its own median does not.
    a = np.nan_to_num(ep[:, idx["act_mean"]].astype(np.float64), nan=0.0)
    parts.append((a / max(float(np.median(a)), 1e-4)).reshape(-1, 1))
    return np.nan_to_num(np.hstack(parts), nan=0.0, posinf=0.0, neginf=0.0)
