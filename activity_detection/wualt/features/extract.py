"""Feature extraction for 30 s windows.

Design rules:
  1. Every feature is invariant to rotation about the gravity axis. The previous
     iteration's top-gain features included corr_xy / z_std_ratio / axis_std_max,
     which encode *where the device sits*. Those transfer badly from a wrist to a
     ring worn at an arbitrary rotation.
  2. Cadence/periodicity features are first-class. Walking vs. non-periodic
     household arm motion is fundamentally a periodicity question, and household
     motion is the dominant false-positive source (cooking 27 %, manual work 48 %,
     showering 78 % in the previous evaluation).
  3. Everything is vectorised over all windows of a session at once. CAPTURE-24 is
     151 subjects x 24 h; a per-window Python loop would not finish.

All functions take/return float32 and operate on arrays shaped [n_windows, win_len].
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats as sstats

FS = 30.0
WIN_S = 30.0
HOP_S = 15.0
WIN_LEN = int(WIN_S * FS)  # 900
HOP_LEN = int(HOP_S * FS)  # 450

_EPS = 1e-8
STILL_STD_G = 0.02  # per-second stillness threshold, in g


def window_view(x: np.ndarray, win_len: int = WIN_LEN, hop: int = HOP_LEN) -> np.ndarray:
    """Non-copying [n_windows, win_len] view (or [n_windows, win_len, C] if 2-D)."""
    if x.shape[0] < win_len:
        shape = (0, win_len) if x.ndim == 1 else (0, win_len, x.shape[1])
        return np.empty(shape, dtype=x.dtype)
    if x.ndim == 1:
        return sliding_window_view(x, win_len)[::hop]
    return sliding_window_view(x, win_len, axis=0)[::hop].transpose(0, 2, 1)


def _percentiles(w: np.ndarray, qs: list[float]) -> np.ndarray:
    return np.percentile(w, qs, axis=1)


def _autocorr(w: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased normalised autocorrelation via FFT, lags 0..max_lag."""
    n = w.shape[1]
    w = w - w.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(w, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, : max_lag + 1]
    # Unbiased: divide by the number of overlapping samples at each lag.
    counts = (n - np.arange(max_lag + 1)).astype(np.float32)
    ac = ac / counts[None, :]
    return ac / (ac[:, :1] + _EPS)


def _stat_block(w: np.ndarray, prefix: str, out: dict) -> None:
    """Shared descriptive statistics for a scalar channel."""
    out[f"{prefix}_mean"] = w.mean(axis=1)
    out[f"{prefix}_std"] = w.std(axis=1)
    out[f"{prefix}_max"] = w.max(axis=1)
    out[f"{prefix}_rms"] = np.sqrt((w**2).mean(axis=1))
    p10, p25, p75, p90 = _percentiles(w, [10, 25, 75, 90])
    out[f"{prefix}_p10"] = p10
    out[f"{prefix}_p90"] = p90
    out[f"{prefix}_iqr"] = p75 - p25
    out[f"{prefix}_range"] = w.max(axis=1) - w.min(axis=1)
    # Skew/kurtosis are undefined for a constant window and scipy warns about
    # catastrophic cancellation there. Those windows come from non-wear periods
    # (a worn device always reads ~1 g) and are dropped later, but zero them so a
    # numerically meaningless value can never reach the model.
    degenerate = out[f"{prefix}_std"] < 1e-6
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # scipy warns per call, not per row, so suppress it here rather than let a
        # handful of constant rows produce noise on every build.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sk = sstats.skew(w, axis=1)
        ku = sstats.kurtosis(w, axis=1)
    out[f"{prefix}_skew"] = np.where(degenerate, 0.0, sk)
    out[f"{prefix}_kurt"] = np.where(degenerate, 0.0, ku)


def extract(
    mag: np.ndarray,
    a_v: np.ndarray,
    a_h: np.ndarray,
    g_unit: np.ndarray,
    hr: np.ndarray | None = None,
    fs: float = FS,
    win_len: int = WIN_LEN,
    hop: int = HOP_LEN,
) -> tuple[np.ndarray, list[str]]:
    """Extract the full feature matrix for one conditioned session.

    Returns (X [n_windows, n_features] float32, feature_names).
    """
    W_mag = window_view(mag, win_len, hop)
    W_av = window_view(a_v, win_len, hop)
    W_ah = window_view(a_h, win_len, hop)
    n_win = W_mag.shape[0]
    if n_win == 0:
        return np.empty((0, 0), dtype=np.float32), []

    out: dict[str, np.ndarray] = {}

    # ---- 1. magnitude and body-acceleration statistics -------------------------
    _stat_block(W_mag, "mag", out)
    _stat_block(W_av, "av", out)
    out["ah_mean"] = W_ah.mean(axis=1)
    out["ah_std"] = W_ah.std(axis=1)
    out["ah_max"] = W_ah.max(axis=1)
    out["ah_rms"] = np.sqrt((W_ah**2).mean(axis=1))

    av_std = out["av_std"]
    ah_std = out["ah_std"]
    # Vertical-vs-horizontal balance: high for gait, low for arm-dominated motion.
    out["vh_ratio"] = av_std / (ah_std + _EPS)
    out["sma"] = np.abs(W_av).mean(axis=1) + W_ah.mean(axis=1)

    # Correlation between vertical and horizontal motion — rotation invariant,
    # unlike the per-axis correlations it replaces.
    av_c = W_av - W_av.mean(axis=1, keepdims=True)
    ah_c = W_ah - W_ah.mean(axis=1, keepdims=True)
    out["corr_vh"] = (av_c * ah_c).mean(axis=1) / (av_std * ah_std + _EPS)

    # Zero-crossing rate of the vertical component (gait produces a steady rate).
    out["av_zcr"] = (np.diff(np.signbit(W_av), axis=1).sum(axis=1) / (win_len / fs)).astype(
        np.float32
    )

    # ---- 2. jerk and impact shape ---------------------------------------------
    jerk = np.diff(W_av, axis=1) * fs
    abs_jerk = np.abs(jerk)
    out["jerk_mean"] = abs_jerk.mean(axis=1)
    out["jerk_std"] = jerk.std(axis=1)
    out["jerk_p95"] = np.percentile(abs_jerk, 95, axis=1)
    # Crest factor separates impact activities (running: sharp heel strike) from
    # smooth periodic ones (cycling, elliptical) at the same intensity.
    out["crest_factor"] = np.abs(W_av).max(axis=1) / (out["av_rms"] + _EPS)
    k = max(1, int(0.05 * abs_jerk.shape[1]))
    out["impact_index"] = np.sort(abs_jerk, axis=1)[:, -k:].mean(axis=1)

    # ---- 3. sub-second dynamics ------------------------------------------------
    # These dominated the previous model's gain ranking (sub_std_iqr, sub_std_p75).
    n_sub = win_len // int(fs)
    sub = W_mag[:, : n_sub * int(fs)].reshape(n_win, n_sub, int(fs))
    sub_std = sub.std(axis=2)
    out["sub_std_mean"] = sub_std.mean(axis=1)
    out["sub_std_median"] = np.median(sub_std, axis=1)
    sp25, sp75, sp90 = np.percentile(sub_std, [25, 75, 90], axis=1)
    out["sub_std_p75"] = sp75
    out["sub_std_p90"] = sp90
    out["sub_std_iqr"] = sp75 - sp25
    out["sub_std_cv"] = sub_std.std(axis=1) / (sub_std.mean(axis=1) + _EPS)

    active = sub_std > STILL_STD_G
    out["active_second_fraction"] = active.mean(axis=1)
    # Run-length statistics: a bout of sustained motion vs. isolated twitches.
    padded = np.pad(active, ((0, 0), (1, 1)), constant_values=False)
    starts = (~padded[:, :-1] & padded[:, 1:]).sum(axis=1)
    out["active_run_count"] = starts.astype(np.float32)
    out["longest_active_run_s"] = _longest_run(active)
    out["longest_still_run_s"] = _longest_run(~active)

    # ---- 4. spectral -----------------------------------------------------------
    win_fn = np.hanning(win_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(W_av * win_fn, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win_len, 1.0 / fs)
    total = spec.sum(axis=1) + _EPS
    psd = spec / total[:, None]

    out["total_power"] = total
    dom_idx = np.argmax(spec[:, 1:], axis=1) + 1
    out["dom_freq_hz"] = freqs[dom_idx]
    out["dom_freq_power"] = psd[np.arange(n_win), dom_idx]
    out["spectral_entropy"] = -(psd * np.log(psd + _EPS)).sum(axis=1) / np.log(len(freqs))
    out["spectral_centroid"] = (psd * freqs).sum(axis=1)
    out["spectral_spread"] = np.sqrt(
        (psd * (freqs[None, :] - out["spectral_centroid"][:, None]) ** 2).sum(axis=1)
    )
    cumsum = np.cumsum(psd, axis=1)
    out["spectral_rolloff"] = freqs[np.argmax(cumsum >= 0.95, axis=1)]

    bands = [(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 10.0)]
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        name = f"band_{lo}_{hi}hz".replace(".", "p")
        out[name] = psd[:, sel].sum(axis=1)

    # Ratio of the dominant peak to the second peak: how single-tone the motion is.
    spec_masked = spec.copy()
    guard = max(1, int(0.3 / (freqs[1] - freqs[0])))
    for i in range(n_win):
        lo = max(1, dom_idx[i] - guard)
        spec_masked[i, lo : dom_idx[i] + guard + 1] = 0
    second = spec_masked[:, 1:].max(axis=1)
    out["spectral_peak_ratio"] = spec[np.arange(n_win), dom_idx] / (second + _EPS)

    # Harmonic ratio: gait puts energy at integer multiples of the step frequency.
    harm = np.zeros(n_win, dtype=np.float32)
    df = freqs[1] - freqs[0]
    for h in (2, 3, 4):
        hidx = np.clip((dom_idx * h).astype(int), 0, len(freqs) - 1)
        lo = np.clip(hidx - guard, 0, len(freqs) - 1)
        hi = np.clip(hidx + guard + 1, 0, len(freqs))
        for i in range(n_win):
            harm[i] += spec[i, lo[i] : hi[i]].max() if hi[i] > lo[i] else 0.0
    out["harmonic_ratio"] = harm / (spec[np.arange(n_win), dom_idx] + _EPS)

    # ---- 5. cadence and gait regularity ---------------------------------------
    max_lag = int(2.5 * fs)
    ac = _autocorr(W_av, max_lag)
    lo_lag, hi_lag = int(0.25 * fs), int(1.5 * fs)  # step rate 0.67-4 Hz
    seg = ac[:, lo_lag : hi_lag + 1]
    step_rel = np.argmax(seg, axis=1)
    step_lag = step_rel + lo_lag
    out["autocorr_peak_height"] = seg[np.arange(n_win), step_rel]
    out["autocorr_peak_lag_s"] = step_lag / fs
    out["cadence_hz"] = fs / np.maximum(step_lag, 1)

    # Step vs. stride regularity: in real gait the autocorrelation has a peak at the
    # step period and a stronger one at the stride (2 steps). Their ratio is a
    # standard gait-symmetry measure and is near 1 for machine-like periodic motion.
    stride_lag = np.clip(step_lag * 2, 0, max_lag)
    step_reg = ac[np.arange(n_win), step_lag]
    stride_reg = ac[np.arange(n_win), stride_lag]
    out["step_regularity"] = step_reg
    out["stride_regularity"] = stride_reg
    out["gait_symmetry"] = stride_reg / (step_reg + _EPS)

    # Cadence stability: how constant the step interval is across the window.
    out["cadence_stability"] = _cadence_stability(W_av, fs)

    # ---- 6. orientation dynamics (rotation invariant) --------------------------
    # Angular speed of the gravity vector: high when the device reorients (yoga,
    # reaching, household work), near zero for steady locomotion.
    W_g = window_view(g_unit, win_len, hop)
    dot = np.clip((W_g[:, :-1, :] * W_g[:, 1:, :]).sum(axis=2), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot)) * fs
    out["g_change_rate"] = ang.mean(axis=1)
    out["g_change_p90"] = np.percentile(ang, 90, axis=1)
    mean_g = W_g.mean(axis=1)
    mean_g /= np.linalg.norm(mean_g, axis=1, keepdims=True) + _EPS
    dev = np.degrees(
        np.arccos(np.clip((W_g * mean_g[:, None, :]).sum(axis=2), -1.0, 1.0))
    )
    out["tilt_dev_std"] = dev.std(axis=1)
    out["tilt_dev_max"] = dev.max(axis=1)

    # ---- 7. heart rate ---------------------------------------------------------
    if hr is not None:
        W_hr = window_view(hr, win_len, hop)
        valid = np.isfinite(W_hr)
        vfrac = valid.mean(axis=1)
        out["hr_valid_frac"] = vfrac
        with np.errstate(invalid="ignore", divide="ignore"):
            out["hr_mean"] = np.nanmean(W_hr, axis=1)
            out["hr_std"] = np.nanstd(W_hr, axis=1)
            out["hr_min"] = np.nanmin(W_hr, axis=1)
            out["hr_max"] = np.nanmax(W_hr, axis=1)
        out["hr_range"] = out["hr_max"] - out["hr_min"]
        # Linear trend in bpm/min — leads the plateau, so it flags effort onset.
        tt = np.arange(win_len, dtype=np.float32) / fs
        out["hr_slope"] = _nan_slope(W_hr, tt) * 60.0
        # Below half coverage the HR features are not trustworthy; emit NaN and let
        # the tree route it rather than imputing a heart rate that was never measured.
        bad = vfrac < 0.5
        for key in ("hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_slope"):
            out[key] = np.where(bad, np.nan, out[key])
    else:
        for key in (
            "hr_valid_frac", "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_slope",
        ):
            out[key] = np.full(n_win, np.nan, dtype=np.float32)
        out["hr_valid_frac"] = np.zeros(n_win, dtype=np.float32)

    names = list(out.keys())
    X = np.column_stack([np.asarray(out[k], dtype=np.float32) for k in names])
    return np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan), names


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True per row, in seconds (sub-windows are 1 s)."""
    n_win, n_sub = mask.shape
    best = np.zeros(n_win, dtype=np.float32)
    cur = np.zeros(n_win, dtype=np.float32)
    for j in range(n_sub):
        col = mask[:, j]
        cur = np.where(col, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _cadence_stability(W: np.ndarray, fs: float) -> np.ndarray:
    """Coefficient of variation of the interval between vertical-acceleration peaks.

    Low for steady gait or pedalling, high for irregular household motion.
    """
    n_win = W.shape[0]
    out = np.full(n_win, np.nan, dtype=np.float32)
    thr = W.std(axis=1) * 0.5
    min_dist = max(1, int(0.25 * fs))
    for i in range(n_win):
        x = W[i]
        if thr[i] <= _EPS:
            continue
        above = x > thr[i]
        # Peak = local maximum above threshold, enforced minimum spacing.
        idx = np.flatnonzero(above[1:-1] & (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
        if idx.size < 3:
            continue
        keep = [idx[0]]
        for p in idx[1:]:
            if p - keep[-1] >= min_dist:
                keep.append(p)
        if len(keep) < 3:
            continue
        iv = np.diff(np.asarray(keep)) / fs
        out[i] = iv.std() / (iv.mean() + _EPS)
    return out


def _nan_slope(W: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least-squares slope per row, ignoring NaN."""
    valid = np.isfinite(W)
    n = valid.sum(axis=1)
    Wz = np.where(valid, W, 0.0)
    tz = np.where(valid, t[None, :], 0.0)
    sum_t = tz.sum(axis=1)
    sum_w = Wz.sum(axis=1)
    sum_tt = (tz**2).sum(axis=1)
    sum_tw = (tz * Wz).sum(axis=1)
    denom = n * sum_tt - sum_t**2
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * sum_tw - sum_t * sum_w) / denom
    return np.where(n >= 2, slope, np.nan).astype(np.float32)
