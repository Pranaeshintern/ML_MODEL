"""L0 — signal conditioning.

Turns heterogeneous accelerometer sources into one canonical, orientation-normalised
representation so that a model trained on wrist data has a chance of transferring to
a ring.

Canonical output per session:
    fs       = 30 Hz
    acc      float32 [N, 3]  in g, device frame
    mag      float32 [N]     ||acc||
    a_v      float32 [N]     body-acceleration component along gravity (signed)
    a_h      float32 [N]     body-acceleration magnitude perpendicular to gravity
    g_vec    float32 [N, 3]  unit gravity direction (low-passed)

(mag, a_v, a_h) is invariant to rotation about the gravity axis, which is exactly
the degree of freedom that changes when a ring rotates on a finger or is worn on
the other hand.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

FS = 30.0  # canonical sampling rate (Hz)
GRAVITY_CUTOFF_HZ = 0.5  # below this is orientation, above is motion


def resample_to(t: np.ndarray, x: np.ndarray, fs_out: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Resample irregular/other-rate samples onto a uniform fs_out grid.

    Linear interpolation is adequate here: every source is already sampled at or
    above 20 Hz, and we low-pass before decimating anything faster.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    fs_in = _estimate_fs(t)
    # Anti-alias before downsampling.
    if fs_in > fs_out * 1.1:
        nyq_out = fs_out / 2.0
        wn = min(nyq_out / (fs_in / 2.0), 0.99)
        sos = sps.butter(4, wn, btype="low", output="sos")
        x = sps.sosfiltfilt(sos, x, axis=0)

    t_new = np.arange(t[0], t[-1], 1.0 / fs_out)
    out = np.empty((len(t_new), x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_new, t, x[:, c])
    return t_new, (out[:, 0] if squeeze else out)


def _estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return FS
    return float(1.0 / np.median(dt))


def autocalibrate(acc: np.ndarray, fs: float = FS) -> tuple[np.ndarray, dict]:
    """Estimate per-axis offset and gain from still periods so that |a| == 1 g at rest.

    This is the cheapest large win in cross-device transfer: it removes the
    manufacturing offset/gain differences between an Axivity, a Shimmer, a watch and
    a ring. Simplified variant of the Oxford `actipy` closed-form calibration.
    """
    acc = np.asarray(acc, dtype=np.float64)
    win = int(10 * fs)
    if acc.shape[0] < win * 3:
        return acc.astype(np.float32), {"applied": False, "reason": "too_short"}

    n_win = acc.shape[0] // win
    trimmed = acc[: n_win * win].reshape(n_win, win, 3)
    win_std = trimmed.std(axis=1)
    win_mean = trimmed.mean(axis=1)

    # A window is "still" when every axis is quiet; those windows should sit on the
    # unit sphere, and any deviation is offset/gain error.
    still = (win_std < 0.013).all(axis=1)
    pts = win_mean[still]
    if pts.shape[0] < 30 or np.unique(np.sign(pts).round(), axis=0).shape[0] < 3:
        return acc.astype(np.float32), {"applied": False, "reason": "insufficient_still_windows"}

    offset = np.zeros(3)
    gain = np.ones(3)
    for _ in range(30):
        corrected = (pts - offset) * gain
        norm = np.linalg.norm(corrected, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-6, None)
        target = corrected / norm  # nearest point on the unit sphere
        # One Gauss-Newton step per axis (linear least squares in offset/gain).
        for ax in range(3):
            A = np.column_stack([np.ones(pts.shape[0]), corrected[:, ax]])
            coef, *_ = np.linalg.lstsq(A, target[:, ax], rcond=None)
            offset[ax] = offset[ax] - coef[0] / max(gain[ax] * coef[1], 1e-6)
            gain[ax] = gain[ax] * coef[1]
        if np.abs(gain - 1).max() < 1e-6:
            break

    if not (np.isfinite(offset).all() and np.isfinite(gain).all()):
        return acc.astype(np.float32), {"applied": False, "reason": "diverged"}
    # Refuse implausible corrections rather than corrupting the signal.
    if np.abs(offset).max() > 0.5 or gain.max() > 1.5 or gain.min() < 0.67:
        return acc.astype(np.float32), {"applied": False, "reason": "implausible"}

    out = ((acc - offset) * gain).astype(np.float32)
    return out, {
        "applied": True,
        "offset": offset.tolist(),
        "gain": gain.tolist(),
        "n_still_windows": int(pts.shape[0]),
    }


def gravity_split(acc: np.ndarray, fs: float = FS) -> dict[str, np.ndarray]:
    """Split into orientation (gravity) and motion, then project into a
    gravity-aligned frame.

    Returns mag / a_v / a_h / g_vec. See module docstring for why.
    """
    acc = np.asarray(acc, dtype=np.float32)
    sos = sps.butter(4, GRAVITY_CUTOFF_HZ / (fs / 2.0), btype="low", output="sos")
    g_vec = sps.sosfiltfilt(sos, acc, axis=0).astype(np.float32)

    g_norm = np.linalg.norm(g_vec, axis=1, keepdims=True)
    g_unit = g_vec / np.clip(g_norm, 1e-6, None)

    a_body = acc - g_vec
    a_v = np.sum(a_body * g_unit, axis=1)  # signed vertical component
    a_h = np.linalg.norm(a_body - a_v[:, None] * g_unit, axis=1)

    return {
        "mag": np.linalg.norm(acc, axis=1).astype(np.float32),
        "a_v": a_v.astype(np.float32),
        "a_h": a_h.astype(np.float32),
        "g_unit": g_unit.astype(np.float32),
    }


def condition(
    t: np.ndarray,
    acc: np.ndarray,
    fs_out: float = FS,
    calibrate: bool = True,
) -> dict:
    """Full L0 chain: resample -> calibrate -> gravity split."""
    t_r, acc_r = resample_to(t, acc, fs_out)
    cal_info = {"applied": False, "reason": "disabled"}
    if calibrate:
        acc_r, cal_info = autocalibrate(acc_r, fs_out)
    out = gravity_split(acc_r, fs_out)
    out["t"] = t_r
    out["acc"] = acc_r
    out["calibration"] = cal_info
    return out


def align_hr(t_acc: np.ndarray, t_hr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """Put HR on the accelerometer time grid, leaving gaps as NaN.

    HR is NOT forward-filled across long gaps: a stale heart rate is worse than a
    missing one, because the model can handle missing (XGBoost routes NaN) but
    cannot know that a value is stale. Gaps up to 30 s are bridged, longer gaps
    become NaN.
    """
    t_hr = np.asarray(t_hr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)
    ok = np.isfinite(hr) & np.isfinite(t_hr)
    if ok.sum() < 2:
        return np.full(len(t_acc), np.nan, dtype=np.float32)
    t_hr, hr = t_hr[ok], hr[ok]

    out = np.interp(t_acc, t_hr, hr, left=np.nan, right=np.nan)
    # Blank out interpolation that spans a gap longer than 30 s.
    idx = np.searchsorted(t_hr, t_acc).clip(1, len(t_hr) - 1)
    gap = t_hr[idx] - t_hr[idx - 1]
    out[gap > 30.0] = np.nan
    return out.astype(np.float32)


def hr_from_ecg(ecg: np.ndarray, fs: float, smooth_s: float = 10.0) -> np.ndarray:
    """Derive instantaneous HR (bpm) from a single ECG lead by R-peak detection.

    This is what unlocks MHEALTH as a second accelerometer+HR dataset — it ships
    2-lead ECG that the original authors explicitly left unused.
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    ecg = ecg - np.median(ecg)

    # Pan-Tompkins-style emphasis: bandpass -> derivative -> square -> integrate.
    sos = sps.butter(3, [5.0 / (fs / 2), min(20.0 / (fs / 2), 0.99)], btype="band", output="sos")
    filt = sps.sosfiltfilt(sos, ecg)
    deriv = np.gradient(filt)
    energy = deriv**2
    win = max(1, int(0.12 * fs))
    integ = np.convolve(energy, np.ones(win) / win, mode="same")

    thresh = np.percentile(integ, 98) * 0.35
    peaks, _ = sps.find_peaks(integ, height=thresh, distance=int(0.3 * fs))
    if peaks.size < 3:
        return np.full(len(ecg), np.nan, dtype=np.float32)

    rr = np.diff(peaks) / fs
    inst_hr = 60.0 / rr
    # Reject physiologically impossible beats before smoothing.
    valid = (inst_hr > 30) & (inst_hr < 220)
    if valid.sum() < 2:
        return np.full(len(ecg), np.nan, dtype=np.float32)

    t_beat = peaks[1:][valid] / fs
    hr_beat = inst_hr[valid]
    t_all = np.arange(len(ecg)) / fs
    hr = np.interp(t_all, t_beat, hr_beat, left=np.nan, right=np.nan)

    k = max(1, int(smooth_s * fs))
    kernel = np.ones(k) / k
    finite = np.isfinite(hr)
    if finite.sum() > k:
        smoothed = np.convolve(np.nan_to_num(hr), kernel, mode="same")
        weight = np.convolve(finite.astype(float), kernel, mode="same")
        hr = np.where(weight > 0.5, smoothed / np.clip(weight, 1e-6, None), np.nan)
    return hr.astype(np.float32)
