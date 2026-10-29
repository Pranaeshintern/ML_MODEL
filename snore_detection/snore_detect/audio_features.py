"""Representations for a learned snore classifier, from 4 kHz audio windows.

Two of them, because they answer different questions:

  `scalar_features`  ~60 hand-designed numbers for a gradient-boosted tree. This
                     is the ship candidate - it fits an MCU and its errors can be
                     traced to a named quantity.
  `log_mel`          40 x ~78 patches for a small CNN. This is the ceiling
                     measurement: how much is left on the table by hand features.

The `subtract_mean` option on `log_mel` exists to settle the question this whole
project keeps circling. Every baseline so far has been a loudness proxy - RMS,
90th-percentile amplitude, band energy - all within 0.02 AUC of each other, which
is what you would expect if they are one feature wearing different hats. Removing
each window's mean log-energy destroys absolute loudness while leaving spectral
shape and temporal structure intact. If a model trained that way still works, there
is real structure beyond loudness; if it collapses, there is not, and the honest
answer is that snore detection here is loudness detection.

That matters beyond curiosity: loudness is exactly the feature that will not
survive the move to a ring, where hand position moves the level by more than the
signal does (spec 5.1).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.fftpack import dct

from snore_detect.conditioning import envelope_batch
from snore_detect.periodicity import FS_ENV, envelope_periodicity
from snore_detect.burst import burst_duration_s

N_FFT = 512
HOP = 512
N_MELS = 40
F_MIN = 20.0
F_MAX = 1900.0  # just under Nyquist at 4 kHz
_EPS = 1e-10

# Band edges chosen around the snore fundamental (20-300 Hz) with finer resolution
# low down, where the discriminative energy is, and coarser above.
BANDS = ((20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 1900))


def _mel_filterbank(fs, n_fft, n_mels, fmin, fmax):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / float(b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / float(c - b)
    return fb


def log_mel(X, fs, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, subtract_mean=False):
    """[n_windows, n_samples] -> [n_windows, n_mels, n_frames] log-mel patches."""
    x = np.asarray(X, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    n_frames = 1 + (n - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[:, idx] * np.hanning(n_fft)[None, None, :]
    P = np.abs(np.fft.rfft(frames, n_fft, axis=2)) ** 2
    fb = _mel_filterbank(fs, n_fft, n_mels, F_MIN, F_MAX)
    mel = np.log(np.einsum("wfk,mk->wmf", P, fb) + _EPS)
    if subtract_mean:
        # Kill absolute level; keep spectral shape and temporal structure.
        mel = mel - mel.mean(axis=(1, 2), keepdims=True)
    return mel.astype(np.float32)


def scalar_features(X, fs):
    """~60 numbers per window: spectrum, envelope shape, burst and rhythm structure."""
    x = np.asarray(X, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    out = {}

    # --- level ---
    rms = np.sqrt((x ** 2).mean(axis=1))
    out["rms"] = rms
    out["log_rms"] = np.log(rms + _EPS)
    for q in (50, 75, 90, 99):
        out[f"amp_p{q}"] = np.percentile(np.abs(x), q, axis=1)
    out["crest"] = np.abs(x).max(axis=1) / np.maximum(rms, _EPS)
    out["zcr"] = np.mean(np.diff(np.signbit(x), axis=1) != 0, axis=1)

    # --- spectrum ---
    P = np.abs(np.fft.rfft(x * np.hanning(n)[None, :], axis=1)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    tot = P.sum(axis=1) + _EPS
    for a, b in BANDS:
        e = P[:, (fr >= a) & (fr < b)].sum(axis=1)
        out[f"band_{a}_{b}"] = np.log(e + _EPS)
        out[f"bandfrac_{a}_{b}"] = e / tot
    out["centroid"] = (P * fr[None, :]).sum(axis=1) / tot
    out["spread"] = np.sqrt((P * (fr[None, :] - out["centroid"][:, None]) ** 2).sum(axis=1) / tot)
    out["flatness"] = np.exp(np.log(P + _EPS).mean(axis=1)) / (P.mean(axis=1) + _EPS)
    csum = np.cumsum(P, axis=1) / tot[:, None]
    for r in (0.5, 0.85, 0.95):
        out[f"rolloff_{int(r*100)}"] = fr[np.argmax(csum >= r, axis=1)]

    # --- cepstral: coarse spectral shape, level-invariant after the first coeff ---
    mel = log_mel(x, fs)
    m = mel.mean(axis=2)
    c = dct(m, type=2, axis=1, norm="ortho")[:, :10]
    for i in range(c.shape[1]):
        out[f"mfcc_{i}"] = c[:, i]
    out["mel_frame_std"] = mel.std(axis=2).mean(axis=1)

    # --- envelope: shape, bursts, rhythm ---
    env = envelope_batch(x, fs)
    e = env / np.maximum(env.mean(axis=1, keepdims=True), _EPS)
    out["env_std"] = e.std(axis=1)
    out["env_skew"] = ((e - e.mean(axis=1, keepdims=True)) ** 3).mean(axis=1) / \
        np.maximum(e.std(axis=1) ** 3, _EPS)
    out["env_kurt"] = ((e - e.mean(axis=1, keepdims=True)) ** 4).mean(axis=1) / \
        np.maximum(e.std(axis=1) ** 4, _EPS)
    for q in (10, 50, 90):
        out[f"env_p{q}"] = np.percentile(env, q, axis=1)
    out["env_dyn_range"] = np.log(np.percentile(env, 90, axis=1) + _EPS) - \
        np.log(np.percentile(env, 10, axis=1) + _EPS)
    out["burst_duration"] = burst_duration_s(env, fs=FS_ENV)
    thr = np.percentile(env, 10, axis=1, keepdims=True) + 0.5 * (
        np.percentile(env, 90, axis=1, keepdims=True)
        - np.percentile(env, 10, axis=1, keepdims=True))
    above = env > thr
    out["duty_cycle"] = above.mean(axis=1)
    pad = np.zeros((above.shape[0], 1), bool)
    out["n_bursts"] = (np.diff(np.concatenate([pad, above], axis=1).astype(np.int8),
                               axis=1) == 1).sum(axis=1).astype(float)

    out["periodicity"] = envelope_periodicity(env, fs=FS_ENV)
    out["periodicity_raw"] = envelope_periodicity(env, fs=FS_ENV, suppress_harmonics=False)
    for lo, hi in ((1.0, 2.5), (2.5, 6.0), (6.0, 10.0)):
        out[f"periodicity_{lo}_{hi}"] = envelope_periodicity(
            env, fs=FS_ENV, lag_lo_s=lo, lag_hi_s=hi, suppress_harmonics=False)

    # Slow amplitude modulation - a snoring stretch swells and fades; room tone does not.
    fr_e, Pe = sps.welch(env - env.mean(axis=1, keepdims=True), fs=FS_ENV,
                         nperseg=min(256, env.shape[1]), axis=1)
    te = Pe.sum(axis=1) + _EPS
    for a, b in ((0.05, 0.15), (0.15, 0.5), (0.5, 2.0)):
        out[f"envband_{a}_{b}"] = Pe[:, (fr_e >= a) & (fr_e < b)].sum(axis=1) / te

    names = sorted(out)
    M = np.column_stack([np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
                         for k in names])
    return M.astype(np.float32), names
