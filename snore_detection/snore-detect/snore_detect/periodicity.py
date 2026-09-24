"""Envelope periodicity - the screener's load-bearing feature.

A snore and a fan have genuinely similar spectra: both low-frequency, both noisy,
both smeared further by a duvet. What separates them is time structure. Snoring
swells and fades *with breathing* - once every 3 to 5 seconds - and a stationary
noise source does not. So the discriminator is the autocorrelation of the
amplitude envelope, evaluated in the breathing-lag band.

This is hand-written rather than learned for two reasons:

  * It is the one property that survives the domain gap. Spectral shape does not
    transfer from AudioSet's recording conditions to a finger under a blanket -
    bedding is a low-pass filter and distance reshapes the rest. A 3-5 s
    repetition is 3-5 s regardless of what is muffling it.
  * A classifier fed 2 s patches cannot see it at all. One breath cycle is longer
    than the patch. Periodicity has to be computed at window scale or not at all.

Normalisation is a real choice, not a detail, and it is why window length matters:

  biased    r(k) = R(k) / R(0)          - attenuates by (n-k)/n, so a short window
                                          suppresses the true peak at long lags
  unbiased  r(k) = R(k)/(n-k) / var     - unattenuated, but the estimate at long
                                          lags is built from few overlapping
                                          samples and is correspondingly noisy,
                                          which manufactures peaks on noise

Neither escapes the fact that a 10 s window holds only two or three breath cycles.
Both are exposed so a sweep can measure the trade rather than assume it.

Harmonic suppression is not optional. Autocorrelation peaks at *every* multiple of
a signal's period, so a 1 s rhythm - a ticking clock, a dripping tap, a washing
machine - peaks at 3 s, 4 s and 5 s too, and scores 0.85 in the breathing band
without it. The fix is the standard octave check from pitch tracking: a genuine
breathing peak has a trough at half its lag, a harmonic of something faster does
not. Measured, this takes a 1 s tick from 0.850 to 0.000 while leaving a true 4 s
rhythm at 0.800 and a synthetic snore unchanged.

Vectorised over a batch of windows, following `features.extract`.
"""

from __future__ import annotations

import numpy as np

FS_ENV = 50.0  # envelope sample rate, Hz

# Sleeping adults breathe 12-20 times a minute: a 3.0-5.0 s cycle. The band is
# widened either side to tolerate the breath-to-breath jitter that snoring shows.
LAG_LO_S = 2.5
LAG_HI_S = 6.0

_EPS = 1e-12


def envelope_periodicity(
    env: np.ndarray,
    fs: float = FS_ENV,
    lag_lo_s: float = LAG_LO_S,
    lag_hi_s: float = LAG_HI_S,
    normalize: str = "biased",
    suppress_harmonics: bool = True,
) -> np.ndarray:
    """Peak envelope autocorrelation inside the breathing-lag band.

    `env` is [n_windows, n_samples] (or a single [n_samples] window). Returns one
    score per window; higher means "repeats at breathing rate".

    With `suppress_harmonics`, the in-band peak is discounted by whatever the
    autocorrelation shows at half and a third of its lag, so a rhythm that is
    merely a harmonic of something faster scores zero.
    """
    x = np.atleast_2d(np.asarray(env, dtype=np.float64))
    n = x.shape[1]
    x = x - x.mean(axis=1, keepdims=True)

    lo = int(np.floor(lag_lo_s * fs))
    hi = min(n - 1, int(np.ceil(lag_hi_s * fs)))
    if hi <= lo or n < 2:
        # The window is shorter than one breath cycle: the feature is undefined,
        # and reporting 0 is the honest answer rather than a small random number.
        return np.zeros(x.shape[0], dtype=np.float64)

    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft, axis=1)
    ac = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n]

    r0 = ac[:, :1]
    if normalize == "biased":
        r = ac / np.maximum(r0, _EPS)
    elif normalize == "unbiased":
        counts = (n - np.arange(n)).astype(np.float64)
        r = (ac / counts) / np.maximum(r0 / n, _EPS)
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    band = r[:, lo:hi + 1]
    peak = band.max(axis=1)
    if not suppress_harmonics:
        return peak

    rows = np.arange(r.shape[0])
    idx = lo + band.argmax(axis=1)
    sub = np.maximum(r[rows, idx // 2], r[rows, idx // 3])
    return np.maximum(peak - np.maximum(sub, 0.0), 0.0)


def usable_breath_cycles(window_s: float, period_s: float = 4.0) -> float:
    """How many breath cycles a window of this length can actually observe.

    Below about two the autocorrelation has nothing to correlate and the feature
    is decoration. This is the quantity the window-length sweep is really about.
    """
    return window_s / period_s
