"""L0 audio conditioning: microphone waveform -> 50 Hz amplitude envelope.

The chain, and why each stage is there:

  resample to 4 kHz    Snore energy is 20-300 Hz with harmonics into the low kHz.
                       4 kHz is ample, and it cuts the sample count - and so the
                       MCU's work - by 4x against a 16 kHz PDM feed.
  bandpass 20-300 Hz   Snoring is soft tissue vibrating, so it lives low. Speech
                       has fundamentals in this range but its energy sits in
                       formants at 500-3000 Hz, so the band is itself a filter.
                       The high-pass side also removes DC and handling rumble.
  rectify + 10 Hz LP   Snore bursts run 0.5-3 s, so the envelope's own bandwidth
                       is only a few Hz. 10 Hz preserves burst shape while
                       sitting well below the 25 Hz Nyquist of the 50 Hz output.
  decimate to 50 Hz    `periodicity.FS_ENV`. Everything downstream works here.

Streaming is the point, not an optimisation. `EnvelopeExtractor` consumes short
frames and carries filter state across them, so the full 10 s window never exists
in memory at once. That makes "audio is never stored" a property of the
architecture a reviewer can check, rather than a policy someone has to keep
honouring. `envelope_from_waveform` is a batch convenience that drives the very
same streaming object, so the two paths cannot drift apart.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import numpy as np
from scipy import signal as sps

from snore_detect.periodicity import FS_ENV

FS_WORK = 4000.0  # working rate after decimation, Hz
BAND_LO_HZ = 20.0
BAND_HI_HZ = 300.0
ENV_LP_HZ = 10.0
BAND_ORDER = 4
ENV_ORDER = 2

_EPS = 1e-12


def design_bandpass(fs: float = FS_WORK, lo: float = BAND_LO_HZ, hi: float = BAND_HI_HZ):
    return sps.butter(BAND_ORDER, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")


def design_envelope_lowpass(fs: float = FS_WORK, cutoff: float = ENV_LP_HZ):
    return sps.butter(ENV_ORDER, cutoff / (fs / 2), btype="low", output="sos")


def design_antialias(fs_in: float, fs_out: float):
    """Anti-alias ahead of integer decimation, at 90 % of the output Nyquist."""
    return sps.butter(6, 0.9 * (fs_out / 2) / (fs_in / 2), btype="low", output="sos")


def resample_to(x: np.ndarray, fs_in: float, fs_out: float = FS_WORK) -> np.ndarray:
    """Polyphase resample. Batch only - on-device this is the decimator in hardware."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    r = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), r.numerator, r.denominator)


class EnvelopeExtractor:
    """Streaming waveform -> envelope. Carries filter state across frames.

    `fs_in` must be an integer multiple of `fs_work`; a PDM front end delivering
    8/16/32/48 kHz all qualify. Rates that are not (44.1 kHz) have to be resampled
    up front with `resample_to`, which is inherently a batch operation.
    """

    def __init__(self, fs_in: float = FS_WORK, fs_work: float = FS_WORK,
                 fs_env: float = FS_ENV):
        ratio = fs_in / fs_work
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"fs_in={fs_in} is not an integer multiple of {fs_work}")
        env_ratio = fs_work / fs_env
        if abs(env_ratio - round(env_ratio)) > 1e-9:
            raise ValueError(f"fs_work={fs_work} is not an integer multiple of {fs_env}")

        self.fs_in, self.fs_work, self.fs_env = fs_in, fs_work, fs_env
        self.decim = int(round(ratio))
        self.env_decim = int(round(env_ratio))

        self._aa = design_antialias(fs_in, fs_work) if self.decim > 1 else None
        self._bp = design_bandpass(fs_work)
        self._lp = design_envelope_lowpass(fs_work)
        self.reset()

    def reset(self) -> None:
        self._zi_aa = sps.sosfilt_zi(self._aa) * 0.0 if self._aa is not None else None
        self._zi_bp = sps.sosfilt_zi(self._bp) * 0.0
        self._zi_lp = sps.sosfilt_zi(self._lp) * 0.0
        # Decimation phase, so a frame whose length is not a multiple of the ratio
        # does not shift the output grid relative to a single-shot call.
        self._phase_in = 0
        self._phase_env = 0

    @staticmethod
    def _decimate(x, factor, phase):
        if factor == 1:
            return x, phase
        start = (-phase) % factor
        return x[start::factor], (phase + x.size) % factor

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Feed one frame of raw audio; return the envelope samples it produced."""
        x = np.asarray(frame, dtype=np.float64).ravel()
        if x.size == 0:
            return np.zeros(0)

        if self._aa is not None:
            x, self._zi_aa = sps.sosfilt(self._aa, x, zi=self._zi_aa)
            x, self._phase_in = self._decimate(x, self.decim, self._phase_in)
            if x.size == 0:
                return np.zeros(0)

        x, self._zi_bp = sps.sosfilt(self._bp, x, zi=self._zi_bp)
        x = np.abs(x)
        x, self._zi_lp = sps.sosfilt(self._lp, x, zi=self._zi_lp)
        env, self._phase_env = self._decimate(x, self.env_decim, self._phase_env)
        return np.maximum(env, 0.0)


def envelope_from_waveform(
    x: np.ndarray,
    fs_in: float,
    frame_s: float = 1.0,
    fs_work: float = FS_WORK,
    fs_env: float = FS_ENV,
) -> np.ndarray:
    """Batch helper driving the streaming extractor frame by frame.

    Resamples first only when the input rate is not an integer multiple of
    `fs_work`, so the common device rates take the same path the device would.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    ratio = fs_in / fs_work
    if abs(ratio - round(ratio)) > 1e-9:
        x = resample_to(x, fs_in, fs_work)
        fs_in = fs_work

    ext = EnvelopeExtractor(fs_in, fs_work, fs_env)
    step = max(1, int(round(frame_s * fs_in)))
    pieces = [ext.process(x[i:i + step]) for i in range(0, x.size, step)]
    return np.concatenate(pieces) if pieces else np.zeros(0)


def envelope_batch(
    X: np.ndarray,
    fs_in: float = FS_WORK,
    fs_work: float = FS_WORK,
    fs_env: float = FS_ENV,
) -> np.ndarray:
    """Vectorised envelope over a batch of complete waveforms, one per row.

    Identical to running `EnvelopeExtractor` over each row from a fresh reset: the
    extractor starts with zero filter state, so a single-shot filter across the
    whole signal produces the same output as feeding it in frames. This exists for
    offline evaluation throughput only - the device path is the streaming one, and
    `test_batch_matches_streaming` pins the two together so they cannot drift.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    ratio = fs_in / fs_work
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"fs_in={fs_in} is not an integer multiple of {fs_work}")
    decim = int(round(ratio))
    env_decim = int(round(fs_work / fs_env))

    if decim > 1:
        X = sps.sosfilt(design_antialias(fs_in, fs_work), X, axis=1)[:, ::decim]
    X = sps.sosfilt(design_bandpass(fs_work), X, axis=1)
    X = np.abs(X)
    X = sps.sosfilt(design_envelope_lowpass(fs_work), X, axis=1)
    return np.maximum(X[:, ::env_decim], 0.0)


LEVEL_TRACE_LP_HZ = 1.5


def smooth_level_trace(x, fs, cutoff_hz=LEVEL_TRACE_LP_HZ):
    """Condition an already-decimated sound-level trace, e.g. a clinical PSG channel.

    UCDDB delivers its tracheal microphone as an 8 Hz `Sound` channel - past the
    point where L0's resample and bandpass apply, but *before* anything equivalent
    to the envelope low-pass. It arrives with 48 % of its power above 2 Hz, and on
    the raw signal the burst-duration measurement reads 0.12 s, about one sample,
    so no window passes the screener's 0.3-3.5 s gate. Low-passing at 1.5 Hz brings
    it to 0.46 s with every window in gate.

    Synthetic envelopes never showed this, because they were generated smooth. It
    is the clearest evidence so far that the envelope low-pass in the main chain is
    load-bearing rather than cosmetic.

    `filtfilt` is used rather than a causal filter: this is offline analysis of a
    complete recording, and zero phase shift keeps bursts aligned with the event
    timestamps they are being compared against. The device path stays causal.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return x
    b, a = sps.butter(2, cutoff_hz / nyq, btype="low")
    axis = -1 if x.ndim == 1 else 1
    return sps.filtfilt(b, a, x, axis=axis)


class NoiseFloorTracker:
    """Rolling estimate of the room's quiet level, persisted across windows.

    Spec 3.2 requires a candidate snore to sit clearly above the room floor,
    otherwise a noisy room manufactures snores all night. The tracker snaps down
    quickly and rises slowly: a genuinely quieter room should be believed at once,
    while a sustained loud stretch must not be absorbed into the baseline - if it
    were, a fan running all night would redefine "quiet" as itself and the SNR
    test would stop rejecting it, which is the exact failure the confuser sweep
    showed costs the most.

    This is the only part of L0 that carries state between windows.
    """

    # Rates are set against the sampling cadence, not by feel. At ~42 windows a
    # night, RELEASE=0.01 would let a 4 h confuser (~24 windows) drag the floor
    # 21 % of the way toward itself in a single night - most of the way to the
    # failure this exists to prevent. 0.003 holds that to 7 %, while still adopting
    # a genuinely noisier room over ~59 % in a week.
    ATTACK = 0.5     # move this far toward a lower observation
    RELEASE = 0.003  # ...and only this far toward a higher one

    def __init__(self, quantile: float = 10.0, initial: Optional[float] = None):
        self.quantile = quantile
        self.floor = initial

    def update(self, env: np.ndarray) -> float:
        level = float(np.percentile(np.asarray(env, dtype=np.float64), self.quantile))
        if self.floor is None:
            self.floor = level
        else:
            a = self.ATTACK if level < self.floor else self.RELEASE
            self.floor += a * (level - self.floor)
        return self.floor

    def snr_db(self, env: np.ndarray) -> float:
        """Peak of this window over the tracked floor, in dB."""
        if self.floor is None:
            return float("inf")
        peak = float(np.percentile(np.asarray(env, dtype=np.float64), 90))
        return float(20.0 * np.log10(max(peak, _EPS) / max(self.floor, _EPS)))
