# -*- coding: utf-8 -*-
"""The inference entry point.

Audio in, score out. Everything below this is the same code the model was
evaluated with; see README for the input contract and its limits.
"""

from __future__ import annotations

import json
import os

import numpy as np
import xgboost as xgb

from snore_detect.aggregate import decide_night
from snore_detect.audio_features import scalar_features
from snore_detect.conditioning import FS_WORK, resample_to

HERE = os.path.dirname(os.path.abspath(__file__))


class SnoreDetector:
    """Scores 10-second windows of audio, and a night of those scores."""

    def __init__(self, model_path: str | None = None, meta_path: str | None = None):
        self.meta = json.load(open(meta_path or os.path.join(HERE, "model_meta.json")))
        self.threshold = float(self.meta["threshold"])
        self.window_s = float(self.meta["window_s"])
        self.fs = float(self.meta["fs_hz"])
        self.n_samples = int(self.window_s * self.fs)
        self.feature_names = list(self.meta["feature_names"])
        self._booster = xgb.Booster()
        self._booster.load_model(model_path or os.path.join(HERE, "model.ubj"))

    # -- features -------------------------------------------------------------
    def features(self, X: np.ndarray, fs: float) -> np.ndarray:
        """(n, samples) audio at `fs` -> (n, 55) features, in training order."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if fs != self.fs:
            X = np.stack([resample_to(row, fs, self.fs) for row in X])
        M, names = scalar_features(X, self.fs)
        if names != self.feature_names:
            raise RuntimeError("feature order does not match the trained model")
        return M

    # -- scoring --------------------------------------------------------------
    def score(self, X: np.ndarray, fs: float) -> np.ndarray:
        """(n, samples) -> (n,) scores in [0, 1]. A ranking, not a probability."""
        return np.asarray(self._booster.inplace_predict(self.features(X, fs)),
                          dtype=np.float64)

    def score_window(self, x: np.ndarray, fs: float) -> dict:
        """One window -> {'score': float, 'snoring': bool}."""
        s = float(self.score(np.asarray(x)[None, :], fs)[0])
        return {"score": s, "snoring": s > self.threshold}

    def score_stream(self, x: np.ndarray, fs: float) -> dict:
        """A longer recording -> per-window results, cut into whole windows."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if fs != self.fs:
            x = resample_to(x, fs, self.fs)
        n = x.size // self.n_samples
        if n == 0:
            raise ValueError(f"need at least {self.window_s:g} s of audio")
        X = x[: n * self.n_samples].reshape(n, self.n_samples)
        p = self.score(X, self.fs)
        t = np.arange(n) * self.window_s
        return {"t_start": t, "score": p, "snoring": p > self.threshold}

    # -- night ----------------------------------------------------------------
    def decide_night(self, t_start, snoring, mic_s: float | None = None):
        """Window results across a night -> one verdict."""
        t_start = np.asarray(t_start, dtype=np.float64)
        snoring = np.asarray(snoring, dtype=bool)
        if mic_s is None:
            mic_s = t_start.size * self.window_s
        return decide_night(t_start, snoring, mic_s=mic_s)
