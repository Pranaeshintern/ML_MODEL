# -*- coding: utf-8 -*-
"""The inference entry point.

Audio in, score out. Everything below this is the same code the model was
evaluated with; see README for the input contract and its limits.

Backend: ONNX Runtime by default (model.onnx), which is what production ships.
XGBoost (model.ubj) is kept for re-export and as a cross-check, and is used only
if onnxruntime is unavailable or `backend="xgboost"` is asked for. The two agree
to 6e-7 over 20,000 random feature vectors with no decision flips at the
threshold; `python tools/export_onnx.py` re-verifies that.
"""

from __future__ import annotations

import json
import os

import numpy as np

from snore_detect.aggregate import decide_night
from snore_detect.audio_features import scalar_features
from snore_detect.conditioning import resample_to

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(HERE, "model.onnx")
UBJ_PATH = os.path.join(HERE, "model.ubj")


class SnoreDetector:
    """Scores 10-second windows of audio, and a night of those scores."""

    def __init__(self, model_path: str | None = None, meta_path: str | None = None,
                 backend: str = "auto"):
        self.meta = json.load(open(meta_path or os.path.join(HERE, "model_meta.json")))
        self.threshold = float(self.meta["threshold"])
        self.window_s = float(self.meta["window_s"])
        self.fs = float(self.meta["fs_hz"])
        self.n_samples = int(self.window_s * self.fs)
        self.feature_names = list(self.meta["feature_names"])
        self._session = None
        self._booster = None
        self.backend = self._load(model_path, backend)

    def _load(self, model_path: str | None, backend: str) -> str:
        if backend not in ("auto", "onnx", "xgboost"):
            raise ValueError("backend must be 'auto', 'onnx' or 'xgboost'")

        if backend in ("auto", "onnx"):
            path = model_path or ONNX_PATH
            if path.endswith(".onnx") and os.path.exists(path):
                try:
                    import onnxruntime as ort
                except ImportError:
                    if backend == "onnx":
                        raise
                else:
                    self._session = ort.InferenceSession(
                        path, providers=["CPUExecutionProvider"])
                    self._input = self._session.get_inputs()[0].name
                    return "onnx"
            elif backend == "onnx":
                raise FileNotFoundError(path)

        import xgboost as xgb
        self._booster = xgb.Booster()
        self._booster.load_model(model_path or UBJ_PATH)
        return "xgboost"

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
    def predict(self, M: np.ndarray) -> np.ndarray:
        """(n, 55) features -> (n,) scores. The model only; no audio handling."""
        M = np.ascontiguousarray(np.atleast_2d(M), dtype=np.float32)
        if M.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, "
                             f"got {M.shape[1]}")
        if self._session is not None:
            out = self._session.run(None, {self._input: M})[1]
            if isinstance(out, list):            # list-of-dicts form
                return np.array([p[1] for p in out], dtype=np.float64)
            out = np.asarray(out)
            return (out[:, 1] if out.ndim == 2 else out).astype(np.float64)
        return np.asarray(self._booster.inplace_predict(M), dtype=np.float64)

    def score(self, X: np.ndarray, fs: float) -> np.ndarray:
        """(n, samples) -> (n,) scores in [0, 1]. A ranking, not a probability."""
        return self.predict(self.features(X, fs))

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
