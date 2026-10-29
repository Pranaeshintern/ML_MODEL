"""Post-training probability calibration (§6, §8): isotonic, per class.

Calibration is load-bearing here for two reasons beyond honest confidence numbers:

  1. The state machine gates Layer 1 on absolute thresholds (0.5, 0.7). Uncalibrated
     probabilities make those numbers arbitrary.
  2. The HMM treats Layer 2's output as an emission. Systematically over-confident
     probabilities distort the whole decode, not just one row.

MUST be fitted on held-out data. Fitting isotonic on the same rows the classifier
trained on learns the classifier's overfit, not its miscalibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..errors import NotFittedError

__all__ = ["IsotonicCalibrator"]


class IsotonicCalibrator:
    """One isotonic regressor per class, renormalised to a simplex after transform.

    Per-class one-vs-rest isotonic does not preserve the sum-to-one constraint, so
    the outputs are renormalised. Where every class maps to ~0 the renormalisation is
    undefined; those rows fall back to the uncalibrated probabilities rather than to
    a uniform distribution, which would silently discard real evidence.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self._models: list[IsotonicRegression] | None = None

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> IsotonicCalibrator:
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(targets, dtype=int).ravel()
        if p.shape[0] != y.shape[0]:
            raise ValueError(f"shape mismatch: {p.shape[0]} rows vs {y.shape[0]} labels")

        models = []
        for k in range(self.n_classes):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            hits = (y == k).astype(np.float64)
            # A class absent from the calibration split has no signal to fit. Use an
            # identity-like mapping so it is passed through untouched rather than
            # being flattened to zero, which would delete the class entirely.
            if hits.sum() == 0 or hits.sum() == hits.size:
                iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            else:
                iso.fit(p[:, k], hits)
            models.append(iso)
        self._models = models
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self._models is None:
            raise NotFittedError("IsotonicCalibrator.transform before fit")
        p = np.asarray(probabilities, dtype=np.float64)
        out = np.column_stack([m.predict(p[:, k]) for k, m in enumerate(self._models)])

        total = out.sum(axis=1, keepdims=True)
        degenerate = (total <= 1e-12).ravel()
        out = np.divide(out, total, out=np.zeros_like(out), where=total > 1e-12)
        out[degenerate] = p[degenerate]
        return out

    def fit_transform(self, probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return self.fit(probabilities, targets).transform(probabilities)
