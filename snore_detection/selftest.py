# -*- coding: utf-8 -*-
"""Conformance check: does this bundle reproduce the evaluated model?

    python selftest.py

Scores 32 stored feature vectors and compares against the values the model
produced when it was evaluated. If both backends are installed it also compares
them against each other, since production runs ONNX and every number in the
validation report came from XGBoost.

A failure means this bundle will not behave like the validation report.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snore_detect import SnoreDetector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ref = np.load(os.path.join(HERE, "snore_detect", "reference.npz"))
M, want, y = ref["M"], ref["score"], ref["y"]

# Reference scores were produced by XGBoost in float64; ONNX runs float32, so a
# few 1e-7 of rounding is expected and harmless. What must not move is any
# decision at the threshold.
TOL = 1e-5

det = SnoreDetector()
got = det.predict(M)
diff = float(np.abs(got - want).max())
flags = got > det.threshold
flips = int((flags != (want > det.threshold)).sum())

print(f"backend       {det.backend}")
print(f"model         {det.meta['model']}, threshold {det.threshold:.4f}")
print(f"features      {len(det.feature_names)}")
print(f"windows       {M.shape[0]} reference vectors")
print(f"max diff      {diff:.3e}   (tolerance {TOL:.0e})")
print(f"agreement     {flips} decision flips")
print(f"sanity        {flags[y].mean():.0%} of snoring windows flagged, "
      f"{flags[~y].mean():.0%} of non-snoring")

ok = diff < TOL and flips == 0

# ---- cross-check the two backends, when both are present ----
try:
    other = "xgboost" if det.backend == "onnx" else "onnx"
    alt = SnoreDetector(backend=other)
except Exception as e:  # noqa: BLE001
    print(f"\ncross-check   skipped ({other} unavailable: {type(e).__name__})")
else:
    d = float(np.abs(alt.predict(M) - got).max())
    f = int(((alt.predict(M) > det.threshold) != flags).sum())
    print(f"\ncross-check   {det.backend} vs {other}: max diff {d:.3e}, "
          f"{f} decision flips")
    ok = ok and d < TOL and f == 0

print("\nPASS" if ok else "\nFAIL - this bundle does not match the evaluated model")
sys.exit(0 if ok else 1)
