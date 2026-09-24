# -*- coding: utf-8 -*-
"""Conformance check: does this bundle reproduce the evaluated model?

    python selftest.py

Compares scores for 32 stored feature vectors against the values produced by the
model as evaluated. Any difference means the bundle will not behave like the
model in the validation report.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snore_detect import SnoreDetector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ref = np.load(os.path.join(HERE, "snore_detect", "reference.npz"))

det = SnoreDetector()
got = np.asarray(det._booster.inplace_predict(ref["M"]), dtype=np.float64)
diff = np.abs(got - ref["score"]).max()

print(f"model         {det.meta['model']}, threshold {det.threshold:.4f}")
print(f"features      {len(det.feature_names)}")
print(f"windows       {ref['M'].shape[0]} reference vectors")
print(f"max diff      {diff:.3e}")

flags = got > det.threshold
print(f"agreement     {(flags == (ref['score'] > det.threshold)).mean():.1%} "
      f"of decisions match")
print(f"sanity        {flags[ref['y']].mean():.0%} of snoring windows flagged, "
      f"{flags[~ref['y']].mean():.0%} of non-snoring")

ok = diff < 1e-9
print("\nPASS" if ok else "\nFAIL - this bundle does not match the evaluated model")
sys.exit(0 if ok else 1)
