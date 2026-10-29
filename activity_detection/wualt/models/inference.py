"""Inference: raw sensor signal -> activity timeline.

Single entry point shared by evaluation and deployment, so the two cannot drift.
The whole chain is:

    raw acc (any rate, any orientation) [+ optional HR]
      -> L0 condition (resample 30 Hz, calibrate, gravity-aligned frame)
      -> 30 s windows @ 15 s hop
      -> features
      -> XGBoost posteriors
      -> abstention
      -> L3 Viterbi + minimum dwell
      -> bouts

Abstention matters more than accuracy for a fitness product: at a 6-9 % per-window
exercise false-positive rate, a day contains 90-130 fabricated active minutes.
Emitting `unknown` is strictly better than inventing a workout.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort

from wualt.features.extract import HOP_LEN, WIN_LEN, extract, window_view
from wualt.signal.conditioning import FS, align_hr, condition
from wualt.temporal import sessions, smooth

UNKNOWN = -1


class ActivityDetector:
    """Loads a trained model and runs the full chain."""

    def __init__(self, model_dir: str, variant: str = "acc_only"):
        """variant: 'acc_only' or 'acc_hr'.

        Both consume the same 77-column feature block; the acc-only variant simply
        never sees the hr_* columns. Selecting the variant at load time (rather than
        branching per window) keeps a single, testable code path.
        """
        meta_path = os.path.join(model_dir, f"model_meta_{variant}.json")
        if not os.path.exists(meta_path):  # pre-variant artefacts
            meta_path = os.path.join(model_dir, "model_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.variant = variant
        self.feature_names: list[str] = self.meta["feature_names"]
        self.classes: list[str] = self.meta["classes"]
        self.stickiness: float = self.meta.get("stickiness", 1.0)
        self.min_dwell_windows: int = self.meta.get("min_dwell", 2)
        self.uses_hr: bool = self.meta.get("uses_hr", False)

        # The model ships as ONNX, so inference needs only onnxruntime -- no
        # training framework on the device. The export is checked against the
        # trained model before release; see PARITY.md.
        model_file = self.meta.get("onnx_file", f"model_{variant}.onnx")
        self.session = ort.InferenceSession(
            os.path.join(model_dir, model_file), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Prefer the transition matrix fitted alongside THIS variant. The untagged
        # file is whichever variant was trained last, so falling straight to it would
        # silently pair one model's posteriors with another model's transitions.
        tagged = os.path.join(model_dir, f"transitions_{variant}.npy")
        untagged = os.path.join(model_dir, "transitions.npy")
        trans_path = tagged if os.path.exists(tagged) else untagged
        self.trans = np.load(trans_path)
        self.prior = self.trans.mean(axis=0)

    # -- feature plumbing ----------------------------------------------------
    def _select(self, X: np.ndarray, names: list[str]) -> np.ndarray:
        """Reorder/subset extracted columns to exactly the trained feature order.

        Guards against a silent misalignment if the extractor gains a feature after
        the model was trained -- a mismatch there would be invisible and disastrous.
        """
        idx = {n: i for i, n in enumerate(names)}
        missing = [n for n in self.feature_names if n not in idx]
        if missing:
            raise ValueError(f"extractor is missing trained features: {missing}")
        return X[:, [idx[n] for n in self.feature_names]]

    def _proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities for each window, [N, 5]."""
        out = self.session.run(None, {self.input_name: X})[1]
        if isinstance(out, list):  # some exports return one dict per row
            out = np.array([[row[k] for k in sorted(row)] for row in out],
                           dtype=np.float32)
        return np.asarray(out, dtype=np.float64)

    # -- public API ----------------------------------------------------------
    def predict(
        self,
        t: np.ndarray,
        acc: np.ndarray,
        hr: np.ndarray | None = None,
        abstain_threshold: float = 0.0,
        abstain_margin: float = 0.0,
        smooth_output: bool = True,
    ) -> dict:
        """Run the chain on one continuous recording.

        `acc` must be in g, shaped [N, 3]. Returns per-window labels, the decoded
        timeline and the bout list.
        """
        if hr is not None and not self.uses_hr:
            # Silently ignoring supplied HR would be worse than saying so: the caller
            # believes heart rate is informing the result when it is not.
            hr = None
        cond = condition(t, acc, FS, calibrate=True)
        hr_grid = None
        if hr is not None:
            hr_grid = align_hr(cond["t"], np.asarray(t, dtype=np.float64), np.asarray(hr))
        X, names = extract(
            cond["mag"], cond["a_v"], cond["a_h"], cond["g_unit"], hr=hr_grid, fs=FS
        )
        if X.shape[0] == 0:
            # Same keys as a full result, so a caller never has to special-case a
            # recording that was too short for one window.
            return {"n_windows": 0, "t_start_s": [], "labels": [], "labels_raw": [],
                    "label_names": [], "confidence": [], "bouts": [], "sessions": [],
                    "abstain_rate": 0.0}

        Xs = self._select(X, names).astype(np.float32)
        proba = self._proba(Xs)
        raw = proba.argmax(axis=1).astype(np.int8)

        t_start = window_view(cond["t"], WIN_LEN, HOP_LEN)[: len(raw), 0]
        sid = np.zeros(len(raw), dtype=np.int32)  # one continuous recording

        if smooth_output:
            decoded = smooth.decode(
                proba, sid, self.trans, self.prior, stickiness=self.stickiness
            )
            decoded = smooth.min_dwell(decoded, sid, self.min_dwell_windows)
        else:
            decoded = raw

        final = decoded.copy()
        if abstain_threshold > 0 or abstain_margin > 0:
            top2 = np.sort(proba, axis=1)[:, -2:]
            weak = (top2[:, -1] < abstain_threshold) | (
                (top2[:, -1] - top2[:, -2]) < abstain_margin
            )
            final[weak] = UNKNOWN

        bout_list = [
            {**b, "label_name": self.classes[b["label"]] if b["label"] >= 0 else "unknown"}
            for b in smooth.bouts(final, t_start, window_s=WIN_LEN / FS)
        ]
        return {
            "n_windows": int(len(final)),
            "t_start_s": t_start.tolist(),
            "labels": final.tolist(),
            "labels_raw": raw.tolist(),
            "label_names": [
                self.classes[c] if c >= 0 else "unknown" for c in final
            ],
            "confidence": proba.max(axis=1).round(4).tolist(),
            "bouts": bout_list,
            # Bouts grouped into workouts, each named — including `other_cardio` for a
            # long walk/run mix, flagged for the user to confirm.
            "sessions": sessions.build_sessions(bout_list),
            "abstain_rate": float((final == UNKNOWN).mean()),
        }

    def summarize_session(self, result: dict) -> dict:
        """Minutes per activity — the shape the app would actually display."""
        out: dict[str, float] = {}
        for b in result.get("bouts", []):
            out[b["label_name"]] = out.get(b["label_name"], 0.0) + b["duration_s"] / 60.0
        return {k: round(v, 2) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}
