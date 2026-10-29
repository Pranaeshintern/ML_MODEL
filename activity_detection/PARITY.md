# Model provenance and parity check

The shipped model is `model_acc_only.onnx`, 0.84 MB.

It is build v10 variant A' (trained 21 Aug 2026) stopped at **50 boosting rounds**
instead of 400. Boosting is sequential, so this is exactly the model that 50
rounds of the same training run produces - not an approximation of the 400-round
model. Exported 23 September 2026 with onnxmltools 1.16.0, opset 15.

## Why 50 rounds

Measured on 182,874 held-out windows from 45 people:

| Rounds | Size | Test acc / macro-F1 | Different study acc / macro-F1 | False exercise |
|---:|---:|---|---|---:|
| 400 | 5.82 MB | 0.8513 / 0.7826 | 0.8065 / 0.8342 | 41.0 min/day |
| 100 | 1.69 MB | 0.8432 / 0.7503 | 0.8263 / 0.8501 | 33.6 min/day |
| **50** | **0.84 MB** | **0.8383 / 0.7418** | **0.8455 / 0.8684** | **30.0 min/day** |

7x smaller costs 1.3 points of test accuracy, and is *better* on a study the
model has never seen (+3.9 accuracy, +3.4 macro-F1) with 11 fewer minutes a day
of false exercise. The extra rounds were fitting detail specific to the test
set, which is also where false workouts come from.

Walking F1 is nearly unchanged: 0.703 at 400 rounds, 0.688 at 50.

## ONNX parity

| Check | Result |
|---|---|
| 5,000 feature vectors, predicted class | identical, 5,000 / 5,000 |
| 5,000 feature vectors, class probabilities | max difference 2.7e-07 |
| 12 min recording end to end | same labels and same session |

Differences of 1e-07 are float rounding.

## Not yet measured

Ring accuracy for this 50-round model. The 400-round model scores 0.9672 on
HealthRing; this one has not been run against it. Measure before release.

## Repeating the check

```bash
pip install xgboost onnxruntime onnxmltools
# model_acc_only_r50.json in the development repo is the XGBoost form of this model
python - <<'PY'
import numpy as np, xgboost as xgb, onnxruntime as ort
m = xgb.XGBClassifier(); m.load_model("model_acc_only_r50.json")
s = ort.InferenceSession("model_acc_only.onnx", providers=["CPUExecutionProvider"])
X = np.random.default_rng(0).normal(size=(5000, 70)).astype(np.float32)
lab, prob = s.run(None, {"input": X})
print("labels match:", (m.predict_proba(X).argmax(1) == np.asarray(lab).ravel()).all())
print("max prob diff:", abs(m.predict_proba(X) - prob).max())
PY
```

An ONNX file dated 13 August 2026 did NOT match the model it claimed to be: it
came from an earlier training run and agreed on only 85% of predictions.
Re-export and re-run this check every time the model is retrained.
