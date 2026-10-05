# WUALT Sleep — model handoff

Three trained models in two forms, and the code that serves them.

**The models run on the paired phone** (D-009, decided 01 Oct 2026). So the ONNX graphs in
`artifacts/onnx/` are what ships into the app, and the Python package is the **reference
implementation and parity oracle** — the thing a native port is checked against, not the
thing that runs.

Contacts: admin1@wualt.com · Pipeline version `0.1.0` · Feature dictionary v2

---

## 1. What is in this package

**For the phone — ONNX**

```
artifacts/onnx/layer1_onset.onnx           146 KB   onset booster, 131 inputs
artifacts/onnx/layer1_calibrator.json        2 KB   isotonic breakpoints (NOT a graph)
artifacts/onnx/layer2_stage.onnx           668 KB   staging TCN, dynamic night length
artifacts/onnx/waso_epoch_scorer.onnx      279 KB   P(wake) per 30 s, 26 inputs
artifacts/onnx/waso_band_head_20min.onnx    51 KB   P(WASO > 20 min)
artifacts/onnx/waso_band_head_45min.onnx    52 KB   P(WASO > 45 min)
artifacts/onnx/waso_band_head_90min.onnx    52 KB   P(WASO > 90 min)
artifacts/onnx/export_report.json                   every parity measurement below
```

**Reference implementation — Python**

```
artifacts/layer1/model.joblib      360 KB   onset, XGBoost + isotonic calibrator
artifacts/layer1/model.meta.json           feature order, 131 inputs, CV numbers
artifacts/layer2/model.pt          667 KB   staging, TCN weights
artifacts/layer2/model.meta.json           feature order, mu/sd, CV numbers
artifacts/waso/model.joblib        982 KB   WASO: epoch scorer + 3 band heads
artifacts/waso/model.meta.json             feature orders, band edges, grid rate
config/features.yaml                       the 32-feature dictionary (v2)
config/default.yaml                        thresholds, timeouts, versions
src/wualt_sleep/                   255 KB   the inference package, 29 modules
```

Models and what is needed to run them, nothing else. No tests, no training or export
pipelines, no example scripts, and **not the Sleep Score** — that is a separate
distribution and is not a model. §8 lists everything left out and how to get it.

ONNX total **1.25 MB** across seven files; Python total **2.0 MB** across six. A
`.meta.json` is not optional on the Python side — the weights alone have no feature order
and no normalisation constants, and loading refuses without it. On the ONNX side the
feature order lives in the same `.meta.json`, so **ship it with the graphs**: a graph that
is handed its 26 or 131 inputs in a different order returns a confident wrong answer
rather than an error.

## 2. Requirements

**To run the ONNX graphs — which is what the phone does — you need `onnxruntime` and
nothing else.** No Python package, no pickles, no version pinning. Everything in §3.

To run the Python reference implementation, put `src/` on the path and install:

```bash
pip install "numpy>=1.26" "scipy>=1.12" "scikit-learn==1.9.0" \
            "xgboost==3.4.0" "torch>=2.6" joblib pyyaml onnxruntime
export PYTHONPATH=src        # no pyproject here; this package is not pip-installable
python -c "import wualt_sleep.runtime.adapters as a; print(a.load_shipped.__doc__)"
```

Pinned versions this was built against. Both `.joblib` bundles are **pickles**, so a
different scikit-learn or XGBoost major version may refuse to load them or, worse, load
them and behave differently. The ONNX graphs have no such coupling:

| package | version | needed for |
|---|---|---|
| onnxruntime | 1.30.0 | the ONNX graphs — the only hard requirement |
| Python | 3.14.3 | the reference implementation |
| numpy | 2.5.2 | " |
| scikit-learn | **1.9.0 exactly** | unpickling the WASO bundle |
| xgboost | **3.4.0 exactly** | unpickling the onset bundle |
| torch | 2.13.0+cpu | loading `layer2/model.pt` |
| scipy, joblib, pyyaml | current | " |

`polars` and `pyarrow` are ingest-only and are not imported by anything here.

**The Sleep Score is not in this package.** `core/session.py` imports it if present and
degrades cleanly if not: a night still streams, the session record is still complete, and
`sleep_score` comes back `null` — already a legal state, since no score is emitted during
a user's first 14 nights either. Calling `score_from_record` without it raises and names
the missing distribution rather than returning a number.

## 3. ONNX for the phone

Produced by `pipelines/20_export_onnx.py`, which is in the repo, not in this package —
ask if you want to re-run the conversion. Opset 17 for the
TCN (it needs `LayerNormalization`), opset 15 for the tree models — those use `ai.onnx.ml`
operators whose version is independent of the base opset, and the XGBoost converter
refuses anything higher. ONNX Runtime Mobile 1.17+ covers both. All graphs take **float32**.

### What converted

| graph | from | parity against Python |
|---|---|---|
| `layer1_onset.onnx` | `XGBClassifier` | max \|Δp\| **3.3e-07** over all 9,743 rows |
| `layer2_stage.onnx` | TCN `state_dict` | max \|Δp\| **1.4e-06**; sleep-stage argmax agrees on 1,110/1,110 rows across 40 nights |
| `waso_epoch_scorer.onnx` | `HistGradientBoostingClassifier` | max \|Δp\| **0.102**, mean 5.2e-04 — see the caveat |
| `waso_band_head_*.onnx` | 3 × `GradientBoostingClassifier` | max \|Δp\| **8.1e-08** |

**Layer 1's NaN rows were checked separately, because that is the one estimator where NaN
reaches a model.** 8,346 of the 9,743 rows carry at least one NaN, the trees split on
missingness directly, and a converter that silently flipped the default direction would
agree on clean data and fail on exactly the nights where a PPG burst dropped. On the
NaN-only subset the error is still 3.3e-07, and **the 0.5 and 0.7 gate decisions disagree
on zero rows** — so the state machine fires identically.

Everything else is NaN-free by construction: Layer 2 standardises then zero-fills, and both
WASO stages read `context_matrix` / `night_features` output, which ends in `nan_to_num`.

### Graph inputs and outputs

| graph | input | outputs |
|---|---|---|
| `layer1_onset.onnx` | `rows` float32 [N, 131] | `label`, `probabilities` [N, 2] |
| `layer2_stage.onnx` | `rows` float32 [1, T, 56] | `logits` [1, T, 4] |
| `waso_epoch_scorer.onnx` | `ctx` float32 [N, 26] | `label`, `probabilities` [N, 2] |
| `waso_band_head_*.onnx` | `night` float32 [N, 40] | `label`, `probabilities` [N, 2] |

Three things to get right:

**Read `probabilities[:, 1]`, never `label`.** Every threshold in this system is custom —
0.5 and 0.7 for the onset gates, 0.40 for the merge, 0.5 for each band crossing. `label`
bakes in an implicit 0.5 argmax, which silently replaces the merge threshold and
over-reports sleep.

**Layer 2 emits logits, not probabilities.** Apply softmax over the last axis yourself,
then drop column 0 — it is the unsupervised Wake logit — and renormalise the remaining
three, exactly as `merge_hypnogram` does.

**Layer 2's batch dimension is fixed at 1 and `T` is dynamic.** That is deliberate: the
dilated convolutions read across the sequence, so batching two nights together would let
one night's rows inform the other's. One night per call.

### The one caveat: the epoch scorer is not bit-exact

0.102 is not float noise. `HistGradientBoostingClassifier` bins its features and the
converter writes split thresholds as float32, and several epoch features are discrete —
`still_frac` is a count out of 30, `n_moves` is an integer — so values land exactly on
thresholds and a single tree takes the other branch. The same model fed float32 inputs in
Python disagrees with itself by 0.092, so this is threshold sensitivity in the model, not
a bug in the conversion. A double-precision export was tried; ONNX Runtime rejects
`TreeEnsembleClassifier` with double outputs.

Measured consequences, whole corpus:

| | |
|---|---|
| epochs whose wake decision at 0.40 flips | **116 / 324,237** (0.036%) = 0.16 min/night |
| nights whose **band** changes, ONNX end to end | **3 / 353** (0.85%) |
| direction | all three move exactly one band, all upward |

The band figure is the one that matters, and it is only visible end to end: feeding the
converted heads the *Python* scorer's night vector gives 60/60 identical, because the
scorer's drift is what shifts the summary the heads then read. `parity_chain` in the export
script runs scorer → summary → heads → monotone → median in ONNX and compares the band a
user would see. Treat 3/353 as the implementation's own error floor and keep it out of any
model-accuracy claim.

### What stays in its current form

**The isotonic calibrator** is a step function over breakpoints. There is no faithful ONNX
operator, so it ships as `layer1_calibrator.json`: interpolate `y` over `x`, clipped at
both ends, applied to the booster's class-1 probability. **It cannot be skipped** — the
state machine gates on absolute 0.5 and 0.7, and an uncalibrated score crosses them at the
wrong time.

**Everything between the models is still Python here and has to be written natively.** No
part of it is in any graph:

- the 50 Hz resample onto a uniform grid
- the 13 epoch features — including an FFT for `bp_lo/bp_mid/bp_hi` and gravity-vector
  angles for the posture pair
- `context_matrix`: 13 → 26, rolling means over 5 and 21 epochs, a rolling max, and
  activity divided by the night's own median
- `night_features`: 40 numbers, with quantiles and run-length logic, then standardised
  with the `mu`/`sd` in the WASO `.meta.json`
- the 131-input onset vector: two backward lags, their deltas, and a running
  `sustained_still_rows` counter, assembled across rows. **NaN must survive into the
  graph** — do not zero-fill it
- the state machine (0.5 / 0.7, two-row confirm, six-hour fallback)
- the 0.40 merge and its "wake if more than half of a row's 30 epochs are wake" rule
- the band readout: `minimum.accumulate` over the three cumulative probabilities, then
  the count of those ≥ 0.5
- both refusal gates: coverage < 0.5 and fewer than 120 post-onset epochs

That is roughly 500 lines, and it is the majority of the port. Build it one stage at a
time against the Python modules named above — `accel_epochs.py`, `waso.py`,
`state_machine.py`, `merge.py` — and check each stage's numbers before wiring the graphs
together. §4 gives the call sequence those modules expect.

## 4. The call sequence

```python
from wualt_sleep.config import load_config
from wualt_sleep.featuredict import load_feature_dictionary
from wualt_sleep.runtime.adapters import load_shipped
from wualt_sleep.runtime.pipeline import SleepPipeline
from wualt_sleep.schema import Row

cfg, fd = load_config(), load_feature_dictionary()
m = load_shipped(fd, "artifacts")                    # all three artifacts, one call

pipe = SleepPipeline(cfg, fd,
                     onset_detector=m.onset, stage_classifier=m.stage,
                     wake_detector=m.wake, band_estimator=m.band,
                     subject_id=uid, session_id=sid)

for row in night:                                     # one 15-minute row at a time
    out = pipe.push_row(row, accel_epochs=epochs_for_this_row)   # (30, 13) or None
    if out.onset_event:  ...                           # fires once
    if out.degraded:     ...                           # "no_motion" / "no_wake_channel"

record = pipe.finalise(wake_time=t_wake, history_nights=n)
hypnogram = pipe.epoch_hypnogram()                    # (E,) int8, 30-second stages
```

`load_shipped` is one call on purpose: loading two of the three and serving a night
without the wake channel under-reports wake by roughly 12 minutes.

Streaming, one row at a time, is the real path — that is how it runs on-device.
`push_row` decides internally which layer runs; the app does not choose.

### What the app must build

**A row** is 15 minutes: `Row(subject_id, start, values, feature_version, n_valid_epochs,
quality)` where `values` is the **32 features of the dictionary, in dictionary order**.
Order is validated on construction. Each model then selects its own columns by name —
the app never maps columns itself. NaN is legal and meaningful; do not impute.

**Epoch features** are the `(30, 13)` block of 30-second accelerometer features covering
that row, in `ACCEL_EPOCH_FEATURES` order. They must be computed on a **50 Hz uniform
grid** (`wualt_sleep.data.accel_epochs.GRID_HZ`). Activity counts scale with sample rate,
so a night resampled to any other rate feeds the WASO model counts on a different scale
and it returns a confident wrong band. The artifact records `grid_hz: 50.0`; changing that
number invalidates it.

Pass `accel_epochs=None` only when the accelerometer genuinely produced nothing for that
row. The wake channel then cannot run for it and the row is flagged degraded.

## 5. The three models

### Layer 1 — sleep onset (`artifacts/layer1`)

XGBoost, causal, with an isotonic calibrator fitted out-of-fold. Returns P(asleep) for
one 15-minute row.

- **131 inputs** from 26 base features: the row, then lag-1, delta-1, lag-2, delta-2, then
  a running `sustained_still_rows` counter. `runtime.adapters.OnsetAdapter` assembles
  these across calls; do not build the vector by hand. Lag history must not cross nights
  (`adapter.reset()`).
- Calibration is not cosmetic. The state machine gates on **absolute** probability: 0.5 to
  enter `CANDIDATE_ONSET`, **0.7** to confirm, **2 consecutive rows** to fire. An
  uncalibrated probability makes those thresholds meaningless.
- If nothing confirms within **6 hours** of bedtime, a rule-based fallback fires and the
  event carries `via_fallback: true`.
- `T_onset` is the confirmation instant; `onset_interval` reaches back 15 minutes.

### Layer 2 — sleep staging (`artifacts/layer2`)

Non-causal TCN, dilations 1/2/4, over the whole night. 28 features standardised with the
stored mu/sd, concatenated with a binary availability mask → 56 inputs.

- Returns `(T, 4)` in `wake, light, deep, rem` order, **but column 0 is meaningless.** The
  artifact is `no_wake: true` — the Wake logit was never supervised. `merge_hypnogram`
  drops it and renormalises the three sleep columns. Nothing else may read it as P(wake).
- The whole night is re-decoded on every row, because later rows legitimately revise
  earlier ones. A night is ~40 rows, so this is cheap.
- **Sleep/wake is not this model's job.** Post-onset wake bouts average 2.8 minutes and a
  15-minute majority vote needs 7.5 to flip one, so the cardiac channel is structurally
  blind to most of them. The accelerometer owns that decision.

### Layer W — WASO (`artifacts/waso`)

One bundle, two halves, both served from the same file.

- **Epoch scorer** (HistGradientBoosting) → P(wake) per 30 seconds, from accelerometry
  alone. It consumes **26** inputs: the 13 stored epoch features plus temporal context
  (rolling means over 5 and 21 epochs, a rolling max, and activity self-normalised against
  the night's median). A single still epoch looks identical asleep or lying awake; the
  neighbourhood is what separates them. `WakeAdapter` builds this transform.
- **Band model** — three cumulative binary heads at **20 / 45 / 90 minutes**, forced
  monotone and combined by median. Runs once, at finalisation, over the post-onset window.

| band | shown as |
|---|---|
| `minimal` | up to 20 min |
| `some` | 20–45 min |
| `notable` | 45–90 min |
| `high` | over 90 min |

**Show the band, not the minutes.** `record.waso_min` exists and is used inside the Sleep
Score, but on its own it is not defensible: 12.3 minutes of error against a true mean of
21.4, where simply printing the population median gives 12.7. Two human scorers of the
same EEG differ by 5.6. The band is what reaches a screen.

The merge happens at P(wake) **0.40**, and a 15-minute row is called wake when more than
half of its 30 epochs are wake.

## 6. Nulls and degraded modes

Every one of these is a state to display, not an error to swallow. None of them is ever
replaced by a plausible-looking number.

| condition | what happens |
|---|---|
| onset never confirmed | `finalise()` returns `None` — no session record at all |
| a row has no accelerometer data | `out.degraded == "no_motion"`; that row's wake comes from the stage model alone |
| no wake detector passed | `out.degraded == "no_wake_channel"` |
| motion covers < 50% of the session | `outcome: DEGRADED_NO_MOTION` on the record |
| accelerometer coverage < 0.5, or < 1 h post-onset | `waso_band` is **null** |
| fewer than 14 nights of history | `sleep_score` is `null` (pass `history_nights`) |

`provenance` on every record stamps which model version produced it, the feature
dictionary version and the pipeline version. Read it before comparing two records.

## 7. Numbers

All figures are out-of-fold under `GroupKFold` on **subject**, so no subject appears in
both train and test. Two independent scorings of the same EEG agree 80.8% of the time,
which is the practical ceiling for any of them.

**Layer 1 — onset**, 353 nights:

| | |
|---|---|
| AUC | 0.971 |
| exact row | 44.5% |
| within ±1 row | 82.2% |
| within ±15 min | 56.7% |
| within ±30 min | 85.8% |
| median error | 13.0 min |
| fired | 353 / 353 nights |
| fires early | 22.1% of nights |

**Layer 2 — staging**, 26,614 rows / 147 subjects, scored post-onset on the three sleep
stages:

| | |
|---|---|
| accuracy | 0.633 (majority baseline 0.562) |
| macro-F1 | 0.612 |
| kappa | 0.386 |
| F1 Light / Deep / REM | 0.694 / 0.624 / 0.517 |

Wake is not scored here — the accelerometer channel owns it.

**Layer W — WASO band**, 346 nights / 147 subjects:

| | this artifact | five-seed mean | constant baseline | count rule |
|---|---|---|---|---|
| within one band | 90.8% | 91.2% | 74.9% | 85.6% |
| exact band | 54.9% | 55.1% | 49.4% | 53.4% |
| Spearman vs true WASO | 0.635 | 0.645 | 0.000 | 0.491 |

The model never places a high-WASO night in the lowest band.

## 8. Not included

Deliberately trimmed to the models and their runtime. Everything below exists in the
project and can be sent on request.

- **No Sleep Score.** `wualt-sleep-score` is a separate distribution and not a model. See
  §2 for how the pipeline behaves without it.
- **No tests.** The suite — 248 tests covering the loaders, the feature contract, the
  state machine, the merge and the WASO bands — stays in the repo.
- **No example scripts, no training or export pipelines, no analysis scripts.**
- **No `pyproject.toml`.** This package is not pip-installable; §2 lists the dependencies
  directly.
- **No training data.** DREAMT and BIDSleep are third-party corpora under their own
  licences and are not redistributable, and nor are the processed tables built from them.
- **Nothing measured on device.** The graphs load and score correctly under ONNX Runtime on
  desktop CPU. Latency, memory and battery on target hardware are unmeasured, and no app
  has loaded them yet.
- **No native feature code.** §3 lists the ~500 lines that have to be written in
  Swift/Kotlin. This package has only the Python reference.
- **No quantisation.** Everything is float32. Only the TCN would benefit much — 169,204
  dense weights, so int8 would take its 668 KB to roughly 170 KB. The four tree graphs are
  most of the remaining size and do not compress that way.
