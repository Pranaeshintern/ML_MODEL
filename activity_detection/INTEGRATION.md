# Integration guide — Python backend

For the team wiring this model into a service. Everything below was measured on
this package, not estimated.

---

## 1. Install and smoke test

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, onnxruntime
```

```python
import numpy as np
from wualt.models.inference import ActivityDetector

det = ActivityDetector("model", variant="acc_only")

fs = 50.0
t = np.arange(fs * 120) / fs                 # 2 minutes of timestamps, seconds
acc = np.zeros((len(t), 3)); acc[:, 2] = 1.0  # device at rest, z pointing down

res = det.predict(t, acc)
print(res["label_names"][:5])   # ['static', 'static', ...]
```

If that prints activity names, the install is good.

## 2. Load the model once

`ActivityDetector("model")` takes **0.22 s** and holds the ONNX session. Create it
at service start-up and reuse it. Do not create one per request.

`predict()` is safe to call from several threads on one detector — tested with 4
concurrent calls. The ONNX session serialises internally, so for real parallelism
run several worker processes.

## 3. The one call you need

```python
res = det.predict(t, acc, hr=None,
                  abstain_threshold=0.0,   # see §7
                  abstain_margin=0.0,
                  smooth_output=True)      # leave True
```

| Argument | Type | Notes |
|---|---|---|
| `t` | float array `[N]` | Seconds from the start of **this** recording. Must increase. |
| `acc` | float array `[N, 3]` | In **g**, gravity included. At rest the total length is 1.0. |
| `hr` | ignored | This model is accelerometer-only. |

**Units matter.** If your source is m/s², divide by 9.80665 first. At rest the
magnitude must be ~1.0, not ~9.8.

`t` is relative, so keep the wall-clock start time of each recording yourself and
add it back to every output timestamp.

## 4. What comes back

```python
{
  "n_windows":   239,                       # 30 s windows scored
  "t_start_s":   [0.0, 15.0, 30.0, ...],    # start of each window, seconds
  "label_names": ["walking", "walking", ...],  # one per window
  "labels":      [1, 1, ...],               # same, as class indices; -1 = unknown
  "labels_raw":  [1, 1, ...],               # before smoothing - diagnostics only
  "confidence":  [0.94, 0.91, ...],         # DO NOT show to users, see §7
  "abstain_rate": 0.0,                      # fraction of windows marked unknown
  "bouts":    [ ... ],                      # §4a
  "sessions": [ ... ],                      # §4b
}
```

Classes, in index order: `static, walking, running, cycling, other`.
`static` means still or standing. `other` is everything else — chores, driving,
eating, sport that is not walk/run/cycle.

### 4a. `bouts` — what to store

One unbroken stretch of a single activity. **This is the right thing to persist**,
and it is what the AZM module consumes.

```python
{"label": 1, "label_name": "walking",
 "t_start_s": 0.0, "t_end_s": 3600.0, "duration_s": 3600.0, "n_windows": 239}
```

Neighbouring bouts **overlap by 15 s** at each switch, because the last window of
one and the first of the next cover the same seconds. When totalling time per
activity, credit each bout only until the next one starts — otherwise every
switch is counted twice.

### 4b. `sessions` — what to show the user

Bouts grouped into one workout and named.

```python
{"t_start_s": 0.0, "t_end_s": 720.0, "duration_s": 720.0,
 "labels": ["walking", "running", "walking", "running"],
 "label_seconds": {"walking": 360.0, "running": 360.0},
 "session_label": "other_cardio",
 "needs_confirmation": True,
 "cardio_stretch": {"t_start_s": 0.0, "t_end_s": 720.0, "duration_s": 720.0}}
```

Grouping rules: only walking, running or cycling can open a session; a short pause
is absorbed; 2 min of stillness or 5 min of `other` ends it.

**`needs_confirmation: True` means ask the user.** It is set only for
`other_cardio`: 10 min or more of mixed walking and running, each at least 20% of
the time. The accelerometer cannot tell interval running from football from HIIT,
so the app should ask and replace the label with the answer. Treat any other
`session_label` as final.

## 5. Chunking a continuous day

**One call per continuous recording.** Split the input wherever the sensor has a
gap over 2 s and call `predict()` separately for each piece — a window spanning a
gap measures a rhythm that is not there.

Measured on this package (50 Hz input, one core):

| Input | Windows | Time | Peak memory |
|---|---:|---:|---:|
| 1 min | 3 | 0.08 s | — |
| 10 min | 39 | 0.11 s | — |
| 60 min | 239 | 0.39 s | 24 MB |
| 24 h (projected) | ~5,750 | ~12 s | ~550 MB |

A whole day in one call is fine on a server. If memory is tight, process in
chunks of a few hours — but see the warning below.

**Chunk boundaries cut sessions.** The model has no memory between calls, so a
workout split across two calls comes back as two sessions, and a 10-minute
walk/run mix split 6 + 4 will not be tagged `other_cardio`. Two ways to handle it:

1. Call once per continuous recording, however long. Simplest, and correct.
2. If you must chunk, re-run `build_sessions()` over the combined bout list:

```python
from wualt.temporal.sessions import build_sessions

all_bouts = []
for chunk in chunks:                      # each with its own offset, in seconds
    r = det.predict(chunk.t, chunk.acc)
    all_bouts += [{**b,
                   "t_start_s": b["t_start_s"] + chunk.offset_s,
                   "t_end_s":   b["t_end_s"]   + chunk.offset_s} for b in r["bouts"]]

sessions = build_sessions(all_bouts)      # correct sessions across chunks
```

Do this only for chunks of one continuous recording. Never join bouts across a
real sensor gap.

## 6. Edge cases, all verified

| Input | Behaviour |
|---|---|
| Under 30 s | `n_windows: 0`, empty lists, no error. Handle it; don't treat it as a failure. |
| 45 s | 2 windows. Partial tail is dropped. |
| Gap over 2 s inside the input | Not handled for you — **you** must split the input. |
| Device not worn | Magnitude near 0 instead of 1. Not detected for you; filter these before calling. |
| m/s² passed as g | No error, nonsense output. Check the resting magnitude is ~1.0 at ingest. |
| `hr` supplied | Silently ignored — this variant does not use heart rate. |

## 7. Confidence and abstention

`confidence` is the model's raw probability. **It is not calibrated.** On a study
the model had not seen, it says 95% and is right 81% of the time. Do not show it
to users and do not threshold product behaviour on it.

`abstain_threshold` and `abstain_margin` default to 0 (off). Setting them makes
low-confidence windows return `unknown` (label `-1`), which then breaks sessions.
Leave them off unless you are running an experiment.

## 8. Known limits to design around

- **About 30 minutes a day of movement is reported as exercise that was not** —
  mostly housework with a repetitive arm motion. Do not build a feature that
  claims a daily exercise total without heart rate confirming it. The AZM module
  filters most of this out by requiring an elevated heart rate.
- **Walking on a ring is detected far less reliably than on a wrist.** Ring
  accuracy for this 50-round model is 0.9595 overall but walking F1 is 0.451 —
  it finds 23 of 62 known walking windows.
- **Activities under about a minute are mostly missed.** Under 1 minute, roughly
  a quarter are caught; over 10 minutes, most are.
- **Latency is up to 45 s** before an activity is confirmed: 30 s to fill a
  window, 15 s to confirm it. Nothing can be reported faster than that.
- **Running and cycling have never been tested on ring hardware** — no public
  ring dataset contains either.

## 9. Checklist before going live

- [ ] Resting magnitude at ingest is ~1.0 g (units are right).
- [ ] Input split at every sensor gap over 2 s.
- [ ] Not-worn stretches filtered out before `predict()`.
- [ ] Recordings under 30 s handled as "no result", not as an error.
- [ ] Bout time credited only until the next bout starts (no double counting).
- [ ] `needs_confirmation` drives a question to the user, not a silent label.
- [ ] `confidence` never shown and never used as a threshold.
- [ ] Detector created once at start-up.
- [ ] Output timestamps converted back to wall clock, with the user's time zone
      kept for any per-day totals.

## 10. Questions this package cannot answer

Send these back to ML rather than guessing:

- Output JSON schema and versioning for your API — not yet defined (Form J item 11).
- What to do when signal quality is poor — scored but no behaviour defined.
- Behaviour on accelerometer failure, or on a model version mismatch — undefined.

Model details and the parity check: `PARITY.md`. How to run it: `README.md`.
