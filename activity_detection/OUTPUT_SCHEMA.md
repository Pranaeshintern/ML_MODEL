# Output schema — WUALT Activity Detection

What `ActivityDetector.predict()` returns. Every example here was produced by
running this package; see `example_output.json` for the same output as a file.

```python
res = det.predict(t, acc)     # one call per continuous recording
```

One call covers a whole recording. The return value is a dict of **arrays** —
one entry per 30 s window, a new window every 15 s — plus two grouped views of
the same information.

---

## Top level

| Field | Type | Meaning |
|---|---|---|
| `n_windows` | int | Number of 30 s windows scored. `0` if the input was under 30 s |
| `t_start_s` | float[] | Start of each window, seconds from the recording start |
| `labels` | int[] | Final class index per window. `-1` means unknown (only if abstention is enabled) |
| `label_names` | string[] | The same labels as names |
| `labels_raw` | int[] | Labels **before** smoothing. Diagnostics only — never show these |
| `confidence` | float[] | Top probability per window. **Uncalibrated, do not display or threshold** |
| `abstain_rate` | float | Fraction of windows marked unknown |
| `bouts` | object[] | Stretches of one activity. **Store these** |
| `sessions` | object[] | Workouts, named. **Show these** |

Classes, in index order:

| Index | Name | Means |
|---:|---|---|
| 0 | `static` | Still, sitting or standing |
| 1 | `walking` | |
| 2 | `running` | |
| 3 | `cycling` | |
| 4 | `other` | Everything else: chores, driving, eating, sport that is not walk/run/cycle |
| −1 | `unknown` | Model declined to answer (only when abstention is switched on) |

## `bouts`

```json
{ "label": 2, "label_name": "running",
  "t_start_s": 165.0, "t_end_s": 375.0, "duration_s": 210.0, "n_windows": 13 }
```

| Field | Type | Meaning |
|---|---|---|
| `label` | int | Class index |
| `label_name` | string | Class name |
| `t_start_s` | float | Start, seconds from the recording start |
| `t_end_s` | float | End of the bout's last window |
| `duration_s` | float | `t_end_s − t_start_s` |
| `n_windows` | int | Windows in the bout |

**Bouts overlap by 15 s at every switch.** The last window of one bout and the
first of the next cover the same seconds. In the example above, walking ends at
180.0 while running starts at 165.0.

When totalling time per activity, credit each bout only until the next one
begins, or every activity change is counted twice:

```python
ordered = sorted(bouts, key=lambda b: b["t_start_s"])
for i, b in enumerate(ordered):
    end = ordered[i + 1]["t_start_s"] if i + 1 < len(ordered) else b["t_end_s"]
    seconds[b["label_name"]] += max(0.0, min(b["t_end_s"], end) - b["t_start_s"])
```

Bouts are also the input the Active Zone Minutes module expects.

## `sessions`

```json
{ "t_start_s": 0.0, "t_end_s": 720.0, "duration_s": 720.0,
  "labels": ["walking", "running", "walking", "running"],
  "label_seconds": { "walking": 330.0, "running": 390.0 },
  "session_label": "other_cardio",
  "needs_confirmation": true,
  "cardio_stretch": { "t_start_s": 0.0, "t_end_s": 720.0, "duration_s": 720.0 } }
```

| Field | Type | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s`, `duration_s` | float | The session's span |
| `labels` | string[] | Bout labels in order, absorbed pauses included |
| `label_seconds` | object | Seconds per label, overlap already removed |
| `session_label` | string | One of the 5 class names, or `other_cardio`, or `""` if no exercise |
| `needs_confirmation` | bool | `true` only for `other_cardio` — **ask the user** |
| `cardio_stretch` | object or null | The walk/run stretch that triggered `other_cardio` |

How sessions are formed: only walking, running or cycling can open one; short
pauses are absorbed; 2 min of stillness or 5 min of `other` ends it. A session
never starts or ends on a pause.

`session_label` is `other_cardio` when a stretch lasts **10 min or more**,
contains **both walking and running** with each at least **20%** of the time, and
contains nothing else but short standing or `other` pauses. The accelerometer
cannot tell interval running from football from HIIT, so the app must ask and
replace the label with the user's answer. Any other `session_label` is final.

## Short or empty input

Input under 30 s returns the same keys with nothing in them. Handle it as "no
result", not as an error:

```json
{ "n_windows": 0, "t_start_s": [], "labels": [], "labels_raw": [],
  "label_names": [], "confidence": [], "bouts": [], "sessions": [],
  "abstain_rate": 0.0 }
```

## Timestamps

Every time is **seconds from the start of this recording**, not wall clock. Keep
the recording's start time yourself and add it back:

```python
wall = recording_start + timedelta(seconds=bout["t_start_s"])
```

---

## Not in this output

These do **not** exist, and if you see them the data did not come from this
package:

| Not returned | Why |
|---|---|
| Per-window probability vectors | Only the top value survives, as `confidence`. The full five numbers are used by the smoothing step and then discarded. Exposing them invites `argmax()`, which disagrees with `labels` and is less accurate |
| A single `label` / `labelName` pair per call | One call covers a recording, not one window |
| Inference timing (`inferenceMicros` etc.) | The model does not time itself. For reference: 0.39 s for 1 hour of 50 Hz input, 0.22 s to load the model |
| camelCase field names | Every field is snake_case, as listed above |

A format such as

```json
{ "label": 1, "labelName": "walking",
  "probabilities": [0.0256, 0.8972, 0.0076, 0.0262, 0.0435],
  "inferenceMicros": 180 }
```

is the **raw ONNX model output** for one window — what you get by loading the
`.onnx` file and feeding it 70 features directly. It bypasses
`conditioning.py`, `extract.py`, `smooth.py` and `sessions.py`, which is where
the smoothing, the bouts and the sessions come from. The result is a flickering
label every 15 seconds with no activity history and no `other_cardio` detection.

Mapping `predict()` output into your own API shape and field names is fine.
Re-implementing the pipeline around the raw ONNX file is not.

---

Running it: `README.md` · Integration: `INTEGRATION.md` · Model provenance:
`PARITY.md`
