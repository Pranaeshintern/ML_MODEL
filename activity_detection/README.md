# WUALT Activity Detection — runtime package

Names the wearer's activity from ring/wrist accelerometer data:
`static · walking · running · cycling · other`, plus grouped sessions.

## Install and run

```bash
pip install -r requirements.txt
python scripts/predict.py your_recording.csv
```

Output looks like this:

```
your_recording.csv  (12.0 min, 47 windows)
    walking      6.75 min
    running      6.00 min
    session   0.0- 12.0 min  other_cardio  -> ask the user what this was
```

## Use from your own code

```python
from wualt.models.inference import ActivityDetector

det = ActivityDetector("model", variant="acc_only")
res = det.predict(t, acc)           # t: seconds [N], acc: [N, 3] in g

res["label_names"]   # activity per 30 s window, one every 15 s
res["bouts"]         # stretches of one activity, with start/end/duration
res["sessions"]      # bouts grouped into workouts, each with session_label
```

## Input

Three accelerometer axes in g, 25-50 Hz, with timestamps.
The CSV needs a timestamp column and `ax,ay,az` (or `x,y,z`); m/s^2 is
detected and converted. Gaps over 2 s must start a new file.
Minimum length 30 s - shorter input produces no output.

## Sessions and "other cardio"

Only walking, running or cycling starts a session. Short pauses are absorbed;
2 min of stillness or 5 min of `other` ends it.

A stretch of 10 min or more of walking and running - each at least 20% of the
time, with only short pauses - is labelled `other_cardio` with
`needs_confirmation: true`. Ask the user what it was (intervals, football,
HIIT); the model has no class for those. Settings are at the top of
`wualt/temporal/sessions.py`.

## Contents

```
model/model_acc_only.onnx        trained model, 0.84 MB (accelerometer only)
model/model_meta_acc_only.json   feature list, class names, decoder settings
model/transitions_acc_only.npy   activity-to-activity transition matrix
wualt/                           the pipeline (conditioning, features, model, sessions)
scripts/predict.py               command-line runner
```

All three model files must stay together. Pairing one model with another's
transition matrix produces silently wrong output.

The model is ONNX, so it runs on onnxruntime alone - no XGBoost needed. It is
the 50-round form of build v10 variant A', chosen to fit 1 MB; see PARITY.md for
the size/accuracy measurements and the parity check.

Heart rate is not used by this model.

## Known limits

- Walking on a ring is detected far less reliably than on a wrist.
- Ring accuracy has not yet been measured for this 50-round model.
- Running and cycling have never been tested on ring hardware.
- About 30 min/day of movement is reported as exercise that was not.
- Confidence values are not calibrated - do not show them to users.
- Activities under a minute are mostly missed; switching between walking and
  running faster than ~1 min is not followed.

Full detail: ARCHITECTURE.md and RESULTS.md in the development repository.
