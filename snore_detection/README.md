# Snore Detection — deployment bundle

Feature M14 / Night Sounds · model `snore_v01` · commit `d0b09ff`

Inference only. Everything needed to score audio, and nothing else — no training
code, no datasets, no evaluation scripts.

## Check it first

```
pip install -r requirements.txt
python selftest.py
```

Must print `PASS`. It compares this bundle's scores against the model as
evaluated. A failure means the bundle will not behave like the validation report.

## Use

```python
from snore_detect import SnoreDetector
import numpy as np

det = SnoreDetector()

# one 10-second window
r = det.score_window(audio, fs=16000)
# -> {'score': 0.9933, 'snoring': True}

# a longer recording
s = det.score_stream(audio, fs=16000)
# -> {'t_start': [...], 'score': [...], 'snoring': [...]}

# a night of window results -> one verdict
v = det.decide_night(s['t_start'], s['snoring'])
# -> v.state == 'detected' | 'not_detected'
```

## Input contract

| | |
|---|---|
| Window | 10.0 seconds exactly |
| Rate | 4,000 Hz to the model; the library resamples from any rate |
| Channels | mono (`score_stream` mixes down) |
| Format | linear PCM float. **Never** MP3, AAC or Opus |

Two things must hold or the input is silently wrong:

1. **Unprocessed capture.** Android and iOS apply automatic gain control and
   noise suppression by default. AGC rewrites the recording level so loudness
   stops meaning anything; noise suppression removes the steady room tone the
   model measures snoring against. Request `UNPROCESSED` on Android and
   measurement mode on iOS, and confirm the handset honours it.
2. **No compression anywhere in the chain.** Lossy codecs discard exactly the
   quiet detail the model uses.

## Output contract

**Window:** a score in [0, 1] and a yes/no at threshold `0.699`.

**Night:** `detected` or `not_detected`. Detected when snoring was observed on
two separate occasions, or once for 600 s or more. Consecutive positives count
as one occasion.

**Nothing else is produced** — no count, no duration, no loudness, no severity.
The microphone hears a small fraction of the night, so episodes cannot be
counted or timed; recorded loudness depends on where the phone lies; and
severity is a clinical claim outside this feature's scope.

## Limits you must not design around

- **The score is a ranking, not a probability.** The model is not calibrated.
  Never display it as a percentage or a confidence.
- **The threshold does not transfer.** 0.699 was set on clinical room-microphone
  recordings. On unrelated audio, recall at this threshold fell to 0.44. It will
  need re-deriving for phone conditions.
- **No phone recordings were used.** Every result rests on clinical microphones.
  Placement, distance, a muffled or face-down phone, and handset variation are
  all untested.
- **Shared beds are untested.** One microphone cannot establish whose snore it is.
- **Accuracy is lowest for light snorers**, who are most of a consumer population.

The model validation report records a **FAIL** verdict on both tracks and a G5
decision of ITERATE. This bundle is for integration work and a pilot, not release.

## Contents

```
snore_detect/
  detector.py        the entry point
  conditioning.py    L0 - resample, band-pass, envelope
  periodicity.py     L1 - breathing rhythm
  burst.py           L1 - burst duration
  audio_features.py  L1 - the 55 features
  aggregate.py       L3 - night verdict
  model.ubj          400 trees, portable XGBoost booster
  model_meta.json    threshold, window, feature order
  reference.npz      32 vectors for the self-test
selftest.py
server.py            optional reference HTTP service
requirements.txt
```

No scikit-learn and no joblib: the model ships as a portable booster, so it is
not tied to a library version.

## Size

Model 862 KB on disk, ~254 KB gzipped. Runtime memory is one window —
40,000 floats, about 160 KB — plus the loaded trees. L0 processes in 1-second
frames, so a full window is never held at once.

No on-device measurement has been made. Latency, memory and energy on a handset
are all still open.
