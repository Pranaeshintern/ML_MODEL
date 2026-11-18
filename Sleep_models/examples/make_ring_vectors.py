"""Regenerate examples/ring_row_vectors.json — parity vectors for the native row builder.

    PYTHONPATH=src python examples/make_ring_vectors.py

Synthetic, so there is no licence question (the DREAMT and BIDSleep rows cannot be
redistributed). Each case gives the raw inputs and the outputs `data/ring_row.py`
produces from them: the 32 row values and the (30, 13) accelerometer epoch block.

The accelerometer is given as a short, exact recipe rather than 46,500 samples, so the
file stays small. Rebuild the samples natively from it:

    t_k   = t0 + from_s + k / rate_hz              for k = 0 .. (to_s - from_s) * rate_hz - 1
    g     = the gravity segment whose [from_s, to_s) contains (t_k - t0)
    a     = g + sum over movements m with m.from_s <= t_k - t0 < m.to_s of
            m.amp_g * sin(2*pi*m.freq_hz*(t_k - t0)) * m.axis_weights
    drop  samples whose (t_k - t0) falls inside any of `gaps`

t0 is the row start as Unix seconds. NN intervals and temperature are listed explicitly.
Compare with a tolerance of 1e-6 relative; the epoch FFT may differ in the last digits.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from wualt_sleep.data import ring_row as rr
from wualt_sleep.featuredict import load_feature_dictionary

OUT = Path(__file__).with_name("ring_row_vectors.json")


def nn_series(n: int, hr: float, rsa_s: float = 0.03, period_beats: float = 4.5,
              missed_at: int | None = None) -> list[float]:
    i = np.arange(n)
    nn = 60.0 / hr + rsa_s * np.sin(2 * math.pi * i / period_beats)
    nn = nn[np.cumsum(nn) <= 90.0]
    out = [round(float(v), 4) for v in nn]
    if missed_at is not None and missed_at + 1 < len(out):      # one dropped beat
        out[missed_at] = round(out[missed_at] + out.pop(missed_at + 1), 4)
    return out


def accel_samples(t0: float, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    n = int(round((spec["to_s"] - spec["from_s"]) * spec["rate_hz"]))
    rel = spec["from_s"] + np.arange(n) / spec["rate_hz"]
    a = np.zeros((n, 3))
    for seg in spec["gravity"]:
        m = (rel >= seg["from_s"]) & (rel < seg["to_s"])
        a[m] = seg["g"]
    for mv in spec["movements"]:
        m = (rel >= mv["from_s"]) & (rel < mv["to_s"])
        a[m] += (mv["amp_g"] * np.sin(2 * math.pi * mv["freq_hz"] * rel[m]))[:, None] \
            * np.asarray(mv["axis_weights"])
    keep = np.ones(n, bool)
    for lo, hi in spec["gaps"]:
        keep &= ~((rel >= lo) & (rel < hi))
    return t0 + rel[keep], a[keep]


FLAT = [{"from_s": -30, "to_s": 900, "g": [0.0, 0.0, 1.0]}]
CASES = [
    {"name": "awake_before_onset",
     "note": "first row of the session: moving all row, rising temperature, within-night "
             "reference (so both deltas are 0), no onset yet",
     "start": "2027-01-14T22:30:00+05:30", "session_start": "2027-01-14T22:30:00+05:30",
     "nights_before": 0, "onset_at": None, "hr_reference_bpm": None, "temp_reference_c": None,
     "nn_s": nn_series(110, 74.0, missed_at=30),
     "temp": [[60.0 * k, round(33.0 + 0.02 * k, 3)] for k in range(15)],
     "accel": {"rate_hz": 50.0, "from_s": -30, "to_s": 900, "gravity": FLAT, "gaps": [],
               "movements": [{"from_s": -30, "to_s": 900, "amp_g": 0.2, "freq_hz": 1.5,
                              "axis_weights": [1.0, 0.5, 0.2]}]}},
    {"name": "asleep_still_after_onset",
     "note": "personal baseline exists (night 4); onset confirmed at 23:15; one 20 s twitch",
     "start": "2027-01-14T23:30:00+05:30", "session_start": "2027-01-14T22:30:00+05:30",
     "nights_before": 3, "onset_at": "2027-01-14T23:15:00+05:30",
     "hr_reference_bpm": 62.0, "temp_reference_c": 33.4,
     "nn_s": nn_series(110, 57.0, rsa_s=0.05),
     "temp": [[60.0 * k, round(34.1 + 0.005 * k, 3)] for k in range(15)],
     "accel": {"rate_hz": 50.0, "from_s": -30, "to_s": 900, "gravity": FLAT, "gaps": [],
               "movements": [{"from_s": 400, "to_s": 420, "amp_g": 0.3, "freq_hz": 2.0,
                              "axis_weights": [0.3, 1.0, 0.1]}]}},
    {"name": "missing_ppg_and_temp_with_accel_gap",
     "note": "no beats and no temperature this row; accelerometer off for 5 minutes",
     "start": "2027-01-15T01:00:00+05:30", "session_start": "2027-01-14T22:30:00+05:30",
     "nights_before": 20, "onset_at": "2027-01-14T23:15:00+05:30",
     "hr_reference_bpm": 62.0, "temp_reference_c": 33.4,
     "nn_s": [], "temp": [],
     "accel": {"rate_hz": 50.0, "from_s": -30, "to_s": 900, "gravity": FLAT,
               "gaps": [[300, 600]], "movements": []}},
    {"name": "turn_over",
     "note": "gravity rotates 45 degrees halfway through the row, with a short movement",
     "start": "2027-01-15T02:15:00+05:30", "session_start": "2027-01-14T22:30:00+05:30",
     "nights_before": 20, "onset_at": "2027-01-14T23:15:00+05:30",
     "hr_reference_bpm": 62.0, "temp_reference_c": 33.4,
     "nn_s": nn_series(110, 60.0, missed_at=55),
     "temp": [[60.0 * k, round(34.0 - 0.01 * k, 3)] for k in range(15)],
     "accel": {"rate_hz": 50.0, "from_s": -30, "to_s": 900, "gaps": [],
               "gravity": [{"from_s": -30, "to_s": 450, "g": [0.0, 0.0, 1.0]},
                           {"from_s": 450, "to_s": 900, "g": [0.70710678, 0.0, 0.70710678]}],
               "movements": [{"from_s": 440, "to_s": 460, "amp_g": 0.4, "freq_hz": 1.0,
                              "axis_weights": [1.0, 0.2, 0.2]}]}},
]


def _num(x: float) -> float | None:
    return None if not np.isfinite(x) else float(round(float(x), 10))


def main() -> None:
    rr.check_against(load_feature_dictionary())
    out = {"generated_by": "examples/make_ring_vectors.py", "module": "wualt_sleep.data.ring_row",
           "row_features": list(rr.ROW_FEATURES), "status": rr.STATUS,
           "used_by": {k: list(v) for k, v in rr.USED_BY.items()},
           "epoch_features": ["act_mean", "act_p95", "jerk_rms", "mag_std", "still_frac",
                              "n_moves", "posture_delta", "posture_range", "zcr", "bp_lo",
                              "bp_mid", "bp_hi", "coverage"],
           "null_means": "NaN", "cases": []}
    for c in CASES:
        start = datetime.fromisoformat(c["start"])
        t0 = start.timestamp()
        at, ag = accel_samples(t0, c["accel"])
        temp = np.asarray(c["temp"], dtype=float).reshape(-1, 2)
        night = rr.NightState(
            session_start=datetime.fromisoformat(c["session_start"]),
            nights_before=c["nights_before"],
            onset_at=datetime.fromisoformat(c["onset_at"]) if c["onset_at"] else None,
            hr_reference_bpm=c["hr_reference_bpm"], temp_reference_c=c["temp_reference_c"])
        inp = rr.RowInput(start=start, nn_s=np.asarray(c["nn_s"], float), accel_t=at,
                          accel_g=ag, temp_t=t0 + temp[:, 0], temp_c=temp[:, 1])
        vals = rr.build_row_values(inp, night)
        ep = rr.accel_epochs_for_row(at, ag, start)
        case = dict(c)
        case["expected"] = {
            "row_values": {k: _num(v) for k, v in zip(rr.ROW_FEATURES, vals)},
            "accel_epochs": None if ep is None else [[_num(x) for x in r] for r in ep],
            "accel_samples_check": {"n": int(len(at)),
                                    "first": [_num(at[0] - t0), *[_num(x) for x in ag[0]]]},
        }
        out["cases"].append(case)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(out['cases'])} cases)")


if __name__ == "__main__":
    main()
