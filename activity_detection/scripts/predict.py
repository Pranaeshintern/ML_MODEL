"""Run the trained detector on a raw sensor CSV.

    python scripts/predict.py data/sample_sensor/real_walking_60s.csv
    python scripts/predict.py data/sample_sensor/*.csv --abstain 0.5

The CSV needs a timestamp column and three accelerometer columns; column names are
detected from a few common conventions. Values may be in g or m/s^2 -- the unit is
inferred from the median magnitude, since every source in this project disagrees
about it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wualt.models.inference import ActivityDetector  # noqa: E402

# The trained model ships beside this package, in ./model.
PROC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")

ACC_ALIASES = [("ax", "ay", "az"), ("x", "y", "z"), ("x_axis", "y_axis", "z_axis")]
TIME_ALIASES = ("timestamp", "time", "t")


def read_signal(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}

    acc_cols = next((a for a in ACC_ALIASES if all(c in cols for c in a)), None)
    if acc_cols is None:
        # Fall back to the first three numeric columns after the timestamp.
        num = df.select_dtypes("number").columns.tolist()
        if len(num) < 3:
            raise ValueError(f"{path}: cannot find three accelerometer columns")
        acc_cols = tuple(num[:3])
        acc = df[list(acc_cols)].to_numpy(dtype=np.float64)
    else:
        acc = df[[cols[c] for c in acc_cols]].to_numpy(dtype=np.float64)

    tcol = next((cols[c] for c in TIME_ALIASES if c in cols), None)
    if tcol is not None:
        ts = pd.to_datetime(df[tcol], errors="coerce")
        if ts.notna().all():
            t = ((pd.DatetimeIndex(ts) - pd.DatetimeIndex(ts)[0])
                 / np.timedelta64(1, "s")).to_numpy(dtype=np.float64)
        else:
            t = df[tcol].to_numpy(dtype=np.float64)
            if t.max() > 1e12:  # nanoseconds
                t = (t - t[0]) / 1e9
    else:
        t = np.arange(len(df)) / 25.0  # assume 25 Hz if no clock is supplied

    # Unit inference: at rest the magnitude is 1 g or 9.81 m/s^2. There is no
    # ambiguous middle ground, so a single threshold is safe.
    if np.median(np.linalg.norm(acc, axis=1)) > 5.0:
        acc = acc / 9.80665

    hrcol = next((cols[c] for c in ("hr", "heart_rate", "bpm") if c in cols), None)
    hr = df[hrcol].to_numpy(dtype=np.float64) if hrcol else None
    return t, acc, hr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model-dir", default=PROC)
    ap.add_argument("--variant", default="acc_only", choices=["acc_only", "acc_hr"],
                    help="acc_hr additionally consumes a heart-rate column if present")
    ap.add_argument("--abstain", type=float, default=0.0,
                    help="minimum top-class probability; below it emit 'unknown'")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="minimum top-2 probability gap")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    det = ActivityDetector(args.model_dir, variant=args.variant)
    files = [p for pat in args.paths for p in sorted(glob.glob(pat))]
    if not files:
        print("no input files matched")
        return

    results = {}
    for path in files:
        t, acc, hr = read_signal(path)
        res = det.predict(
            t, acc, hr,
            abstain_threshold=args.abstain,
            abstain_margin=args.margin,
            smooth_output=not args.no_smooth,
        )
        summary = det.summarize_session(res)
        results[path] = {"summary_minutes": summary, "n_windows": res["n_windows"],
                         "abstain_rate": res["abstain_rate"], "bouts": res["bouts"],
                         "sessions": res["sessions"]}

        dur = (t[-1] - t[0]) / 60.0
        print(f"\n{os.path.basename(path)}  ({dur:.1f} min, {res['n_windows']} windows)")
        if not res["n_windows"]:
            print("  too short for a 30 s window")
            continue
        for name, mins in summary.items():
            print(f"    {name:10s} {mins:6.2f} min")
        conf = np.asarray(res["confidence"])
        print(f"    [mean confidence {conf.mean():.3f}, abstained {res['abstain_rate']:.1%}]")
        for s in res["sessions"]:
            ask = "  -> ask the user what this was" if s["needs_confirmation"] else ""
            print(f"    session {s['t_start_s'] / 60:5.1f}-{s['t_end_s'] / 60:5.1f} min  "
                  f"{s['session_label']}{ask}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
