# -*- coding: utf-8 -*-
"""Reference HTTP service. Optional - the library is the deliverable.

    pip install -r requirements.txt
    python server.py          ->  http://127.0.0.1:8765

POST /detect   an audio file (WAV/FLAC/OGG, any rate, any channels)
GET  /health   model, threshold, window size
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snore_detect import SnoreDetector  # noqa: E402

DET = SnoreDetector()
NIGHT_RULE_MIN_S = 3600.0
app = FastAPI(title="WUALT snore detection", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "model": DET.meta["model"], "threshold": DET.threshold,
            "window_s": DET.window_s, "fs_hz": DET.fs}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        x, fs = sf.read(io.BytesIO(raw), dtype="float64", always_2d=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode audio: {e}. "
                                 "Use WAV, FLAC or OGG - not MP3/AAC/Opus.")
    try:
        r = DET.score_stream(np.asarray(x), float(fs))
    except ValueError as e:
        raise HTTPException(400, str(e))

    dur = float(r["t_start"][-1] + DET.window_s)
    out = {
        "duration_s": round(dur, 2),
        "n_windows": int(r["score"].size),
        "threshold": DET.threshold,
        "n_snoring": int(r["snoring"].sum()),
        "snoring": bool(r["snoring"].any()),
        "windows": [{"start_s": float(t), "score": round(float(p), 4),
                     "snoring": bool(d)}
                    for t, p, d in zip(r["t_start"], r["score"], r["snoring"])],
    }
    if dur >= NIGHT_RULE_MIN_S:
        v = DET.decide_night(r["t_start"], r["snoring"])
        out["night_rule"] = {"applicable": True, "state": v.state,
                             "n_clusters": v.n_clusters,
                             "longest_span_s": v.max_cluster_span_s}
    else:
        out["night_rule"] = {"applicable": False,
                             "note": "Night verdict needs an hour or more; for a "
                                     "short clip read the window results."}
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8765)))
