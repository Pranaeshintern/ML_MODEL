"""Run the multilingual scenarios against a live model and keep every rejected draft.

The service itself never returns rejected prose. For review only, this script wraps the
validator so each rejected draft and its issues are saved next to the final result.
"""
import argparse
import json
import runpy
import statistics
import time
from pathlib import Path

import wualt_slm.signal_interpretation as interpretation
from wualt_slm import SignalInterpreter

HERE = Path(__file__).parent


def capture_rejections(store):
    original = interpretation.draft_issues

    def recording(draft, facts, *, section="distress"):
        issues = original(draft, facts, section=section)
        if issues:
            store.append({"section": section, "draft": " ".join(s.text for s in draft.sentences), "issues": issues})
        return issues

    interpretation.draft_issues = recording
    return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-url", default="http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = runpy.run_path(str(HERE / "multilingual_scenarios.py"))["multilingual_cases"]()
    rejected = []
    capture_rejections(rejected)
    interpreter = SignalInterpreter(args.model_url)
    rows = []
    for case in cases:
        rejected.clear()
        started = time.monotonic()
        try:
            case["output"] = interpreter.interpret(case["input"])
        except Exception as exc:
            case["error"] = type(exc).__name__ + ": " + str(exc)
        case["seconds"] = round(time.monotonic() - started, 3)
        case["rejected_drafts"] = list(rejected)
        rows.append(case)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        statuses = "request_error" if "error" in case else {k: v["status"] for k, v in case["output"]["sections"].items()}
        print(case["scenario"], statuses, case["seconds"], "rejected:", len(case["rejected_drafts"]), flush=True)
    sections = [s for r in rows for s in r.get("output", {}).get("sections", {}).values()]
    summary = {"cases": len(rows), "prompt_version": interpretation.PROMPT_VERSION,
               "generated_sections": sum(s["status"] == "generated" for s in sections),
               "failed_sections": sum(s["status"] == "failed" for s in sections),
               "unavailable_sections": sum(s["status"] == "not_available" for s in sections),
               "rejected_drafts": sum(len(r["rejected_drafts"]) for r in rows),
               "request_errors": sum("error" in r for r in rows),
               "median_seconds": round(statistics.median(r["seconds"] for r in rows), 3),
               "scope": "Fictional multilingual transcripts; generation success is not semantic accuracy."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    runpy.run_path(str(HERE / "interpretation_review.py"))["save_html"](
        rows, args.output / "Multilingual_Review.html",
        {"suite": "multilingual", "model": "Live model at the configured endpoint"})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
