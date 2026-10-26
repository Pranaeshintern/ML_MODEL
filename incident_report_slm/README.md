# WUALT signal interpretation wrapper 0.13.0

The backend sends a compact JSON snapshot. The model explains what the supplied signals suggest and returns **two separate interpretation paragraphs**. Report forms, officer/wearer details, report numbers, PDF rendering and button-trigger logic are not part of this path.

## Architecture

Backend signals → validate JSON and timing → two isolated model calls → check each paragraph → return both interpretations.

1. **Audio interpretation:** receives only transcript, tone and sound-event labels, with their reliability/confidence and speaker-attribution metadata. No supplied state, keywords, HR/HRV, activity, chat, location or snapshot/window timing is sent to this call. It explains what the audio observations suggest.
2. **Distress explanation:** receives the validated backend evidence. It explains which observations support or conflict with the supplied state, and what remains uncertain. Keywords are derived hints, not independent corroboration. HR/HRV provide supporting context only. It does not claim to know the upstream classifier's internal reasoning or establish an unobserved cause.

The calls are independent: the distress call uses original evidence, not the generated audio paragraph. Both must pass before a response is returned.

The wrapper consumes already-produced transcript, tone labels, sound-event labels and a normal/stress/distress classification. It does not transcribe audio, detect sounds or train a classifier. The supplied state is context to assess, not ground truth to repeat. HR and HRV support the explanation; values alone do not establish distress, cause or a diagnosis.

The model configuration remains Qwen3-14B Q8_0 with llama.cpp b11582. This release changes the input contract, prompt and output interface; it does not train model weights. Exact model revision and checksums are in `wualt_slm/assets/runtime.lock.json`.

## Input

Use `schema_version: "interpretation-v1"`. A complete fictional example is in [interpretation-v1.json](wualt_slm/assets/examples/interpretation-v1.json).

Only `time` (an ISO timestamp with timezone) and `state.label` are required. Send available evidence; omit unknown measurements instead of supplying zero. `state.label` is `normal`, `stress`, `distress` or `unknown`.

| Field | Meaning |
| --- | --- |
| `time` | Timestamp of the backend snapshot; observations cannot be from its future. |
| `observation_window` | Optional backend-selected start/end window. Timed sound events and HR must fall within it. |
| `state` | Upstream label and optional confidence from 0 to 1. |
| `transcript` | Transcribed wording, preserving negation, quotations and tense. |
| `transcript_reliability` | usable / limited / unreliable / unknown / unavailable. |
| `speaker_verified` | Whether the transcript is verified as the subject's speech; defaults to false. |
| `tone` | Supplied label and optional confidence, e.g. frightened or calm. |
| `keywords` | Separate stress and distress lists. These are extraction hints, not independent observations. |
| `events` | Detected sound labels such as dog barking or gunshot, with optional timestamp/confidence. |
| `hr` | bpm, measurement time, signal quality and optional comparable reference. |
| `hrv` | RMSSD or SDNN in ms, measurement window, quality, artifact correction and optional reference. |
| `location` | Coordinates, fix time, optional accuracy and last-known flag. |
| `activity`, `chat_context` | Optional supporting context; no activity or conversation is required. |
| `request_id` | Optional caller reference. This stateless endpoint does not store or deduplicate requests. |

HR comparison requires usable quality and an explicitly comparable reference. HRV comparison additionally requires matching metric and duration, usable quality, artifact correction and overlap with the observation window. Invalid timestamps, reversed windows, invalid ranges, blank text and unknown fields fail before inference. Bounds are transport checks, not clinical thresholds.

The distress prompt relates the transcript, tone and sound events to the supplied state; the audio prompt never receives that state. It preserves disagreements: a normal label must not erase a help request. A detected gunshot is not proof of a shooting; barking does not establish a dog attack. Location fix age/accuracy and time provide context, not a danger score. Raw coordinates and exact numeric measurements/timestamps are validated and reduced to comparison/timing descriptions before inference; the caller retains the original values. Only last-known or pre-window location fixes are included as interpretation limitations. Recent coordinates remain with the caller. This reduces confusion between units, locations and elapsed time in the paragraph. Input text is treated as untrusted data. Instruction-like labels and zero-confidence tone/sound labels are excluded from usable evidence; this is a heuristic, not a guarantee against all prompt injections.

## Output

```json
{
  "audio_interpretation": "A paragraph interpreting only transcript, tone and sound events.",
  "distress_interpretation": "A paragraph explaining evidence for or against the supplied state and relevant uncertainty."
}
```

Only these two interpretation fields are returned. There are no report sections or echoed input fields. Internal sentence citations help detect invalid source references, wrong supplied-state names and invented numbers; they are not exposed in the response.

Each model call allows one bounded revision for a failed grounding check. Truncated, invalid or still-flagged output in either call fails the entire request; a partial pair is never presented as success. Two model calls are required, so the old single-call latency does not apply. **Fallback is disabled.** These checks do not certify semantic accuracy; human review remains necessary.

## Run

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m wualt_slm.setup_model --model-only --directory /workspace/slm-runtime
llama-server -m /workspace/slm-runtime/models/Qwen3-14B-Q8_0.gguf --alias wualt-context-slm --host 127.0.0.1 --port 8081 -c 8192 -np 1 -ngl 99 --jinja --reasoning off
```

The verified llama.cpp binary must be installed separately; the model-only command downloads weights. In another terminal:

```bash
export WUALT_MODEL_URL=http://127.0.0.1:8081
wualt-slm serve --host 0.0.0.0 --port 8000
```

Open the root page for the interpretation-only preview. Configure `WUALT_API_KEY` and `WUALT_ALLOWED_ORIGINS` before exposing the service. The configured API key is sent in `X-API-Key`.

```bash
curl -X POST http://127.0.0.1:8000/v1/interpretations \
  -H 'Content-Type: application/json' \
  --data-binary @wualt_slm/assets/examples/interpretation-v1.json

wualt-slm interpret wualt_slm/assets/examples/interpretation-v1.json --output /workspace/review/interpretation.json
```

`GET /v1/interpretation-input-example` returns the example; `/openapi.json` contains the full contract. The API returns 422 for invalid input, 502 for inference failure and 503 with Retry-After when busy. It does not dispatch assistance.

Python:

```python
from wualt_slm import SignalInterpreter
result = SignalInterpreter("http://127.0.0.1:8081").interpret(payload)
print(result["audio_interpretation"])
print(result["distress_interpretation"])
```

## Compatibility and verification

**v0.13 response change:** `/v1/interpretations`, `SignalInterpreter.interpret()` and the `interpret` CLI now return the two keys above instead of the v0.12 `interpretation` key. The `interpretation-v1` input JSON is unchanged. Update callers to read both keys.

The earlier `IncidentReporter`, `generate` command and `/v1/incident-reports` endpoints remain for explicit legacy callers. They are not used by the new interpretation endpoint or home page. Existing report requests retain their old schema; they are not silently converted into the compact contract.

```bash
pytest -q
python scripts/evaluate_signal_interpretations.py --model-url http://127.0.0.1:8081 --output /workspace/review/signals
```

Unit tests simulate inference. The evaluation command runs 34 fictional cases against a live model and writes interpretation results separately from source code. Generation success and structural checks are not semantic accuracy scores. Do not commit generated outputs or model weights.
