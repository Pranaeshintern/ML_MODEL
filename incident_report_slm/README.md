# WUALT interpretation API — 0.19.2

See [the deployment guide](../DEPLOYMENT.md) for model download and startup commands.

The wrapper accepts backend JSON and uses Qwen3-14B to return two independent paragraphs:
1. `audio_interpretation`: transcript, tone and sound events only.
2. `distress_interpretation`: original validated evidence, the supplied state and supporting context, including usable heart comparisons/current user chat.

The backend owns the full report layout. This package does not transcribe audio, detect sounds, train weights, save reports or dispatch assistance.

**0.19.1:** resolves four human-review findings from v0.18.0. Heart comparisons are not described as trends across the window. A game character's call for help stays attributed to the character. Past fear followed by "I am okay" or "I feel calm now" is not treated as support for current distress. A usable transcript gets no invented reliability caveat. **0.19.2** limits that check to the transcript itself, so doubts about zero-confidence tone or sound labels are allowed. The request and response contract is unchanged.

## Request

POST `/v1/interpretations`, Content-Type: application/json, X-API-Key: your configured key.

Use schema_version `interpretation-v1`. Only time (ISO timestamp with timezone) and state.label are required. Labels: normal, stress, distress, unknown. Omit unknown evidence instead of using zero; unknown fields are rejected.

See the [complete fictional JSON example](wualt_slm/assets/examples/interpretation-v1.json). Optional fields include observation_window, transcript/reliability/speaker verification, tone, keywords, sound events, HR/HRV with references, location, activity labels and chat_context. The backend chooses the observation window; preserve negation, attribution and original timestamps. `/openapi.json` contains the full schema; authenticated GET `/v1/interpretation-input-example` returns a validated example.

Physiological comparisons require validated signal quality and comparable references; HRV additionally requires matching metric/duration and artifact correction. Physiology, coordinates or a supplied label alone do not establish the cause of distress. Speaker verification does not verify the truth of a reported event.

## Response

~~~json
{
  "audio_interpretation": null,
  "distress_interpretation": null,
  "sections": {
    "audio": {"status": "not_available", "reason": "no_audio_evidence", "attempts": 0, "diagnostics": []},
    "distress": {"status": "not_available", "reason": "insufficient_context", "attempts": 0, "diagnostics": []}
  }
}
~~~

This is an insufficient-data example, not generated prose. Each section is generated, not_available or failed. Text is present only for generated sections. A successful section can coexist with a failed/unavailable section. Fallback prose is disabled.

| HTTP | Handling |
| --- | --- |
| 200 | Inspect both statuses; can include missing data or one failed section. |
| 401 | Missing/incorrect wrapper API key. |
| 403 | Browser Origin not allowed. |
| 413 | Body exceeds 64 KiB. |
| 422 | Invalid input; inspect field errors. |
| 502 | No section generated due to a model failure; can contain sections or a detail error. |
| 503 | Busy; obey Retry-After and use bounded retries. |

GET `/health` is unauthenticated and returns HTTP 200 even when model_ready is false; inspect the boolean. /docs and /openapi.json expose the contract. The small root preview/static assets remain because the API mounts them at startup. The inference service is stateless.

Run one wrapper worker per model instance and queue requests in the backend. Model calls permit a bounded revision. Passing validation is not semantic accuracy certification; existing review limitations remain.
