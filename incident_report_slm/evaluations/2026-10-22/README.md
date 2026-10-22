# Interpretation comparison — 22 October 2026

This is a development-set comparison of 83 fictional snapshots: 23 original scenarios and a 60-case context × audio × button matrix. The prompts were iterated on this same set. It is not an independent holdout benchmark or a measure of production accuracy.

The model was Qwen3-1.7B Q8_0 served by llama.cpp b11582 on the RunPod GPU. No fine-tuning was performed. Both runs used raw-review mode without fallback substitution. The final run generated 83/83 reports with no generation errors.

| Targeted diagnostic | Before | After |
| --- | ---: | ---: |
| At least one observation evidence ID | 30/83 | 83/83 |
| No metadata labels in prose | 8/83 | 83/83 |
| Received audio not described as missing | 67/83 | 83/83 |
| Supplied heart-rate measurements not described as absent | 16/83 | 83/83 |
| Required SOS priority phrase / no invented urgent handling | 42/83 | 83/83 |
| Location-status pattern check | 78/83 | 83/83 |

All 83 final outputs satisfy the 13 narrow diagnostics in the comparison script. These regex checks do not establish that observations are faithfully paraphrased or that reasoning is correct. In particular, presence of an evidence ID is not proof of grounding.

The service's separate validator still flags **18 interpretations** for missing limitation citations; five of those also omit the assessment citation. **53 complete reports** have at least one description or interpretation warning. Repetition and awkward phrasing also remain. Every output is an unvalidated draft for manual inspection. The original description prompt was retained.

The wrapper made a second interpretation attempt in 37 cases and selected it in 25; it retained the first in 58. Median end-to-end wrapper generation time was 1.087 seconds on this running GPU server, excluding PDF rendering; this is not a load or latency guarantee.

## Reproduce

Install the package, start its pinned model, then from incident_report_slm:

```bash
python scripts/evaluate_interpretations.py --baseline evaluations/2026-10-22/baseline.json --model-url http://127.0.0.1:8081 --output outputs/interpretation-comparison
```

- baseline.json: all fictional input snapshots and original interpretation text.
- comparison.json: exact before/after text, evidence IDs, diagnostics and service warnings.
- summary.json: aggregate diagnostics and runtime metadata.

Review the source input alongside both paragraphs. Inspect SOS handling, missing-data statements, unsupported causality, invented chronology, citations, and repetition manually. The script leaves model output unchanged.
