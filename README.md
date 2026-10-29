# WUALT model modules

Each folder has its own installation and integration instructions.

| Module | Purpose |
| --- | --- |
| [activity_detection](activity_detection/) | Wearable activity detection |
| [Sleep_models](Sleep_models/) | Sleep pipeline and model artifacts |
| [snore_detection](snore_detection/) | Snore inference bundle |
| [incident_report_slm](incident_report_slm/) | Pretrained Qwen3 wrapper and REST API for audio and distress signal interpretation |

## Incident report SLM

- [Backend deployment guide](DEPLOYMENT.md)
- [API contract](incident_report_slm/README.md)
- [Complete input JSON](incident_report_slm/wualt_slm/assets/examples/interpretation-v1.json)

Model weights are downloaded from the pinned official Qwen revision and verified by SHA-256; weights are not stored in GitHub.

Branches: `deploy/slm` holds only the SLM deployment files. `codex/interpretation-only-json` holds the SLM with its tests and evaluation tools.
