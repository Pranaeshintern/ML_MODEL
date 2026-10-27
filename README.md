# WUALT SLM deployment

Deployment-only branch: **deploy/slm**. Start with [DEPLOYMENT.md](DEPLOYMENT.md).

This branch contains the runtime wrapper, required modules/assets, one input example, dependency constraints, a wrapper Dockerfile and Linux service launchers. Model weights are downloaded from the pinned official Qwen revision and verified by SHA-256; weights are not stored in GitHub.

- [Backend deployment guide](DEPLOYMENT.md)
- [API contract](incident_report_slm/README.md)
- [Complete input JSON](incident_report_slm/wualt_slm/assets/examples/interpretation-v1.json)

The development branch `codex/interpretation-only-json` retains tests and evaluation tools. This branch excludes those tools, reports, logs, weights and unrelated activity/sleep/snore projects. Wrapper version 0.19.0 and interpretation behavior are unchanged by deployment packaging.
