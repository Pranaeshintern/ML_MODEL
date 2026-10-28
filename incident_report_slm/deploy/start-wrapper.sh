#!/usr/bin/env bash
set -euo pipefail
: "${WUALT_API_KEY:?Set WUALT_API_KEY before starting the wrapper.}"
# Must point at the model port chosen in start-model.sh (WUALT_MODEL_PORT, default 8081).
export WUALT_MODEL_URL="${WUALT_MODEL_URL:-http://127.0.0.1:${WUALT_MODEL_PORT:-8081}}"
exec /opt/wualt/venv/bin/wualt-slm serve \
  --host "${WUALT_HTTP_HOST:-127.0.0.1}" --port "${WUALT_HTTP_PORT:-8000}"
