#!/usr/bin/env bash
set -euo pipefail
: "${WUALT_API_KEY:?Set WUALT_API_KEY before starting the wrapper.}"
export WUALT_MODEL_URL="${WUALT_MODEL_URL:-http://127.0.0.1:8081}"
exec /opt/wualt/venv/bin/wualt-slm serve \
  --host "${WUALT_HTTP_HOST:-127.0.0.1}" --port "${WUALT_HTTP_PORT:-8000}"
