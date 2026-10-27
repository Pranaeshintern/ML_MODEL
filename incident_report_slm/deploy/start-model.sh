#!/usr/bin/env bash
set -euo pipefail
runtime_dir="${WUALT_RUNTIME_DIR:-/var/lib/wualt}"
model_file="$runtime_dir/models/Qwen3-14B-Q8_0.gguf"
llama_bin="$(find "$runtime_dir/llama" -type f -name llama-server -print -quit)"
if [[ -z "$llama_bin" || ! -f "$model_file" ]]; then
  echo "Install the pinned runtime and model using DEPLOYMENT.md first." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$llama_bin"):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$llama_bin" -m "$model_file" --alias wualt-context-slm \
  --host 127.0.0.1 --port 8081 -c 8192 -np 1 -ngl 99 -t 8 \
  --jinja --reasoning off --no-ui
