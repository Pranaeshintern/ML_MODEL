#!/usr/bin/env bash
set -euo pipefail
runtime_dir="${WUALT_RUNTIME_DIR:-/var/lib/wualt}"
# Private loopback port for llama.cpp. RunPod's standard templates already run a
# proxy on 8081; set WUALT_MODEL_PORT (for example 18083) there.
model_port="${WUALT_MODEL_PORT:-8081}"
if [[ ! "$model_port" =~ ^[0-9]+$ ]] || (( model_port < 1 || model_port > 65535 )); then
  echo "WUALT_MODEL_PORT must be a port number (got: $model_port)." >&2
  exit 1
fi
model_file="$runtime_dir/models/Qwen3-14B-Q8_0.gguf"
llama_bin="$(find "$runtime_dir/llama" -type f -name llama-server -print -quit)"
if [[ -z "$llama_bin" || ! -f "$model_file" ]]; then
  echo "Install the pinned runtime and model using DEPLOYMENT.md first." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$llama_bin"):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$llama_bin" -m "$model_file" --alias wualt-context-slm \
  --host 127.0.0.1 --port "$model_port" -c 8192 -np 1 -ngl 99 -t 8 \
  --jinja --reasoning off --no-ui
