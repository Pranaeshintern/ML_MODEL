# WUALT SLM deployment guide

**Backend handoff — 27 October 2026**

Wrapper 0.20.1 · branch `deploy/slm` · Qwen3-14B Q8_0 · llama.cpp b11582

## Architecture

~~~text
Backend JSON -> wrapper :8000 -> private llama.cpp :8081 -> Qwen3-14B
Backend receives audio_interpretation, distress_interpretation and section statuses.
~~~

The backend inserts these paragraphs into its own report. The wrapper and model server are separate processes on the same GPU host. The weights are original pretrained weights; custom behavior is in wrapper prompts and validation. GitHub contains code, not weights. These commands do not provision paid compute.

## 1. Prerequisites and persistent storage

Target: Linux x86_64, Ubuntu 24.04, Python 3.12, NVIDIA drivers and CUDA 12.8 runtime including cuBLAS. Prior inference used an A40 with 48 GB VRAM and about 50 GB host RAM. This is the tested hardware reference, not a minimum-memory guarantee. Follow [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Reserve at least 40 GB free persistent disk, preferably 50 GB for upgrades. The model is 15,698,533,728 bytes (about 15.7 GB). Mount persistent storage at `/var/lib/wualt` BEFORE downloading. A directory alone is not persistent storage on an ephemeral pod. On RunPod, configure a persistent/network volume and map this path to it. Code goes in `/opt/wualt/source`; weights/runtime stay outside Git.

~~~bash
nvidia-smi
uname -m   # expected: x86_64
sudo apt-get update
sudo apt-get install -y python3 python3-venv git curl ca-certificates libgomp1 openssl
python3 --version   # use Python 3.12
~~~

GPU driver/CUDA installation are prerequisites; these commands do not modify drivers. Use sudo-capable Bash. Systemd instructions require a VM/host with systemd; in a container/RunPod shell without it, use foreground launchers and your platform supervisor.

## 2. Clone and install

~~~bash
id -u wualt >/dev/null 2>&1 || sudo useradd --system --create-home --shell /usr/sbin/nologin wualt
sudo install -d -o wualt -g wualt /opt/wualt /var/lib/wualt
sudo -u wualt git clone --depth 1 --single-branch --branch deploy/slm \
  https://github.com/Pranaeshintern/ML_MODEL.git /opt/wualt/source
sudo -u wualt python3 -m venv /opt/wualt/venv
sudo -u wualt /opt/wualt/venv/bin/python -m pip install --upgrade pip
sudo -u wualt /opt/wualt/venv/bin/python -m pip install \
  -c /opt/wualt/source/incident_report_slm/requirements-tested.txt \
  /opt/wualt/source/incident_report_slm
sudo -u wualt git -C /opt/wualt/source rev-parse HEAD
~~~

Record the commit in your release record. The shallow single-branch clone contains only deployment files. For private-repository access use your normal GitHub read authentication; do not embed a token in the URL.

## 3. Download Qwen3-14B

~~~bash
sudo -u wualt /opt/wualt/venv/bin/python -m wualt_slm.setup_model \
  --model-only --directory /var/lib/wualt
~~~

Expected file: `/var/lib/wualt/models/Qwen3-14B-Q8_0.gguf`.

The installer reads `wualt_slm/assets/runtime.lock.json`: official revision `530227a7d994db8eca5ab5ced2fb692b614357fd`, SHA-256:

~~~text
a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d
~~~

It skips verified files and resumes partial downloads where supported. Do not proceed after checksum failure. Remove only the corrupt .partial file reported by the installer before retrying.

Source: [official Qwen GGUF repository at the pinned revision](https://huggingface.co/Qwen/Qwen3-14B-GGUF/tree/530227a7d994db8eca5ab5ced2fb692b614357fd). It lists Apache-2.0 licensing; retain applicable notices when redistributing weights.

## 4. Install the Linux llama.cpp runtime

The model-only command downloads weights only. Run this whole Bash block; it exits on download/checksum failure before extraction.

~~~bash
(
  set -euo pipefail
  sudo -u wualt mkdir -p /var/lib/wualt/llama
  sudo -u wualt curl --fail --location --retry 3 \
    https://github.com/ggml-org/llama.cpp/releases/download/b11582/llama-b11582-bin-ubuntu-cuda-12.8-x64.tar.gz \
    --output /var/lib/wualt/llama-b11582.tar.gz
  printf '%s  %s\n' \
    e7a5c43a70d73c05a52122a84dfd250fd856f2f610c3f3669077b7abdc7d2e7f \
    /var/lib/wualt/llama-b11582.tar.gz | sha256sum --check -
  sudo -u wualt tar -xzf /var/lib/wualt/llama-b11582.tar.gz \
    -C /var/lib/wualt/llama
  LLAMA_BIN="$(find /var/lib/wualt/llama -type f -name llama-server -print -quit)"
  test -n "$LLAMA_BIN"
  LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN"):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LLAMA_BIN" --version
)
~~~

The asset/digest are from [llama.cpp b11582](https://github.com/ggml-org/llama.cpp/releases/tag/b11582). Do not silently switch to latest. Missing CUDA/cuBLAS libraries mean host prerequisites need attention. Use ldd on the discovered llama-server to identify unresolved libraries. The launcher includes the unpacked runtime directory and conventional CUDA directory in LD_LIBRARY_PATH.

## 5. First start and smoke test

Start the model in terminal A:

~~~bash
sudo -u wualt env WUALT_RUNTIME_DIR=/var/lib/wualt \
  bash /opt/wualt/source/incident_report_slm/deploy/start-model.sh
~~~

It binds **127.0.0.1:8081**, loads all possible layers on GPU, uses an 8192-token context, one model slot, alias wualt-context-slm, and disables thinking/model web UI. Wait for loading. In another terminal:

~~~bash
curl --fail http://127.0.0.1:8081/health
# Expected once loaded: {"status":"ok"}
~~~

Create the wrapper secret/configuration once. The key goes to a protected file without printing. Do not rerun this block for ordinary updates: it replaces the key.

~~~bash
sudo install -d -m 750 -o root -g wualt /etc/wualt
sudo sh -c 'umask 027; { printf "WUALT_MODEL_URL=http://127.0.0.1:8081\nWUALT_HTTP_HOST=127.0.0.1\nWUALT_HTTP_PORT=8000\nWUALT_API_KEY="; openssl rand -hex 32; } > /etc/wualt/wrapper.env'
sudo chown root:wualt /etc/wualt/wrapper.env
sudo chmod 640 /etc/wualt/wrapper.env
~~~

Start the wrapper in terminal B:

~~~bash
sudo -u wualt bash -c 'set -a; source /etc/wualt/wrapper.env; set +a; exec bash /opt/wualt/source/incident_report_slm/deploy/start-wrapper.sh'
~~~

From terminal C, check readiness and submit the complete fictional input:

~~~bash
curl --fail http://127.0.0.1:8000/health
# Check model_ready is true; HTTP 200 alone is not readiness.
WUALT_API_KEY="$(sudo sed -n 's/^WUALT_API_KEY=//p' /etc/wualt/wrapper.env)"
curl --silent --show-error --max-time 600 \
  --output /tmp/wualt-smoke-response.json --write-out 'HTTP %{http_code}\n' \
  http://127.0.0.1:8000/v1/interpretations \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $WUALT_API_KEY" \
  --data-binary @/opt/wualt/source/incident_report_slm/wualt_slm/assets/examples/interpretation-v1.json
python3 -m json.tool /tmp/wualt-smoke-response.json
unset WUALT_API_KEY
~~~

Expect the two interpretation keys and sections.audio / sections.distress. Generated text need not match a fixed sentence. Inspect statuses; completing an HTTP request alone does not establish success.

## 6. Restartable Linux services

Stop both foreground processes with Ctrl+C before installing services so ports are free. Keep weights on persistent storage.

~~~bash
sudo install -m 644 /opt/wualt/source/incident_report_slm/deploy/wualt-model.service /etc/systemd/system/
sudo install -m 644 /opt/wualt/source/incident_report_slm/deploy/wualt-wrapper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wualt-model.service wualt-wrapper.service
sudo systemctl status wualt-model.service wualt-wrapper.service --no-pager
curl --fail http://127.0.0.1:8000/health
~~~

The wrapper can start while the model loads. Gate traffic on model_ready: true. Logs, restart and stop:

~~~bash
sudo journalctl -u wualt-model -u wualt-wrapper -n 100 --no-pager
sudo systemctl restart wualt-wrapper
sudo systemctl stop wualt-wrapper wualt-model
~~~

Stopping a service/restarting a machine should not remove persistent weights. Avoid recording full real-user request bodies in surrounding application/proxy logs.

## 7. Backend integration

Route the backend to the wrapper, not directly to llama.cpp. Keep port 8081 private. Both launchers bind to loopback. Put your existing TLS reverse proxy before port 8000, or set WUALT_HTTP_HOST to a private interface and restrict network access. Store the key in the backend secret manager; send it in X-API-Key.

Use POST /v1/interpretations. Supply the selected observation window and original evidence. This service does not operate a button trigger. Persist inputs/reports on the backend if needed; the wrapper is stateless.

- generated: use the paragraph.
- not_available: show a missing/insufficient-data notice. Absent audio can coexist with a distress explanation.
- failed: handle a model/validation failure. Do not present it as missing data or a generated paragraph.

HTTP 200 can include one failed section when the other generated. HTTP 502 may have structured sections or a detail error. HTTP 422 means invalid input; HTTP 503 means busy and includes Retry-After. Use a backend queue and bounded retries. Run one wrapper worker per model instance; do not increase workers against the single-slot model. Allow a generous upstream read timeout (the example uses 600 seconds), since each section can require one revision. Do not retry invalid JSON unchanged.

Server-to-server clients normally omit Origin. For browser requests, set WUALT_ALLOWED_ORIGINS to exact origins, comma-separated. This is an origin check, not a complete CORS deployment; use a same-origin proxy or configure CORS at your gateway. /docs, /openapi.json and /health are outside the /v1 API-key dependency; scope their exposure at the gateway.

## 8. Wrapper updates

Record the current commit, pull, reinstall and restart. Prompt-only updates do not need another Qwen download/model restart.

~~~bash
sudo -u wualt git -C /opt/wualt/source rev-parse HEAD
sudo -u wualt git -C /opt/wualt/source pull --ff-only origin deploy/slm
sudo -u wualt /opt/wualt/venv/bin/python -m pip install \
  -c /opt/wualt/source/incident_report_slm/requirements-tested.txt \
  /opt/wualt/source/incident_report_slm
sudo systemctl restart wualt-wrapper
curl --fail http://127.0.0.1:8000/health
~~~

Repeat the JSON smoke test before routing live traffic. If runtime/model pins change, install and verify those assets and restart both services. Roll back by deploying your recorded prior commit into a separate checkout, reinstalling it and repeating checks. Preserve a working model copy during upgrades.

## 9. Optional wrapper-only Docker image

The Dockerfile installs the Python wrapper only. It does not download weights or run llama.cpp. On Linux, with the model running on the host:

~~~bash
cd /opt/wualt/source/incident_report_slm
sudo docker build -t wualt-wrapper:0.20.1 .
# Stop the systemd wrapper first if port 8000 is in use.
sudo docker run --rm --name wualt-wrapper --network host \
  --env-file /etc/wualt/wrapper.env \
  wualt-wrapper:0.20.1 wualt-slm serve --host 127.0.0.1 --port 8000
~~~

This uses Linux host networking. In a bridged network, localhost means the container; use a private model-service hostname instead. Configure restart/readiness policies in your orchestrator.

## 10. Troubleshooting and verification scope

| Symptom | Check |
| --- | --- |
| Download fails | Persistent disk space, outbound HTTPS, checksum output; retry the pinned installer. |
| GPU out of memory / model load fails | nvidia-smi, CUDA libraries and model logs. Changing quantization requires a tested release. |
| model_ready: false | Model startup, port 8081 and WUALT_MODEL_URL. |
| 401 | API key and X-API-Key header. |
| 422 | Field errors and /openapi.json; timestamps need timezones. |
| 503 | Queue/concurrency; obey Retry-After. |
| Null paragraph | Section status/reason; both do not always generate. |
| Restart loses weights | Ephemeral storage; configure the persistent mount before redownloading. |

The pinned model/runtime previously ran on an A40. Deployment packaging is checked through package installation, installed-package API tests with simulated inference, and Bash syntax parsing. This is not a fresh GPU run or a claim that systemd/Docker commands were executed on your production host. Complete the live smoke test there. Existing interpretation-quality limitations remain; generation success does not guarantee factual accuracy.
