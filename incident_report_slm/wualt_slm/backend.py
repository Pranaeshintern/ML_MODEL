"""Pinned local llama.cpp inference adapter for signal interpretation."""
import json
import os
from pathlib import Path
from threading import Lock

import httpx

from .settings import ASSET_ROOT, PROJECT_ROOT

RUNTIME_LOCK = json.loads((ASSET_ROOT / 'runtime.lock.json').read_text(encoding='utf-8'))


class BackendUnavailable(RuntimeError):
    pass


class LlamaBackend:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = (base_url or os.getenv('WUALT_MODEL_URL', 'http://127.0.0.1:8081')).rstrip('/')
        self._api_key = api_key
        self._lock = Lock()

    def headers(self):
        key = self._api_key or os.getenv('WUALT_MODEL_API_KEY')
        key_file = Path(os.getenv('WUALT_MODEL_API_KEY_FILE', str(PROJECT_ROOT / '.runtime/model-api-key.txt')))
        if not key and key_file.is_file():
            key = key_file.read_text(encoding='utf-8-sig').strip()
        return {'Authorization': f'Bearer {key}'} if key else {}

    def ready(self):
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                response = client.get(self.base_url + '/health', headers=self.headers())
                if response.status_code != 200:
                    return False
                health = response.json()
                return isinstance(health, dict) and health.get('status') == 'ok'
        except (httpx.HTTPError, ValueError):
            return False

    def generate_interpretation(self, evidence):
        from .signal_interpretation import generate_signal_interpretation
        return generate_signal_interpretation(self, evidence)
