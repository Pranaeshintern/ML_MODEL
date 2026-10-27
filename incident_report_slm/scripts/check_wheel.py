"""Installed-wheel integration check; inference is explicitly simulated."""
import json, os


def main():
    os.environ.pop('WUALT_API_KEY', None)
    from fastapi.testclient import TestClient
    from wualt_slm import SignalInterpreter
    from wualt_slm.settings import ASSET_ROOT
    import wualt_slm.api as api

    class SimulatedSignals:
        def ready(self): return True
        def generate_interpretation(self, evidence):
            return {"audio_interpretation": "The request for help relates to the reported barking; an attack is not established.", "distress_interpretation": "The request for help supports the supplied distress label.", "sections": {name: {"status": "generated", "reason": None, "attempts": 1, "diagnostics": []} for name in ("audio", "distress")}}
    api.backend = SimulatedSignals()
    signal_payload = json.loads((ASSET_ROOT / 'examples/interpretation-v1.json').read_text())
    assert set(SignalInterpreter(backend=api.backend).interpret(signal_payload)) == {"audio_interpretation", "distress_interpretation", "sections"}
    with TestClient(api.app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/health').json()['schemas'] == ['interpretation-v1']
        assert client.get('/static/interpretation.js').status_code == 200
        assert client.get('/static/interpretation.css').status_code == 200
        compact = client.post('/v1/interpretations', json=signal_payload)
        assert compact.status_code == 200 and set(compact.json()) == {"audio_interpretation", "distress_interpretation", "sections"}
        assert client.get('/v1/interpretation-input-example').json()["schema_version"] == "interpretation-v1"
        assert 'Interpret the situation.' in client.get('/').text
        assert client.get('/v1/incident-examples').status_code == 404
    print('Installed wheel passed: interpretation contract, assets and REST. Simulated inference only.')


if __name__ == '__main__': main()
