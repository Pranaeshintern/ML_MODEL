"""Installed-wheel integration check; inference is explicitly simulated."""
from io import BytesIO
from contextlib import chdir
import json, os
from pathlib import Path
import tempfile

def main():
    with tempfile.TemporaryDirectory(prefix='wualt-wheel-') as directory, chdir(directory):
        os.environ['WUALT_DATA_DIR']=directory
        os.environ.pop('WUALT_API_KEY',None)
        from fastapi.testclient import TestClient
        from pypdf import PdfReader
        from wualt_slm import IncidentReporter, SignalInterpreter
        from wualt_slm.settings import ASSET_ROOT
        class Simulated:
            def generate_contextual_incident(self, evidence):
                part={'text':'The transcript reports a request for help, but speaker identity and the situation remain unconfirmed. The recorded SOS requires urgent handling.', 'evidence_ids':['T01','B']}
                return json.dumps({'interpretation':part,'interpretation_sentences':[dict(part,kind='interpretation')]})
        payload=json.loads((ASSET_ROOT/'examples/demo-001.json').read_text(encoding='utf-8'))
        payload['incident_description']='Supplied template description.'
        reporter=IncidentReporter(backend=Simulated(),mode='raw_review')
        report=reporter.generate(payload)
        assert report['view']['colour']=='RED'
        assert report['model_sections']==['interpretation'] and not report['fallback_enabled']
        assert report['narrative']['description']['text']==payload['incident_description']
        reader=PdfReader(BytesIO(reporter.pdf(report)))
        assert len(reader.pages)>=2
        assert 'SLM INTERPRETATION' in ' '.join(p.extract_text() for p in reader.pages)
        import wualt_slm.incident_api as api
        api.service=reporter.service
        class SimulatedSignals:
            def ready(self): return True
            def generate_interpretation(self, evidence):
                return {"interpretation":"The request for help and reported barking are consistent with the supplied distress label; an attack is not established."}
        api.backend=SimulatedSignals()
        signal_payload=json.loads((ASSET_ROOT/'examples/interpretation-v1.json').read_text())
        assert set(SignalInterpreter(backend=api.backend).interpret(signal_payload))=={"interpretation"}
        with TestClient(api.app) as client:
            assert client.get('/').status_code==200
            assert client.get('/static/app.js').status_code==200
            assert client.get('/static/interpretation.js').status_code==200
            assert client.get('/static/interpretation.css').status_code==200
            compact=client.post('/v1/interpretations',json=signal_payload)
            assert compact.status_code==200 and set(compact.json())=={"interpretation"}
            assert client.get('/v1/interpretation-input-example').json()["schema_version"]=="interpretation-v1"
            assert 'Interpret the situation.' in client.get('/').text
            assert len(client.get('/v1/incident-examples').json()['items'])==4
            saved=client.post('/v1/incident-reports',json=payload)
            assert saved.status_code==200,saved.text
            assert client.post('/v1/incident-reports',json=payload).json()['id']==saved.json()['id']
            assert client.get('/v1/incident-reports/'+saved.json()['id']+'/pdf').content.startswith(b'%PDF')
        assert (Path(directory)/'incident-reports.sqlite3').is_file()
        print('Installed wheel passed: supplied template fields, interpretation contract, assets, PDF, REST and retry reuse. Simulated inference only.')

if __name__=='__main__': main()
