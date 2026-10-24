from model_fixtures import FixtureModel, draft
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfReader
from wualt_slm.backend import BackendUnavailable
from wualt_slm.incident_schema import IncidentBundle, Narrative
from wualt_slm.incident_facts import build_incident_view
from wualt_slm.incident_service import IncidentReportService,validate_narrative
from wualt_slm.incident_pdf import render_incident_pdf
from wualt_slm.incident_store import IncidentStore
from wualt_slm.incident_sources import load_from_source,SourceUnavailable

from wualt_slm.settings import ASSET_ROOT


def example(n=1):return json.loads((ASSET_ROOT/f'examples/demo-{n:03}.json').read_text())


class Offline:
    def generate_contextual_incident(self,evidence):raise BackendUnavailable('model_unavailable')


@pytest.mark.parametrize('number,colour,priority',[(1,'RED','Urgent SOS handling'),(2,'GREEN','Standard review'),(3,'GREY','Urgent SOS handling'),(4,'GREY','Verification needed')])
def test_four_reference_cases_preserve_colours_and_separate_sos_priority(number,colour,priority):
    view=build_incident_view(IncidentBundle.model_validate(example(number)))
    assert view['colour']==colour and view['priority']==priority
    assert len(view['sensor_rows'])==5


def test_location_age_and_missing_audio_are_reported_exactly():
    view=build_incident_view(IncidentBundle.model_validate(example(3)))
    assert '8 min 10 sec before' in view['gps']['fix']
    assert 'Current position unconfirmed' in view['gps']['label']
    assert view['speaker_count']=='Unavailable'
    assert 'not silence' in view['sensor_rows'][2]['interpretation']
    first=build_incident_view(IncidentBundle.model_validate(example()))
    assert '15 seconds after' in first['gps']['fix'] and 'not a start-time fix' in first['gps']['fix']


@pytest.mark.parametrize('change',[
    lambda d:d['audio'].update(received=False),
    lambda d:d['location'].update(latitude=95),
    lambda d:d['physiology'].update(hr_min_bpm=150,hr_max_bpm=100),
    lambda d:d['observation_window'].update(end=d['observation_window']['start']),
    lambda d:d['button'].update(at=None),
    lambda d:d.update(reported_at='2026-10-14T21:15:00'),
    lambda d:d['audio']['transcript'][0].update(speaker='MISSING'),
    lambda d:d['events'][0].update(at='2026-10-14T01:00:00+05:30'),
])
def test_bad_or_inconsistent_source_data_is_rejected(change):
    data=example();change(data)
    with pytest.raises(ValidationError):IncidentBundle.model_validate(data)


def test_green_context_does_not_override_sos_or_missing_observations():
    data=example(3);data['assessment']['context']='exercise_consistent'
    view=build_incident_view(IncidentBundle.model_validate(data))
    assert view['colour']=='GREY' and view['priority']=='Urgent SOS handling'


def test_no_actions_contacts_or_names_are_exposed_to_the_model():
    class Capture:
        def generate_contextual_incident(self,evidence):
            serialized=json.dumps(evidence)
            assert 'Demo Guardian' not in serialized and 'Demo Officer' not in serialized
            assert '+91' not in serialized and 'DEMO-USER' not in serialized
            raise BackendUnavailable('model_unavailable')
    with pytest.raises(BackendUnavailable, match='model_unavailable'):
        IncidentReportService(Capture()).generate(IncidentBundle.model_validate(example()))


def test_unsupported_numbers_and_invented_safety_are_rejected():
    data=IncidentBundle.model_validate(example());view=build_incident_view(data)
    bad=Narrative(text='An SOS was recorded and heart rate reached 999 bpm.',evidence_ids=['P','B'])
    assert 'unsupported_number' in validate_narrative(bad,'description',data,view)
    bad=Narrative(text='The wearer is safe. Urgent SOS handling is not needed.',evidence_ids=['C','B'])
    assert 'unsupported_conclusion' in validate_narrative(bad,'interpretation',data,view)


@pytest.mark.parametrize('text,issue',[
    ('An SOS was recorded at 14:21:05 IST.', 'unsupported_timestamp'),
    ('An SOS was recorded with a normal heart rate.', 'unsupported_clinical_classification'),
    ('An SOS was recorded at the last-known location.', 'unsupported_location_status'),
])
def test_timestamp_recombination_and_unsupported_interpretations_are_rejected(text,issue):
    data=IncidentBundle.model_validate(example());view=build_incident_view(data)
    draft=Narrative(text=text,evidence_ids=[e['id'] for e in view['evidence']])
    assert issue in validate_narrative(draft,'description',data,view)


def test_guarded_generation_rejects_unsafe_interpretation_without_substitution():
    class BadModel:
        def generate_contextual_incident(self, evidence):
            return json.dumps(draft({'text':'The wearer is safe.', 'evidence_ids':['B']}))
    with pytest.raises(BackendUnavailable, match='contextual_quality_rejected'):
        IncidentReportService(BadModel()).generate(IncidentBundle.model_validate(example()))


def test_no_sos_handling_or_missing_physiology_is_invented_for_exercise_case():
    data=IncidentBundle.model_validate(example(2));view=build_incident_view(data)
    draft=Narrative(text='Exercise is possible. SOS handling is retained; there is no HRV or physiology data.',evidence_ids=['C','P','B','L'])
    issues=validate_narrative(draft,'interpretation',data,view)
    assert 'unsupported_sos_action' in issues and 'unsupported_missing_physiology' in issues


@pytest.mark.parametrize('claim',['No emergency response was initiated.','No SOS handling was recorded.'])
def test_model_cannot_infer_whether_an_operational_response_happened(claim):
    data=IncidentBundle.model_validate(example());view=build_incident_view(data)
    draft=Narrative(text=claim,evidence_ids=['C','B'])
    assert 'unsupported_response_claim' in validate_narrative(draft,'interpretation',data,view)


def test_unreliable_audio_is_not_missing_and_absent_sos_is_not_invented():
    data=IncidentBundle.model_validate(example(4));view=build_incident_view(data)
    draft=Narrative(text='A recorded SOS was present, but missing audio prevented analysis.',evidence_ids=['C','A','B'])
    issues=validate_narrative(draft,'description',data,view)
    assert 'unsupported_missing_audio' in issues and 'unsupported_sos_action' in issues


@pytest.mark.parametrize('claim',['No location was provided.','No location or response was provided.'])
def test_last_known_location_is_not_erased_by_model_text(claim):
    data=IncidentBundle.model_validate(example(3));view=build_incident_view(data)
    draft=Narrative(text='An SOS was recorded. '+claim,evidence_ids=['B','G'])
    assert 'unsupported_missing_location' in validate_narrative(draft,'description',data,view)


def test_pdf_pack_preserves_supplied_fields_for_all_four_reference_cases():
    reports=[IncidentReportService(FixtureModel(), mode='raw_review').generate(IncidentBundle.model_validate(example(n))) for n in range(1,5)]
    reader=PdfReader(BytesIO(render_incident_pdf(reports)))
    text=' '.join(p.extract_text() for p in reader.pages)
    for title in ('INCIDENT INFORMATION','SENSOR FINDINGS','AUDIO TRANSCRIPT / SOURCE INFORMATION','SLM INTERPRETATION','EVIDENCE / FOLLOW-UP'):
        assert title in text
    for n in range(1,5): assert f'WLT-DEMO-{n:03}' in text
    assert '8 min 10 sec' in text and 'Officer Signature' in text


def test_source_database_is_read_only_and_id_lookup_is_parameterized(tmp_path,monkeypatch):
    path=tmp_path/'source.sqlite3';data=example()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE incident_source_records(incident_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL)')
        db.execute('INSERT INTO incident_source_records VALUES(?,?)',(data['incident_id'],json.dumps(data)))
    before=path.read_bytes();monkeypatch.setenv('WUALT_SOURCE_DB',str(path))
    loaded=load_from_source('database','DEMO-001')
    assert loaded.report_number=='WLT-DEMO-001' and path.read_bytes()==before
    monkeypatch.delenv('WUALT_SOURCE_API_BASE',raising=False)
    with pytest.raises(SourceUnavailable):load_from_source('server','DEMO-001')


def test_server_adapter_rejects_a_different_incident_id(monkeypatch):
    import wualt_slm.incident_sources as sources
    original=httpx.Client
    transport=httpx.MockTransport(lambda request:httpx.Response(200,json=example(2)))
    monkeypatch.setenv('WUALT_SOURCE_API_BASE','https://wualt.example')
    monkeypatch.setattr(sources.httpx,'Client',lambda **kwargs:original(transport=transport,**kwargs))
    with pytest.raises(SourceUnavailable):load_from_source('server','DEMO-001')


@pytest.fixture
def client(tmp_path,monkeypatch):
    import wualt_slm.incident_api as api
    monkeypatch.delenv('WUALT_API_KEY',raising=False)
    monkeypatch.setattr(api,'store',IncidentStore(tmp_path/'reports.sqlite3'))
    monkeypatch.setattr(api,'service',IncidentReportService(FixtureModel(),mode='raw_review'))
    with TestClient(api.app) as client:yield client


def test_active_application_generates_only_this_report_and_reuses_retries(client):
    assert 'Incident reports' in client.get('/').text
    for old in ['/studio','/v1/reports','/v1/app/sample-week','/v1/app/cards']:
        assert client.get(old).status_code==404
    first=client.post('/v1/incident-reports',json=example())
    assert first.status_code==200,first.text
    report=first.json();assert report['generation_mode']=='model_unvalidated'
    assert client.post('/v1/incident-reports',json=example()).json()['id']==report['id']
    changed=example();changed['physiology']['hr_peak_bpm']=129
    assert client.post('/v1/incident-reports',json=changed).status_code==409
    pdf=client.get('/v1/incident-reports/'+report['id']+'/pdf?download=true')
    assert pdf.status_code==200 and pdf.content.startswith(b'%PDF')
    assert pdf.headers['content-disposition'].startswith('attachment;')


def test_all_incident_record_and_pdf_routes_require_the_configured_key(client,monkeypatch):
    report=client.post('/v1/incident-reports',json=example()).json();monkeypatch.setenv('WUALT_API_KEY','test-secret')
    for url in ['/v1/incident-examples','/v1/incident-reports','/v1/incident-reports/'+report['id'],'/v1/incident-reports/'+report['id']+'/pdf']:
        assert client.get(url).status_code==401
    assert client.post('/v1/incident-reports',json=example()).status_code==401
    assert client.post('/v1/incident-reports/from-source',json={'source':'demo','incident_id':'DEMO-001','request_id':'retry'}).status_code==401
    assert client.get('/v1/incident-reports',headers={'X-API-Key':'test-secret'}).status_code==200
