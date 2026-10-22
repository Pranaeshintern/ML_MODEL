import json
import pytest
from pydantic import ValidationError
from wualt_slm.backend import BackendUnavailable
from wualt_slm.incident_schema import IncidentBundle
from wualt_slm.incident_service import IncidentReportService
from wualt_slm.settings import ASSET_ROOT


def fixture():
    return IncidentBundle.model_validate_json((ASSET_ROOT/'examples/demo-001.json').read_text())


RAW={'description':{'text':'An SOS was recorded. The wearer is safe.','evidence_ids':['B']},
     'interpretation':{'text':'Possible distress is unconfirmed. Urgent SOS handling is unnecessary.','evidence_ids':['C','B','L']}}


class Model:
    def generate_incident(self,evidence):return json.dumps(RAW)


def test_review_preserves_exact_model_text_and_flags_without_replacement():
    report=IncidentReportService(Model(),mode='raw_review').generate(fixture())
    assert report['narrative']==RAW
    assert report['generation_mode']=='model_unvalidated'
    assert report['fallback_enabled'] is False
    assert report['validation_issues']
    assert report['status']=='unvalidated_draft'


@pytest.mark.parametrize('raw',['null','[]','{}','not-json',json.dumps({'description':RAW['description']})])
def test_bad_model_schema_is_failure_not_report(raw):
    class Broken:
        def generate_incident(self,evidence):return raw
    with pytest.raises(BackendUnavailable,match='invalid_narrative_schema'):
        IncidentReportService(Broken(),mode='raw_review').generate(fixture())


@pytest.mark.parametrize('reason',['model_busy','model_unavailable_or_invalid_response','generation_incomplete'])
def test_model_failure_never_substitutes_fallback(reason):
    class Offline:
        def generate_incident(self,evidence):raise BackendUnavailable(reason)
    with pytest.raises(BackendUnavailable,match=reason):
        IncidentReportService(Offline(),mode='raw_review').generate(fixture())


def test_source_schema_validation_still_applies():
    raw=json.loads((ASSET_ROOT/'examples/demo-001.json').read_text());raw['button']['at']=None
    with pytest.raises(ValidationError):IncidentBundle.model_validate(raw)


def test_default_guarded_mode_remains_available():
    report=IncidentReportService(Model(),mode='guarded').generate(fixture())
    assert report['narrative']['description']!=RAW['description']
    assert report['fallback_enabled'] is True


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):IncidentReportService(Model(),mode='typo')


@pytest.mark.parametrize('reason,status',[('model_unavailable',502),('model_busy',503)])
def test_raw_review_api_preserves_error_semantics(reason,status,tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import wualt_slm.incident_api as api
    from wualt_slm.incident_store import IncidentStore
    class Failed:
        def generate_incident(self,evidence):raise BackendUnavailable(reason)
    monkeypatch.delenv('WUALT_API_KEY',raising=False)
    monkeypatch.setattr(api,'service',IncidentReportService(Failed(),mode='raw_review'))
    monkeypatch.setattr(api,'store',IncidentStore(tmp_path/'raw.sqlite3'))
    with TestClient(api.app) as client:
        response=client.post('/v1/incident-reports',json=fixture().model_dump(mode='json'))
        assert response.status_code==status
        if status==503:assert response.headers['Retry-After']=='3'
        else:assert 'no fallback' in response.json()['detail']
        assert api.store.list()==[]


@pytest.mark.parametrize('synthetic',[False,True])
def test_raw_pdf_only_labels_synthetic_input_as_fictional(synthetic):
    from io import BytesIO
    from pypdf import PdfReader
    from wualt_slm.incident_pdf import render_incident_pdf
    data=fixture().model_copy(update={'synthetic':synthetic})
    report=IncidentReportService(Model(),mode='raw_review').generate(data)
    text=' '.join(p.extract_text() for p in PdfReader(BytesIO(render_incident_pdf(report))).pages)
    assert 'FALLBACK DISABLED' in text
    assert ('fictional test report' in text)==synthetic
