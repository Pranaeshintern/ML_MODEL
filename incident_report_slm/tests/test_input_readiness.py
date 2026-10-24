import json

import httpx
import pytest
from pydantic import ValidationError

from wualt_slm import IncidentBundle
from wualt_slm.backend import LlamaBackend
from wualt_slm.settings import ASSET_ROOT


@pytest.mark.parametrize('status,body,expected', [
    (200, '{"status":"ok"}', True),
    (200, '<html><title>502 | Runpod</title></html>', False),
    (200, '{"status":"loading model"}', False),
    (503, '{"status":"ok"}', False),
    (200, '[]', False),
])
def test_readiness_requires_model_health_not_a_proxy_page(monkeypatch,status,body,expected):
    client=httpx.Client
    monkeypatch.setattr('wualt_slm.backend.httpx.Client',lambda **kwargs:client(
        transport=httpx.MockTransport(lambda request:httpx.Response(status,text=body)),**kwargs))
    assert LlamaBackend('http://test.invalid').ready() is expected


@pytest.mark.parametrize('field',['transcript','chat','activity'])
def test_whitespace_is_not_usable_context(field):
    data=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text())
    if field=='transcript': data['audio']['transcript'][0]['text']=' \n\t '
    elif field=='chat': data['chat_context']['messages'][0]['text']=' \n\t '
    else: data['activity']['label']=' \n\t '
    with pytest.raises(ValidationError): IncidentBundle.model_validate(data)


def test_valid_source_text_is_preserved_exactly():
    data=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text())
    original='  I am waiting, not running.  '
    data['audio']['transcript'][0]['text']=original
    assert IncidentBundle.model_validate(data).audio.transcript[0].text==original


def test_v2_errors_point_to_the_transcript_field_without_legacy_audio_requirements(monkeypatch):
    from fastapi.testclient import TestClient
    from wualt_slm.incident_api import app
    monkeypatch.delenv('WUALT_API_KEY',raising=False)
    data=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text())
    data['audio']['transcript'][0]['at']='PRIVATE_INVALID_VALUE'
    with TestClient(app) as client:
        response=client.post('/v1/incident-reports',json=data)
    assert response.status_code==422
    errors=response.json()['detail']
    assert len(errors)==1
    assert errors[0]['location']==['body','audio','transcript',0,'at']
    assert errors[0]['message']
    assert 'PRIVATE_INVALID_VALUE' not in response.text
    assert 'received' not in response.text
