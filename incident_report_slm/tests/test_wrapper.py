from io import BytesIO
import json

from fastapi.testclient import TestClient
import httpx
import pytest
from pypdf import PdfReader

from wualt_slm import IncidentBundle, IncidentReporter
from wualt_slm.backend import BackendUnavailable, LlamaBackend
from wualt_slm.settings import ASSET_ROOT


class Offline:
    def generate_incident(self, evidence):
        raise BackendUnavailable("model_unavailable")


def fixture():
    return json.loads((ASSET_ROOT / "examples/demo-001.json").read_text(encoding="utf-8"))


def test_public_wrapper_returns_report_and_pdf_without_database_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reporter = IncidentReporter(backend=Offline())
    report = reporter.generate(fixture())
    assert report["generation_mode"] == "fallback"
    assert report["view"]["report_number"] == "WLT-DEMO-001"
    pdf = reporter.pdf(report)
    assert len(PdfReader(BytesIO(pdf)).pages) == 1
    assert list(tmp_path.iterdir()) == []
    assert reporter.generate(IncidentBundle.model_validate(fixture()))["view"]["colour"] == "RED"


def test_wrapper_rejects_unknown_source_and_bad_snapshot_ids(monkeypatch):
    import wualt_slm.incident_sources as sources
    monkeypatch.setattr(sources, "load_from_source", lambda *args: IncidentBundle.model_validate(fixture()))
    reporter = IncidentReporter(backend=Offline())
    with pytest.raises(ValueError):
        reporter.generate_from_source("unknown", "DEMO-001", "request-1")
    with pytest.raises(ValueError):
        reporter.generate_from_source("demo", "DEMO-001", "invalid/request")
    report = reporter.generate_from_source("demo", "DEMO-001", "request-1")
    assert report["input"]["request_id"] == "request-1"


def test_inference_uses_two_json_calls_and_keeps_keys_out_of_prompts(monkeypatch):
    import wualt_slm.backend as module
    captured = []
    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert request.headers["Authorization"] == "Bearer fake-test-key"
        assert "fake-test-key" not in request.content.decode()
        section = json.loads(payload["messages"][1]["content"])["section"]
        result = {"text": "Recorded SOS." if section == "description" else "Possible distress; urgent SOS handling is required.", "evidence_ids": ["B"] if section == "description" else ["C", "B"]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]})
    original = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    backend = LlamaBackend("http://model.test", api_key="fake-test-key")
    report = IncidentReporter(backend=backend).generate(fixture())
    assert len(captured) == 2 and report["generation_mode"] == "model"
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert all("Demo Guardian" not in json.dumps(p) for p in captured)


def test_incomplete_model_output_is_an_explicit_fallback(monkeypatch):
    import wualt_slm.backend as module
    original = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}))
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    report = IncidentReporter(model_url="http://model.test").generate(fixture())
    assert report["generation_mode"] == "fallback"
    assert "generation_incomplete" in report["validation_issues"]


def test_busy_backend_preserves_retry_signal():
    class Busy:
        def generate_incident(self, evidence):
            raise BackendUnavailable("model_busy")
    with pytest.raises(BackendUnavailable, match="model_busy"):
        IncidentReporter(backend=Busy()).generate(fixture())


def test_custom_ui_origin_and_bounded_body(monkeypatch):
    from wualt_slm.api import app
    monkeypatch.setenv("WUALT_ALLOWED_ORIGINS", "https://reports.example")
    with TestClient(app) as client:
        assert client.get("/", headers={"Origin": "https://reports.example"}).status_code == 200
        assert client.get("/", headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert client.post("/v1/incident-reports", content=b"x" * 65537).status_code == 413
