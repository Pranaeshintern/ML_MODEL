"""Reusable application boundary: validated record in, report and PDF out."""
from collections.abc import Mapping
from typing import Any

from .backend import LlamaBackend
from .incident_schema import IncidentBundle
from .incident_service import IncidentReportService


class IncidentReporter:
    """Reuse one instance per process; the backend serializes inference.

    Generation returns an unsaved draft. The REST API additionally persists the
    source snapshot and applies request-id idempotency. Inspect generation_mode
    and validation_issues to distinguish model text from factual fallbacks.
    """

    def __init__(self, model_url: str | None = None, *, api_key: str | None = None, backend=None, mode: str | None = None):
        self.backend = backend if backend is not None else LlamaBackend(model_url, api_key=api_key)
        self.service = IncidentReportService(self.backend, mode=mode)

    def generate(self, payload: Mapping[str, Any] | IncidentBundle) -> dict:
        data = payload if isinstance(payload, IncidentBundle) else IncidentBundle.model_validate(payload)
        return self.service.generate(data)

    def generate_from_source(self, source: str, incident_id: str, request_id: str) -> dict:
        from .incident_sources import load_from_source
        from pydantic import TypeAdapter
        from .schemas import SourceId

        if source not in {"demo", "database", "server"}:
            raise ValueError("source must be demo, database or server")
        record = load_from_source(source, TypeAdapter(SourceId).validate_python(incident_id))
        record.request_id = TypeAdapter(SourceId).validate_python(request_id)
        return self.service.generate(record)

    @staticmethod
    def pdf(report: dict) -> bytes:
        from .incident_pdf import render_incident_pdf

        return render_incident_pdf(report)
