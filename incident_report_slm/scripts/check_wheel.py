"""Run with an installed wheel and python -I, from any directory."""
from io import BytesIO
from contextlib import chdir
import json
import os
from pathlib import Path
import tempfile


def main():
    with tempfile.TemporaryDirectory(prefix="wualt-wheel-") as directory, chdir(directory):
        os.environ["WUALT_DATA_DIR"] = directory
        os.environ.pop("WUALT_API_KEY", None)
        from fastapi.testclient import TestClient
        from pypdf import PdfReader
        from wualt_slm import IncidentReporter
        from wualt_slm.backend import BackendUnavailable
        from wualt_slm.settings import ASSET_ROOT

        class Offline:
            def generate_incident(self, evidence):
                raise BackendUnavailable("model_unavailable")

        payload = json.loads((ASSET_ROOT / "examples/demo-001.json").read_text(encoding="utf-8"))
        reporter = IncidentReporter(backend=Offline())
        report = reporter.generate(payload)
        assert report["view"]["colour"] == "RED"
        assert len(PdfReader(BytesIO(reporter.pdf(report))).pages) == 1
        import wualt_slm.incident_api as api
        api.service = reporter.service
        with TestClient(api.app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/static/app.js").status_code == 200
            assert len(client.get("/v1/incident-examples").json()["items"]) == 4
            saved = client.post("/v1/incident-reports", json=payload)
            assert saved.status_code == 200, saved.text
            assert client.get(f"/v1/incident-reports/{saved.json()['id']}/pdf").content.startswith(b"%PDF")
        assert (Path(directory) / "incident-reports.sqlite3").is_file()
        print("Installed wheel passed: assets, Python wrapper, PDF, REST and external data directory.")


if __name__ == "__main__":
    main()
