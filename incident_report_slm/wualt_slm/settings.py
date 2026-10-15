"""Installed assets are read-only; report data lives outside the package."""
import os
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
PROJECT_ROOT = Path(os.getenv("WUALT_PROJECT_ROOT", str(Path.cwd()))).resolve()
DATA_ROOT = Path(os.getenv("WUALT_DATA_DIR", str(Path.home() / ".wualt-slm" / "data"))).resolve()


def allowed_origins():
    configured = os.getenv("WUALT_ALLOWED_ORIGINS", "").strip()
    if configured:
        return {origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()}
    return {"http://127.0.0.1:8000", "http://localhost:8000"}
