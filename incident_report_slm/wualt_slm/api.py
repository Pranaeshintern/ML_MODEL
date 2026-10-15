"""ASGI entry point: uvicorn wualt_slm.api:app --workers 1."""
from .incident_api import app

__all__ = ["app"]
