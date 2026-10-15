"""Read-only adapters. Credentials/URLs are server config, never model output."""
import json
from contextlib import closing
import os
from pathlib import Path
import sqlite3
from urllib.parse import quote
import httpx
from .incident_schema import IncidentBundle
from .settings import ASSET_ROOT, DATA_ROOT


class SourceUnavailable(Exception):pass
class SourceNotFound(Exception):pass


def demo_cases():
    return [IncidentBundle.model_validate_json(p.read_text(encoding='utf-8')) for p in sorted((ASSET_ROOT/'examples').glob('*.json'))]


def seed_demo_database():
    path=DATA_ROOT/'incident-demo-source.sqlite3';path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS incident_source_records(incident_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL)')
        for case in demo_cases():
            db.execute('INSERT OR IGNORE INTO incident_source_records VALUES(?,?)',(case.incident_id,case.model_dump_json()))
        db.commit()
    return path


def load_from_source(source,incident_id):
    if source=='server':
        base=os.getenv('WUALT_SOURCE_API_BASE','').rstrip('/')
        if not base:raise SourceUnavailable('WUALT_SOURCE_API_BASE is not configured.')
        headers={}
        if key:=os.getenv('WUALT_SOURCE_API_TOKEN'):headers['Authorization']='Bearer '+key
        try:
            with httpx.Client(timeout=15,trust_env=False,follow_redirects=False) as client:
                response=client.get(base+'/incidents/'+quote(incident_id,safe=''),headers=headers)
                if response.status_code==404:raise SourceNotFound()
                response.raise_for_status()
                raw=response.content
                if len(raw)>65536:raise SourceUnavailable('Upstream incident exceeds 64 KiB.')
                bundle=IncidentBundle.model_validate_json(raw)
        except (httpx.HTTPError,ValueError) as exc:raise SourceUnavailable('The configured server did not return a valid incident record.') from exc
    else:
        if source=='demo':path=seed_demo_database()
        else:
            configured=os.getenv('WUALT_SOURCE_DB')
            if not configured:raise SourceUnavailable('WUALT_SOURCE_DB is not configured.')
            path=Path(configured).resolve()
        if not path.is_file():raise SourceUnavailable('Configured source database is unavailable.')
        try:
            with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
                row=db.execute('SELECT payload_json FROM incident_source_records WHERE incident_id=?',(incident_id,)).fetchone()
            if row is None:raise SourceNotFound()
            if len(row[0].encode('utf-8'))>65536:raise SourceUnavailable('Stored incident exceeds 64 KiB.')
            bundle=IncidentBundle.model_validate_json(row[0])
        except (sqlite3.Error,ValueError) as exc:raise SourceUnavailable('The source database does not contain a valid incident record in the expected schema.') from exc
    if bundle.incident_id!=incident_id:raise SourceUnavailable('Source incident ID does not match the requested record.')
    return bundle
