import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from .errors import Conflict, NotFound


class IncidentStore:
    def __init__(self,path):self.path=Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(self.path,timeout=10);db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE IF NOT EXISTS incident_reports(id TEXT PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,input_hash TEXT NOT NULL,body TEXT NOT NULL,content_hash TEXT NOT NULL)')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def input_hash(data):return hashlib.sha256(json.dumps(data.model_dump(mode='json'),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @staticmethod
    def unpack(row):
        if row is None:raise NotFound()
        return {**json.loads(row['body']),'content_hash':row['content_hash']}

    def find(self,request_id,input_hash):
        with self.connect() as db:row=db.execute('SELECT * FROM incident_reports WHERE request_id=?',(request_id,)).fetchone()
        if row is None:return None
        if row['input_hash']!=input_hash:raise Conflict('This request_id already belongs to different incident data. Use a new ID for a new snapshot.')
        return self.unpack(row)

    def save(self,report,input_hash):
        body=json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        content_hash=hashlib.sha256(body.encode()).hexdigest()
        with self.connect() as db:db.execute('INSERT INTO incident_reports VALUES(?,?,?,?,?)',(report['id'],report['input']['request_id'],input_hash,body,content_hash))
        return self.get(report['id'])

    def get(self,report_id):
        with self.connect() as db:return self.unpack(db.execute('SELECT * FROM incident_reports WHERE id=?',(report_id,)).fetchone())

    def list(self):
        with self.connect() as db:rows=db.execute('SELECT * FROM incident_reports ORDER BY rowid DESC LIMIT 50').fetchall()
        return [{'id':r['id'],'report_number':(body:=json.loads(r['body']))['view']['report_number'],
                 'colour':body['view']['colour'],'incident_type':body['input']['incident_type'],'generated_at':body['generated_at']} for r in rows]
