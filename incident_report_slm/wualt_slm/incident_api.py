import os
import secrets
from threading import Lock
from typing import Literal
from uuid import UUID
from fastapi import APIRouter,Depends,FastAPI,Header,HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from .backend import LlamaBackend,BackendUnavailable
from .incident_schema import IncidentBundle
from .incident_service import IncidentReportService
from .incident_store import IncidentStore
from .incident_sources import demo_cases,load_from_source,SourceNotFound,SourceUnavailable
from .incident_pdf import render_incident_pdf
from .errors import Conflict,NotFound
from .schemas import SourceId,StrictModel
from .settings import ASSET_ROOT,DATA_ROOT,allowed_origins

backend=LlamaBackend();service=IncidentReportService(backend);store=IncidentStore(DATA_ROOT/'incident-reports.sqlite3');generation_lock=Lock()
app=FastAPI(title='WUALT Incident Reports',version='0.11.0',description='Generate incident-v1 reference reports and incident-v2 contextual reports from physiology, transcript, keywords and optional activity/chat.')


def authenticate(x_api_key:str|None=Header(default=None)):
    configured=os.getenv('WUALT_API_KEY')
    if configured and not secrets.compare_digest(configured.encode(),(x_api_key or '').encode()):raise HTTPException(401,'Enter the workspace API key.')


router=APIRouter(prefix='/v1',dependencies=[Depends(authenticate)])


@app.middleware('http')
async def bounded_request(request,call_next):
    if request.headers.get('origin') and request.headers['origin'] not in allowed_origins():
        return JSONResponse({'detail':'Origin not allowed'},status_code=403)
    if request.method in {'POST','PUT','PATCH'}:
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>65536:return JSONResponse({'detail':'Request exceeds 64 KiB'},status_code=413)
        request._body=bytes(body)
    response=await call_next(request)
    response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer'
    if request.url.path=='/' or request.url.path.startswith('/static/'):
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; frame-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
    # Explain field failures without returning submitted values or exception context.
    return JSONResponse({'detail':[{'location':list(e['loc']),'type':e['type'],'message':e['msg']}
                                   for e in exc.errors()]},status_code=422)


@app.exception_handler(Conflict)
async def conflict(request,exc):return JSONResponse({'detail':str(exc)},status_code=409)


@app.exception_handler(NotFound)
async def missing(request,exc):return JSONResponse({'detail':'Report not found.'},status_code=404)


@app.get('/health')
def health():return {'api':'ready','model_ready':backend.ready(),'prototype':True,'service':'wualt-incident-slm','workflow':'incident-report-only','version':'0.11.0','schemas':['incident-v1','incident-v2'],'report_mode':service.mode,'auth_required':bool(os.getenv('WUALT_API_KEY')),'database_configured':bool(os.getenv('WUALT_SOURCE_DB')),'server_configured':bool(os.getenv('WUALT_SOURCE_API_BASE'))}


@router.get('/incident-examples')
def examples():return {'synthetic':True,'items':[{'incident_id':c.incident_id,'report_number':c.report_number,'incident_type':c.incident_type} for c in demo_cases()]}


@router.get('/incident-input-example')
def input_example():
    return IncidentBundle.model_validate_json((ASSET_ROOT/'examples/physiology-v2.json').read_text(encoding='utf-8')).model_dump(mode='json')


def create(data):
    digest=store.input_hash(data)
    if existing:=store.find(data.request_id,digest):return {**existing,'reused':True}
    if not generation_lock.acquire(blocking=False):raise HTTPException(503,'A report is being generated. Retry the same request ID.',headers={'Retry-After':'3'})
    try:
        if existing:=store.find(data.request_id,digest):return {**existing,'reused':True}
        try:report=service.generate(data)
        except BackendUnavailable as exc:
            if str(exc)=='model_busy':raise HTTPException(503,'The model is busy. Retry the same request ID.',headers={'Retry-After':'3'})
            raise HTTPException(502,'Model generation failed; no fallback was substituted.')
        render_incident_pdf(report)
        return {**store.save(report,digest),'reused':False}
    finally:generation_lock.release()


@router.post('/incident-reports')
def generate_report(data:IncidentBundle):return create(data)


class SourceRequest(StrictModel):
    source:Literal['demo','database','server']='demo'
    incident_id:SourceId
    request_id:SourceId


@router.post('/incident-reports/from-source')
def from_source(request:SourceRequest):
    try:data=load_from_source(request.source,request.incident_id)
    except SourceNotFound:raise HTTPException(404,'Incident not found in the selected source.')
    except SourceUnavailable as exc:raise HTTPException(503,str(exc))
    data.request_id=request.request_id
    return create(data)


@router.get('/incident-reports')
def history():return {'items':store.list()}


@router.get('/incident-reports/{report_id}')
def get_report(report_id:UUID):return store.get(str(report_id))


@router.get('/incident-reports/{report_id}/pdf')
def pdf(report_id:UUID,download:bool=False):
    report=store.get(str(report_id));content=render_incident_pdf(report)
    return Response(content,media_type='application/pdf',headers={'Content-Disposition':('attachment' if download else 'inline')+f'; filename="wualt-incident-{report_id}.pdf"'})


app.include_router(router)
app.mount('/static',StaticFiles(directory=ASSET_ROOT/'web'),name='static')


@app.get('/',include_in_schema=False)
def home():return FileResponse(ASSET_ROOT/'web/index.html')
