import os
import secrets
from threading import Lock
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .backend import LlamaBackend, BackendUnavailable
from .signal_schema import InterpretationInput, InterpretationResponse
from .signal_interpretation import SignalInterpreter
from .settings import ASSET_ROOT, allowed_origins

backend = LlamaBackend()
generation_lock = Lock()
app = FastAPI(title='WUALT Signal Interpretation', version='0.20.2', description='Compact backend signals in; separate audio and distress interpretations out.')


def authenticate(x_api_key: str | None = Header(default=None)):
    configured = os.getenv('WUALT_API_KEY')
    if configured and not secrets.compare_digest(configured.encode(), (x_api_key or '').encode()):
        raise HTTPException(401, 'Enter the workspace API key.')


router = APIRouter(prefix='/v1', dependencies=[Depends(authenticate)])


@app.middleware('http')
async def bounded_request(request, call_next):
    if request.headers.get('origin') and request.headers['origin'] not in allowed_origins():
        return JSONResponse({'detail': 'Origin not allowed'}, status_code=403)
    if request.method in {'POST', 'PUT', 'PATCH'}:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse({'detail': 'Request exceeds 64 KiB'}, status_code=413)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    if request.url.path == '/' or request.url.path.startswith('/static/'):
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; frame-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Explain field failures without returning submitted values or exception context.
    return JSONResponse({'detail': [{'location': list(e['loc']), 'type': e['type'], 'message': e['msg']} for e in exc.errors()]}, status_code=422)


@app.get('/health')
def health():
    return {'api': 'ready', 'model_ready': backend.ready(), 'prototype': True, 'service': 'wualt-incident-slm', 'workflow': 'interpretation-only', 'version': '0.20.2', 'schemas': ['interpretation-v1'], 'auth_required': bool(os.getenv('WUALT_API_KEY'))}


@router.get('/interpretation-input-example')
def interpretation_example():
    return InterpretationInput.model_validate_json((ASSET_ROOT / 'examples/interpretation-v1.json').read_text(encoding='utf-8')).model_dump(mode='json', exclude_none=True)


@router.post('/interpretations', response_model=InterpretationResponse)
def interpret_signals(data: InterpretationInput):
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(503, 'The model is busy. Retry later.', headers={'Retry-After': '3'})
    try:
        try:
            result = SignalInterpreter(backend=backend).interpret(data)
            if not any(result[name] is not None for name in ("audio_interpretation", "distress_interpretation")) and any(section["status"] == "failed" for section in result["sections"].values()):
                return JSONResponse(result, status_code=502)
            return result
        except BackendUnavailable as exc:
            if str(exc) == 'model_busy':
                raise HTTPException(503, 'The model is busy. Retry later.', headers={'Retry-After': '3'})
            raise HTTPException(502, 'Model generation failed; no fallback was substituted.')
    finally:
        generation_lock.release()


app.include_router(router)
app.mount('/static', StaticFiles(directory=ASSET_ROOT / 'web'), name='static')


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ASSET_ROOT / 'web/interpretation.html')
