import json
import os
import secrets
from uuid import uuid4
from fastapi import FastAPI, Depends, Header, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from .domain import EngineError
from .runtime import Workspace
from .synthetic.service import SyntheticService
from .synthetic.api import register_routes

def create_app(root=None,auth=None,*,synthetic_provider=None,simulator=None):
    workspace=Workspace(root or os.getenv('WHITE_RENDER_ROOT','work'))
    service=SyntheticService(workspace,provider=synthetic_provider,simulator=simulator)
    credentials=auth if auth is not None else json.loads(os.getenv('WHITE_RENDER_AUTH_JSON','{}'))
    app=FastAPI(title='白模与视频渲染后端',version='1.0.0')
    app.state.synthetic=service
    origins=[x.strip() for x in os.getenv('WHITE_RENDER_CORS_ORIGINS','').split(',') if x.strip()]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','HEAD','POST','OPTIONS'],
                          allow_headers=['Authorization','Content-Type','Idempotency-Key','Range'],
                          expose_headers=['X-Request-ID','Content-Range','Content-Length','Accept-Ranges'])
    @app.middleware('http')
    async def requests(request:Request,call_next):
        request.state.request_id=uuid4().hex
        response=await call_next(request);response.headers['X-Request-ID']=request.state.request_id
        return response
    def error(request,status,code,message,details=None,headers=None):
        value={'error':{'code':code,'message':message},'request_id':request.state.request_id}
        if details is not None:value['error']['details']=details
        return JSONResponse(status_code=status,content=value,headers=headers)
    @app.exception_handler(EngineError)
    async def domain(request,exc):return error(request,exc.status,exc.code,exc.message)
    @app.exception_handler(StarletteHTTPException)
    async def http(request,exc):return error(request,exc.status_code,'HTTP_'+str(exc.status_code),str(exc.detail),headers=exc.headers)
    @app.exception_handler(RequestValidationError)
    async def validation(request,exc):
        return error(request,422,'REQUEST_VALIDATION','Invalid request',
                     [{'location':list(e['loc']),'message':e['msg'],'type':e['type']} for e in exc.errors()])
    def identity(authorization:str|None=Header(default=None)):
        if not credentials:raise HTTPException(503,'Backend authentication not configured')
        token=(authorization or '').removeprefix('Bearer ')
        user=next((v for k,v in credentials.items() if secrets.compare_digest(k,token)),None)
        if user is None:raise HTTPException(401,'Invalid bearer token')
        return user
    def access(project,user,roles):
        if user.get('project')!=project or not set(user.get('roles',[]))&set(roles):raise HTTPException(403,'Project or role denied')
    @app.get('/healthz')
    def health():return {'status':'ok','service':'white-render','version':'1.0.0'}
    @app.post('/api/projects/{project}/uploads')
    def upload(project:str,file:UploadFile=File(...),user=Depends(identity)):
        access(project,user,['editor'])
        return service.register_upload(project,workspace.upload(project,file.filename,file.file))
    @app.get('/api/projects/{project}/jobs/{job_id}')
    def job(project:str,job_id:str,user=Depends(identity)):
        access(project,user,['viewer','editor','publisher'])
        return service.get_job(project,job_id)
    @app.post('/api/projects/{project}/jobs/{job_id}/{action}')
    def action(project:str,job_id:str,action:str,user=Depends(identity)):
        access(project,user,['editor'])
        return service.action(project,job_id,action)
    register_routes(app,service,identity,access)
    return app
