"""Phase18 FastAPI入口：本地单用户，单进程运行。"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from . import chat, notes, quiz, settings
from .runtime import ApiError, Runtime, runtime_from


class UploadBodyLimit:
    """在multipart解析前限制上传请求体，避免无限占用临时磁盘。"""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        limit = 101 * 1024 * 1024
        if scope['type'] != 'http' or scope.get('path') != '/notes/upload':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        length = headers.get(b'content-length')
        if length and (not length.isdigit() or int(length) > limit):
            return await JSONResponse({'status':'file_too_large','data':None,'message':'上传请求体超限。'},status_code=413)(scope,receive,send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body',b''))
            if total > limit:
                raise HTTPException(413, '上传请求体超限。')
            return message
        await self.app(scope, limited_receive, send)


def create_app(runtime=None):
    """【业务需求】组装V2 HTTP入口，不在启动时调用模型。
    【输入】可注入Runtime，缺省使用本地正式配置。
    【执行步骤】注册安全错误处理、上传限制和业务路由。
    【输出】FastAPI应用，Swagger/OpenAPI自动生成。
    """
    app = FastAPI(title='个人笔记复习助手 V2', version='20.0')
    app.state.runtime = runtime or Runtime()
    app.add_middleware(UploadBodyLimit)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError):
        return JSONResponse({'status':error.code,'data':None,'message':error.message},status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # 不回显输入值/异常上下文，避免错误响应泄露敏感文本。
        fields = [{'location':list(e['loc']),'type':e['type']} for e in error.errors()]
        return JSONResponse({'status':'invalid_request','data':None,'message':'请求参数无效。','fields':fields},status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        return JSONResponse({'status':'http_error','data':None,'message':'请求未完成。'},status_code=error.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        return JSONResponse({'status':'internal_error','data':None,'message':'服务处理失败。'},status_code=500)

    @app.get('/health', tags=['health'])
    async def health(request: Request):
        import os
        runtime = runtime_from(request)
        return {'status':'ok','data':{'service':'v2','quiz_store':'in_process','single_user':True,
            'instance_id':runtime.instance_id,'test_mode':runtime.test_mode,
            'embedding_provider':'fake' if runtime.test_mode else 'dashscope',
            'collection':'e2e_notes' if runtime.test_mode else 'v2_notes',
            'text_model':'Fake Model' if runtime.test_mode else os.getenv('DEEPSEEK_TEXT_MODEL','deepseek-v4-flash'),
            'vision_model':'Fake Vision' if runtime.test_mode else os.getenv('DEEPSEEK_VISION_MODEL','deepseek-flash'),
            'embedding_model':'Mock Embedding' if runtime.test_mode else 'text-embedding-v4',
            'settings_enabled':bool(runtime.settings_token and runtime.keys.storage),
            'cloud_demo':bool(getattr(runtime,'cloud_demo',False))}}

    app.include_router(notes.router)
    app.include_router(chat.router)
    app.include_router(quiz.router)
    app.include_router(settings.router)
    return app


app = create_app()
