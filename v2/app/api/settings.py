"""仅可信本机使用的密钥管理；实际验证须显式费用同意。"""
import hmac
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from pydantic import Field, SecretStr

from .runtime import ApiError, run_job, runtime_from
from .schemas import RequestData

router = APIRouter(prefix="/settings", tags=["settings"])
Provider = Literal["deepseek", "dashscope"]


class KeySubmission(RequestData):
    api_key: SecretStr = Field(min_length=1, max_length=512)
    paid_validation_consent: bool
    max_requests: Literal[1]


class DeleteSubmission(RequestData):
    confirm: Literal[True]


def guard(request):
    runtime = runtime_from(request)
    hostname = urlsplit("http://" + request.headers.get("host", "")).hostname
    origin = request.headers.get("origin")
    if not runtime.settings_token or runtime.keys.storage is None:
        raise ApiError(503, "settings_disabled", "安全设置未启用，请使用本机 start_v2.bat 启动。")
    if (not request.client or request.client.host not in {"127.0.0.1", "::1"}
        or hostname not in {"127.0.0.1", "localhost", "::1"}
        or (origin and urlsplit(origin).hostname not in {"127.0.0.1", "localhost", "::1"})
        or not hmac.compare_digest(request.headers.get("x-v2-settings-token", ""), runtime.settings_token)):
        raise ApiError(403, "local_only", "仅允许可信本机应用管理配置。")
    return runtime


@router.get("/providers")
async def providers(request: Request):
    runtime = guard(request)
    return {"status":"ok", "data":{name:runtime.keys.status(name) for name in ("deepseek","dashscope")}}


@router.get("/deepseek/balance")
async def balance(request: Request):
    """【业务需求】可信本机主动查询余额，不返回密钥。
    【输入】受本机令牌保护的请求，不接受前端指定密钥或查询端点。
    【执行步骤】复用配置优先级；在线程中读取官方余额；失败返回安全状态。
    【输出】实际金额、币种和查询时间；未配置/失败均不伪造余额。
    """
    runtime = guard(request)
    def perform():
        secret = runtime.keys.effective("deepseek")
        if not secret:
            raise ApiError(422, "not_configured", "请先配置 DeepSeek，再查询余额。")
        try:
            return runtime.balance_reader(secret)
        except Exception:
            raise ApiError(502, "balance_failed", "余额查询失败，请检查配置、权限或网络；未自动重试。") from None
    return {"status":"ok", "data":await run_job(runtime, perform)}


@router.put("/providers/{provider}")
async def save(request: Request, provider: Provider, payload: KeySubmission):
    runtime = guard(request)
    if not payload.paid_validation_consent:
        raise ApiError(422, "consent_required", "验证会请求真实服务，须明确同意一次调用及费用。")
    def perform():
        try:
            result = runtime.keys.save(provider, payload.api_key.get_secret_value())
        except Exception:
            raise ApiError(502, "validation_failed", "验证或安全保存失败，原有效配置已保留。请检查密钥、权限和网络。") from None
        runtime.invalidate_embedding(provider)
        return result
    return {"status":"saved", "data":await run_job(runtime, perform)}


@router.delete("/providers/{provider}")
async def delete(request: Request, provider: Provider, payload: DeleteSubmission):
    runtime = guard(request)
    def perform():
        result = runtime.keys.delete(provider)
        runtime.invalidate_embedding(provider)
        return result
    return {"status":"deleted", "data":await run_job(runtime, perform)}
