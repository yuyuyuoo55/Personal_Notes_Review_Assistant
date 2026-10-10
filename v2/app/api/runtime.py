"""共享依赖、线程调度及统一状态；默认单用户本地环境。"""
import asyncio
import os
from uuid import uuid4
from pathlib import Path
from threading import Lock

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ..model import create_model
from ..quiz.store import QuizStore
from ..rag.vector_retrieval import create_vector_store
from .credentials import KeyManager, WindowsCredentials
from .balance import fetch_balance


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


class Runtime:
    """可注入离线模型/存储；连接延迟创建，不在健康检查时调用模型。"""
    def __init__(self, *, store_factory=create_vector_store, model_factory=create_model,
                 vision=None, upload_root=None, timeout=180, keys=None, settings_token=None, test_mode=False,
                 balance_reader=fetch_balance, web_search=None):
        self.settings_token = settings_token if settings_token is not None else os.getenv("V2_SETTINGS_TOKEN", "")
        self.keys = keys or KeyManager(WindowsCredentials() if self.settings_token and os.name == "nt" else None)
        self.balance_reader = balance_reader
        self.web_search = web_search
        self.instance_id = uuid4().hex
        self.test_mode = test_mode
        self.store_factory = self.configured_store if store_factory is create_vector_store else store_factory
        self.model_factory = (lambda: create_model(api_key=self.keys.effective("deepseek"))) if model_factory is create_model else model_factory
        if vision is None:
            from ..integrations.vision import describe_image
            self.vision = lambda data, mime: describe_image(data, mime, api_key=self.keys.effective("deepseek"))
        else:
            self.vision = vision
        self.upload_root = Path(upload_root or Path(__file__).resolve().parents[2] / "data" / "uploads").resolve()
        self.timeout = timeout
        self.quiz_store = QuizStore()
        self._store = None
        self._init_lock = Lock()
        self.job_lock = Lock()
        self._jobs = set()

    def configured_store(self):
        return create_vector_store(api_key=self.keys.effective("dashscope"))

    def invalidate_embedding(self, provider):
        if provider == "dashscope":
            # 仅释放客户端缓存，不改变 Collection 或已有向量。
            with self._init_lock:
                self._store = None

    def store(self):
        with self._init_lock:
            if self._store is None:
                self._store = self.store_factory()
            return self._store


async def run_job(runtime, function, *args, **kwargs):
    """【业务需求】同步模型/导入任务不占用事件循环，超时不假装成功。
    【输入】运行时、同步函数及参数。
    【执行步骤】线程中限制一次业务任务；异步等待限时；安全映射异常。
    【输出】真实业务返回值；超时返回状态未知，后台线程可能仍在收尾。
    """
    if not runtime.job_lock.acquire(blocking=False):
        raise ApiError(409, "busy", "已有业务任务正在处理，请稍后重试。")
    def work():
        try:
            return function(*args, **kwargs)
        finally:
            runtime.job_lock.release()
    future = asyncio.get_running_loop().run_in_executor(None, work)
    runtime._jobs.add(future)
    def finished(job):
        runtime._jobs.discard(job)
        if not job.cancelled():
            job.exception()  # 超时请求已离开时也回收后台异常，不输出敏感正文。
    future.add_done_callback(finished)
    try:
        # 不取消尚未启动的线程任务，确保它最终执行finally解除占用。
        return await asyncio.wait_for(asyncio.shield(future), timeout=runtime.timeout)
    except TimeoutError:
        raise ApiError(504, "timeout", "处理超时，任务可能仍在收尾；请查询结果，不要立即重复提交。") from None
    except ApiError:
        raise
    except Exception:
        raise ApiError(500, "internal_error", "处理失败，请检查本地配置或服务状态。") from None


def runtime_from(request: Request):
    return getattr(request.state, "runtime", None) or request.app.state.runtime


def response(data, *, http_status=None):
    """如实保留业务状态；统一外层status/data，不将partial标为完整成功。"""
    value = jsonable_encoder(data)
    status = value.get("status", "ok") if isinstance(value, dict) else "ok"
    mapping = {"invalid_request": 422, "quiz_not_found": 404, "question_not_found": 404,
               "submission_conflict": 409, "in_progress": 202, "model_error": 502,
               "invalid_model_output": 502, "failed": 422, "chapter_unavailable": 422}
    return JSONResponse({"status": status, "data": value}, status_code=http_status or mapping.get(status, 200))


class LazyQuizModel:
    """只有业务确需结构化模型时才初始化；缓存选择题/报告读取不调用工厂。"""
    def __init__(self, factory):
        self.factory = factory
        self.model = None

    def with_structured_output(self, schema, method):
        if self.model is None:
            self.model = self.factory()
        return self.model.with_structured_output(schema, method=method)


class DeferredGraphStore:
    """只在Graph实际检索时打开库，Daily不依赖Embedding配置。"""
    def __init__(self, runtime):
        self.runtime = runtime

    def get(self, **kwargs):
        return self.runtime.store().get(**kwargs)

    def similarity_search_with_score(self, *args, **kwargs):
        return self.runtime.store().similarity_search_with_score(*args, **kwargs)
