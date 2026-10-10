"""Streamlit 云端临时演示：独立资料与 DeepSeek，会话可使用明确配置的演示者服务。"""
import secrets
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from threading import RLock, Lock

from starlette.responses import JSONResponse

from .credentials import CredentialRecord, KeyManager
from .runtime import Runtime, ApiError
from ..rag.vector_retrieval import create_vector_store
from ..integrations.tavily_mcp import search_web


class MemoryCredentials:
    """凭据仅在当前会话内存保存，退出时清空。"""
    def __init__(self):
        self.rows = {}

    def read(self, name):
        return self.rows.get(name)

    def write(self, name, secret, metadata):
        self.rows[name] = CredentialRecord(secret, dict(metadata))

    def delete(self, name):
        self.rows.pop(name, None)


class CloudKeys(KeyManager):
    """演示者提供百炼；访客只管理自己的 DeepSeek，状态不暴露共享密钥。"""
    def __init__(self, storage, defaults):
        super().__init__(storage, environment={}, storage_source="session_memory")
        self.defaults = defaults

    def effective(self, provider):
        if provider == "dashscope":
            return self.defaults.get("DASHSCOPE_API_KEY", "")
        return super().effective(provider)

    def status(self, provider):
        if provider == "dashscope":
            return {"provider":provider, "configured":bool(self.effective(provider)),
                    "source":"demo_host", "masked_key":"", "validation_state":"not_verified",
                    "capabilities":[], "validated_at":None}
        return super().status(provider)


class CloudSessions:
    """【业务需求】匿名浏览器之间不共享密钥、文件、知识库或小测。
    【输入】会话容量、过期时间及可注入离线 Runtime 工厂。
    【执行步骤】随机 token 建立独立运行时；请求续期；闲置且无请求时清理。
    【输出】只允许已注册 token 获取自己的 Runtime。
    """
    def __init__(self, *, max_sessions=8, ttl=1800, runtime_factory=None, clock=time.monotonic, sponsor_keys=None):
        self.root = Path(tempfile.mkdtemp(prefix="v2-cloud-")).resolve()
        self.max_sessions, self.ttl = max_sessions, ttl
        self.runtime_factory, self.clock = runtime_factory, clock
        self.rows = {}
        self.lock = RLock()
        # 单进程共享重精排模型；同时只做一个业务任务，限制演示资源峰值。
        self.job_lock = Lock()
        self.chroma_client = None
        self.sponsor_keys = sponsor_keys

    def register(self):
        with self.lock:
            self.prune()
            if len(self.rows) >= self.max_sessions:
                raise ApiError(503, "capacity", "演示会话已满，请稍后再试。")
            token = secrets.token_urlsafe(32)
            session_id = secrets.token_hex(16)
            credentials = MemoryCredentials()
            keys = KeyManager(credentials, environment={}, storage_source="session_memory")
            if self.sponsor_keys is not None:
                keys = CloudKeys(credentials, self.sponsor_keys)
            runtime = (self.runtime_factory(keys, token) if self.runtime_factory else
                       Runtime(keys=keys, settings_token=token))
            runtime.cloud_demo = True
            runtime.cloud_sponsored = self.sponsor_keys is not None
            runtime.job_lock = self.job_lock
            runtime.upload_root = self.root / "uploads" / session_id
            collection = "session_" + session_id
            if self.runtime_factory is None:
                def store_factory():
                    key = keys.effective("dashscope")
                    if not key:
                        raise ApiError(422, "not_configured", "演示者尚未配置百炼服务，请联系演示者。" if runtime.cloud_sponsored else "请先在设置中配置当前会话的百炼 Key。")
                    # 共享临时数据库连接，集合按会话隔离；只使用明确注入的百炼配置。
                    import chromadb
                    self.chroma_client = chromadb.PersistentClient(path=str(self.root / "chroma"))
                    return create_vector_store(self.root / "chroma", api_key=key, collection_name=collection)
                runtime.store_factory = store_factory
            async def session_search(query):
                if self.sponsor_keys is not None:
                    return await search_web(query, api_key=self.sponsor_keys.get("TAVILY_API_KEY", ""))
                return await search_web(query, api_key=credentials.read("tavily").secret if credentials.read("tavily") else "")
            runtime.web_search = session_search
            self.rows[token] = {"runtime":runtime, "credentials":credentials, "collection":collection,
                                "seen":self.clock(), "active":0, "requests":deque()}
            return token

    def acquire(self, token):
        with self.lock:
            row = self.rows.get(token)
            if not row or self.clock() - row["seen"] > self.ttl:
                raise ApiError(401, "session_expired", "演示会话已失效，请刷新页面重新建立。")
            now = self.clock()
            requests = row["requests"]
            while requests and now - requests[0] > 60:
                requests.popleft()
            if len(requests) >= 90:
                raise ApiError(429, "rate_limit", "请求过于频繁，请稍后再试。")
            requests.append(now)
            row["seen"] = now
            row["active"] += 1
            return row["runtime"]

    def release(self, token):
        with self.lock:
            if token in self.rows:
                self.rows[token]["active"] -= 1

    def valid(self, token):
        with self.lock:
            row = self.rows.get(token)
            return bool(row and self.clock() - row["seen"] <= self.ttl)

    def set_tavily(self, token, key):
        with self.lock:
            if self.sponsor_keys is not None:
                raise ApiError(403, "host_managed", "联网服务由演示者管理。")
            if not self.valid(token):
                raise ApiError(401, "session_expired", "演示会话已失效。")
            if self.job_lock.locked():
                raise ApiError(409, "busy", "已有任务正在使用配置，请稍后重试。")
            storage = self.rows[token]["credentials"]
            if key:
                storage.write("tavily", key, {})
            else:
                storage.delete("tavily")

    def prune(self):
        with self.lock:
            # 请求进入后就计数，上传解析和 SSE 也不允许被清理。
            if not self.job_lock.acquire(blocking=False):
                return
            try:
                for token, row in list(self.rows.items()):
                    runtime = row["runtime"]
                    if row["active"] or self.clock() - row["seen"] <= self.ttl:
                        continue
                    if self.chroma_client is not None:
                        names = {c.name if hasattr(c,"name") else c for c in self.chroma_client.list_collections()}
                        if row["collection"] in names:
                            self.chroma_client.delete_collection(row["collection"])
                    target = runtime.upload_root.resolve()
                    if target.is_relative_to(self.root / "uploads") and target.exists():
                        shutil.rmtree(target)
                    row["credentials"].rows.clear()
                    runtime.quiz_store = None
                    runtime._store = None
                    del self.rows[token]
            finally:
                self.job_lock.release()


class CloudSessionGuard:
    """ASGI 保护所有云端接口，保持 SSE 逐帧流，不缓冲响应。"""
    def __init__(self, app, sessions):
        self.app, self.sessions = app, sessions

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        token = headers.get(b"x-v2-session-token", b"").decode("ascii", errors="ignore")
        acquired = False
        try:
            if scope.get("client", (None,))[0] not in {"127.0.0.1", "::1"}:
                raise ApiError(403, "internal_only", "仅允许内部服务请求。")
            runtime = self.sessions.acquire(token)
            acquired = True
            if (getattr(runtime, "cloud_sponsored", False)
                and scope.get("path") == "/settings/providers/dashscope"
                and scope.get("method") in {"PUT", "DELETE"}):
                raise ApiError(403, "host_managed", "百炼服务由演示者管理，访客不能修改。")
            scope.setdefault("state", {})["runtime"] = runtime
            if scope.get("path") == "/notes/upload":
                # 演示端单次请求最多 5 MiB，每个会话上传目录最多 20 MiB。
                limit = 5 * 1024 * 1024
                length = headers.get(b"content-length", b"0")
                if not length.isdigit() or int(length) > limit:
                    raise ApiError(413, "file_too_large", "云端演示每次上传最多 5 MiB。")
                used = sum(p.stat().st_size for p in runtime.upload_root.rglob("*") if p.is_file())
                if used + int(length) > 20 * 1024 * 1024:
                    raise ApiError(413, "storage_limit", "当前演示会话资料已达 20 MiB 上限。")
                total = 0
                original_receive = receive
                async def bounded_receive():
                    nonlocal total
                    message = await original_receive()
                    total += len(message.get("body", b""))
                    if total > limit:
                        raise ApiError(413, "file_too_large", "云端上传超限。")
                    return message
                receive = bounded_receive
            await self.app(scope, receive, send)
        except ApiError as error:
            await JSONResponse({"status":error.code,"data":None,"message":error.message},
                               status_code=error.status)(scope, receive, send)
        finally:
            if acquired:
                self.sessions.release(token)
