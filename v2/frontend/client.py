"""Streamlit 服务端 HTTP 客户端及 SSE 协议解析。"""
import json
import os
from urllib.parse import urlsplit

import httpx


class BackendError(Exception):
    def __init__(self, message, *, status="failed", data=None):
        super().__init__(message)
        self.status, self.data = status, data


def parse_sse(lines):
    """【业务需求】处理真正 SSE 帧，不把不完整流当成成功回答。
    【输入】HTTP iter_lines 文本流。
    【执行步骤】按空行分帧、忽略注释、拼接 data 行、校验 JSON 对象和事件类型。
    【输出】事件名与数据；格式异常明确失败。
    """
    name, data = "message", []
    for line in lines:
        if not line:
            if data:
                try:
                    value = json.loads("\n".join(data))
                except ValueError:
                    raise BackendError("回答流格式异常，本轮草稿未保存。") from None
                if name not in {"status", "token", "sources", "done", "error"} or not isinstance(value, dict):
                    raise BackendError("回答流事件无效，本轮草稿未保存。")
                yield name, value
            name, data = "message", []
        elif line.startswith("event:"):
            name = line[6:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        raise BackendError("回答流提前结束，本轮草稿未保存。")


class ApiClient:
    def __init__(self, base_url=None, settings_token=None):
        # 云端连接仅存在当前 Streamlit 服务端会话，不写进进程环境。
        connection = {}
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is not None:
            import streamlit as st
            connection = st.session_state.get("_cloud_connection", {})
        self.session_token = connection.get("token", "")
        self.base_url = base_url or connection.get("url") or os.getenv("V2_API_URL", "http://127.0.0.1:8000")
        parsed = urlsplit(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username:
            raise ValueError("本版本仅连接本机 HTTP 服务。")
        self.settings_token = (connection.get("token") or os.getenv("V2_SETTINGS_TOKEN", "")) if settings_token is None else settings_token

    def request(self, method, path, **kwargs):
        headers = {"X-V2-Session-Token":self.session_token} if self.session_token else {}
        if path.startswith("/settings/"):
            headers["X-V2-Settings-Token"] = self.settings_token
        try:
            with httpx.Client(base_url=self.base_url, timeout=210, trust_env=False, follow_redirects=False) as client:
                reply = client.request(method, path, headers=headers, **kwargs)
            body = reply.json()
            if not isinstance(body,dict) or "status" not in body:
                raise ValueError("无效响应")
        except (httpx.HTTPError, ValueError):
            raise BackendError("无法连接后端或响应异常，请检查 FastAPI 是否启动；未自动重试。") from None
        if reply.is_error:
            data = body.get("data")
            raise BackendError(body.get("message") or (data or {}).get("message") or "请求未完成，请检查输入或服务配置。",
                               status=body["status"], data=data)
        return body

    def stream(self, payload):
        """【业务需求】接收后端真实增量，必须收到合法 done 才结束成功。
        【输入】问题、历史及联网开关。
        【执行步骤】读取 SSE；校验终止事件；错误/断流抛异常，不自动重复付费调用。
        【输出】逐帧事件，由页面暂存 token；成功 done 后才提交历史。
        """
        try:
            with httpx.Client(base_url=self.base_url, timeout=210, trust_env=False) as client:
                with client.stream("POST", "/chat/stream", json=payload,
                                   headers={"X-V2-Session-Token":self.session_token} if self.session_token else {}) as reply:
                    if reply.status_code!=200 or "text/event-stream" not in reply.headers.get("content-type",""):
                        raise BackendError("问答请求未建立，请检查服务。")
                    for name, data in parse_sse(reply.iter_lines()):
                        if name=="token" and not isinstance(data.get("text"),str):
                            raise BackendError("回答增量格式无效，本轮草稿未保存。")
                        if name=="sources" and (not isinstance(data.get("items"),list) or
                                                  not all(isinstance(item,dict) for item in data["items"])):
                            raise BackendError("回答来源格式无效，本轮草稿未保存。")
                        if name=="error":
                            raise BackendError(data.get("message","回答未完成，本轮草稿未保存。"))
                        if name=="done":
                            if data.get("status")!="completed" or not isinstance(data.get("answer"),str) or not data["answer"]:
                                raise BackendError("回答完成标记无效，本轮草稿未保存。")
                            yield name,data
                            return
                        yield name,data
            raise BackendError("回答流中断，未收到完成标记；本轮草稿未保存。")
        except httpx.HTTPError:
            raise BackendError("问答连接中断，本轮草稿未保存；未自动重试。") from None
