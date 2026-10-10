"""Streamlit Cloud 专用入口；百炼与 Tavily 来自平台 Secrets，DeepSeek 由访客配置。"""
import runpy
import sys
from pathlib import Path

import streamlit as st

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_backend import start_backend
from app.api.runtime import ApiError
from app.api.cloud import CloudSessions


@st.cache_resource(show_spinner=False)
def backend():
    # 只从平台 Secrets 明确读取共享服务；不加载本机 .env 或 DeepSeek。
    try:
        defaults = {name:str(st.secrets.get(name, "")).strip()
                    for name in ("DASHSCOPE_API_KEY", "TAVILY_API_KEY")}
    except FileNotFoundError:
        defaults = {"DASHSCOPE_API_KEY":"", "TAVILY_API_KEY":""}
    return start_backend(CloudSessions(sponsor_keys=defaults))


try:
    sessions, url = backend()
    if not sessions.server_thread.is_alive():
        backend.clear()
        sessions, url = backend()
    connection = st.session_state.get("_cloud_connection", {})
    if connection.get("url") != url or not sessions.valid(connection.get("token", "")):
        # 过期后的旧输入与聊天记录都不带入新会话。
        st.session_state.clear()
        st.session_state["_cloud_connection"] = {"url":url, "token":sessions.register()}
except (ApiError, RuntimeError):
    st.error("云端演示暂时无法建立会话，请稍后刷新；未调用模型。")
    st.stop()

# 复用现有页面布局、HTTP/SSE 客户端及所有真实业务。
runpy.run_path(str(V2 / "frontend" / "app.py"), run_name="__main__")


st.caption("百炼 Embedding 与 Tavily 联网由演示者提供；额度或服务异常会明确提示。")
