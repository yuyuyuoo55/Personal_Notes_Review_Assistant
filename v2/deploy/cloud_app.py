"""Streamlit Community Cloud 专用入口；不读取本机或平台的供应商密钥。"""
import runpy
import sys
from pathlib import Path

import streamlit as st

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_backend import start_backend
from app.api.runtime import ApiError


@st.cache_resource(show_spinner=False)
def backend():
    return start_backend()


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

with st.expander("云端联网配置（可选）"):
    st.caption("Tavily Key 只保存在当前服务端会话内存；保存不发请求，开启联网提问时才调用 Tavily MCP，可能收费。")
    if st.session_state.pop("clear_cloud_tavily", False):
        st.session_state["cloud_tavily_key"] = ""
    with st.form("cloud_tavily"):
        key = st.text_input("Tavily API Key", type="password", key="cloud_tavily_key", max_chars=512)
        consent = st.checkbox("同意将此 Key 交给本演示服务，并在启用联网时用于 Tavily")
        save = st.form_submit_button("保存或清除联网配置")
    if save:
        if not consent:
            st.warning("请确认密钥用途；不填写 Key 可清除配置。")
        else:
            try:
                sessions.set_tavily(st.session_state["_cloud_connection"]["token"], key.strip())
                st.session_state["clear_cloud_tavily"] = True
                st.rerun()
            except ApiError as error:
                st.warning(error.message)
