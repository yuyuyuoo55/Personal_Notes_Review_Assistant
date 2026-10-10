"""正式 Streamlit 入口：所有业务通过本机 HTTP API，不直接调用业务函数。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from client import ApiClient, BackendError
from pages import chat_page, knowledge_page, quiz_page, report_page, settings_page

st.set_page_config(page_title="笔记复习助手 V2", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>" + Path(__file__).with_name("style.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
pages = {"chat":st.Page(chat_page,title="智能问答",url_path="chat",default=True),
         "knowledge":st.Page(knowledge_page,title="知识库",url_path="knowledge"),
         "quiz":st.Page(quiz_page,title="章节小测",url_path="quiz"),
         "report":st.Page(report_page,title="学习报告",url_path="report"),
         "settings":st.Page(settings_page,title="设置",url_path="settings")}
st.session_state["pages"] = pages
navigation = st.navigation(list(pages.values()), position="hidden")
with st.container(key="top_navigation"):
    columns = st.columns([3,1.2,1.2,1.2,1,2])
columns[0].markdown('<div class="brand">🌿 笔记复习助手 <small>V2</small></div>',unsafe_allow_html=True)
for column, name in zip(columns[1:5],("chat","knowledge","quiz","settings")):
    column.page_link(pages[name])
try:
    health = ApiClient().request("GET","/health")["data"]
    previous = st.session_state.get("backend_instance")
    if previous and previous!=health["instance_id"]:
        for key in ("active_quiz","completion","quiz_index","submitted_answers","report_id","quiz_locked"):
            st.session_state.pop(key,None)
        st.session_state["restart_notice"] = True
    st.session_state["backend_instance"] = health["instance_id"]
    st.session_state["health"] = health
    columns[5].caption("● 离线联调服务" if health.get("test_mode") else "● 云端会话已连接" if health.get("cloud_demo") else "● 本机服务已连接")
    if health.get("cloud_demo"):
        st.info("V2 云端临时演示：各浏览器会话独立，使用你自己的 Key；刷新、休眠或重启可能清空资料与小测。闲置 30 分钟后清理。请勿上传敏感资料。")
    if health.get("test_mode"):
        st.caption("离线联调 · Fake 模型 / Mock HTTP / 临时 Chroma；不代表真实模型质量验收。")
except BackendError:
    columns[5].caption("○ 后端未连接")
    st.warning("无法连接后端，请检查 FastAPI；本页不会自动重试模型请求。")
if st.session_state.pop("restart_notice",False):
    st.warning("后端已重启：进程内小测与报告已失效，请重新生成。当前聊天记录仅保留在本次会话。")
navigation.run()
