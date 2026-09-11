"""个人笔记复习助手 Streamlit 前端。"""

import json
import os
import time
from html import escape
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEEPSEEK_API_KEY_HEADER = "X-DeepSeek-API-Key"

st.set_page_config(page_title="笔记复习助手", page_icon="📚", layout="wide")

if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = ""
if "validated_api_key" not in st.session_state:
    st.session_state.validated_api_key = ""
if "chat_image_uploader_version" not in st.session_state:
    st.session_state.chat_image_uploader_version = 0
if "pending_note_delete" not in st.session_state:
    st.session_state.pending_note_delete = None
if "confirm_delete_all" not in st.session_state:
    st.session_state.confirm_delete_all = False
if "note_uploader_version" not in st.session_state:
    st.session_state.note_uploader_version = 0
if "note_import_in_progress" not in st.session_state:
    st.session_state.note_import_in_progress = False


def begin_note_import() -> None:
    """Lock the import action before the processing rerun starts."""
    st.session_state.note_import_in_progress = True

st.markdown(
    """
    <style>
    :root {
        --ink: #203047;
        --muted: #718096;
        --cream: #faf8f3;
        --paper: #fffdf9;
        --sage: #dce9df;
        --sage-strong: #4f7a63;
        --peach: #f5d9c5;
        --peach-strong: #c76d4a;
        --line: #e9e2d7;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 82% 10%, rgba(220,233,223,.75), transparent 25rem),
            radial-gradient(circle at 55% 88%, rgba(245,217,197,.48), transparent 28rem),
            var(--cream);
    }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
    .block-container { max-width: 1260px; padding-top: .65rem; padding-bottom: 1.5rem; }
    .app-brand {
        position: fixed; top: .72rem; left: max(1.4rem, calc(50vw - 620px)); z-index: 1000001;
        color: #184d38; font-size: 1.25rem; line-height: 2rem; font-weight: 800;
    }
    .app-brand span { color: var(--muted); font-size: .8rem; font-weight: 500; margin-left: .65rem; }
    [data-testid="stToolbar"] .rc-overflow {
        justify-content: center !important; padding-left: 250px !important; box-sizing: border-box;
    }
    .page-heading { margin: .45rem 0 .65rem; }
    .page-heading h1 { font-size: 1.75rem; margin: 0 0 .12rem; }
    .page-heading p { color: var(--muted); margin: 0; }
    [data-testid="stFileUploader"] {
        background: #ffffffb8;
        border: 1px dashed #9db9a6;
        border-radius: 16px;
        padding: .7rem .85rem;
    }
    [data-testid="stFileUploader"] section { padding: .2rem; }
    [data-testid="stFileUploaderDropzone"] { border: 0; background: transparent; }
    .stButton > button {
        background: var(--peach-strong);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
    }
    .stButton > button:disabled {
        background: #d8d9d3;
        color: #8b928c;
    }
    .stButton > button p { white-space: nowrap; }
    /* 问答区已选模式使用鼠尾草绿，和 UI 图一致；侧边栏导入按钮仍保持桃色。 */
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"] {
        background: var(--sage-strong);
        border-color: var(--sage-strong);
        color: #ffffff;
    }
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #426b55;
        border-color: #426b55;
    }
    :is(.st-key-mode_fast, .st-key-mode_accurate) [data-testid="stButton"] > button[kind="primary"] {
        background: #4f8066; color: #ffffff; border: 1px solid #4f8066;
        box-shadow: 0 3px 10px rgba(49, 93, 69, .16);
    }
    :is(.st-key-mode_fast, .st-key-mode_accurate) [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #426f58; color: #ffffff; border-color: #426f58;
    }
    :is(.st-key-mode_fast, .st-key-mode_accurate) [data-testid="stButton"] > button[kind="secondary"] {
        background: #f7f6f2; color: #4f5965; border: 1px solid #d7d4cf;
    }
    :is(.st-key-mode_fast, .st-key-mode_accurate) [data-testid="stButton"] > button[kind="secondary"]:hover {
        background: #eeece7; color: #315d45; border-color: #b8c9bc;
    }
    :is(.st-key-mode_fast, .st-key-mode_accurate) [data-testid="stButton"] > button { min-height: 2.45rem; }
    .st-key-chat_image_popover [data-testid="stPopover"] > button {
        min-height: 2.9rem; background: #fffdf9; color: #315d45;
        border: 1px solid #b8c9bc; border-radius: 14px;
    }
    [data-testid="stChatInput"] {
        background: #fffdf9;
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(51, 67, 54, .08);
    }
    [data-testid="stChatInput"] textarea { color: var(--ink); }
    .mode-card {
        min-height: 5.5rem; padding: .85rem 1rem; border: 1px solid var(--line);
        border-radius: 14px; background: rgba(255,253,249,.72); margin: .1rem 0 .8rem;
    }
    .mode-card b { color: var(--ink); display: block; font-size: .98rem; margin-bottom: .22rem; }
    .mode-card span { color: var(--muted); font-size: .82rem; line-height: 1.4; }
    .mode-flow { color: var(--sage-strong); font-size: .84rem; font-weight: 650; margin: -.25rem 0 .85rem; }
    .mode-history-divider {
        display: flex; align-items: center; gap: .65rem; margin: 1rem 0;
        color: #567265; font-size: .82rem; font-weight: 700;
    }
    .mode-history-divider::before, .mode-history-divider::after {
        content: ""; height: 1px; flex: 1; background: #bdd1c1;
    }
    .mode-history-divider span {
        padding: .32rem .7rem; border-radius: 999px;
        background: #edf5ee; border: 1px solid #bdd1c1;
    }
    .eyebrow {
        display: inline-flex; align-items: center; gap: .45rem;
        color: var(--sage-strong); font-size: .76rem; font-weight: 750;
        letter-spacing: .11em;
    }
    .eyebrow::before { content: ""; width: .55rem; height: .55rem; border-radius: 50%; background: #6b9a76; }
    .hero-card {
        background: rgba(255,253,249,.82); border: 1px solid var(--line);
        border-radius: 24px; padding: 2.3rem 2.4rem; margin-bottom: 1.3rem;
        box-shadow: 0 14px 36px rgba(62, 74, 63, .07);
    }
    .hero-card h1 { margin: .55rem 0 .65rem; font-size: 3.1rem; line-height: 1.08; }
    .hero-card h1 em { color: var(--sage-strong); font-style: normal; }
    .hero-card p { color: var(--muted); font-size: 1.02rem; margin-bottom: 0; }
    .scope-pill {
        display: inline-block; margin-top: 1.1rem; padding: .42rem .75rem;
        border-radius: 99px; background: var(--sage); color: #426550;
        font-size: .84rem; font-weight: 650;
    }
    .section-title { color: var(--ink); font-size: 1rem; font-weight: 750; margin: .2rem 0 .38rem; }
    .empty-card, .focus-card {
        background: rgba(255,253,249,.76); border: 1px solid var(--line);
        border-radius: 18px; padding: 1.25rem 1.35rem; margin: .7rem 0;
    }
    .empty-card strong, .focus-card strong { display: block; color: var(--ink); margin-bottom: .35rem; }
    .empty-card span, .focus-card span { color: var(--muted); font-size: .9rem; line-height: 1.55; }
    .mini-step {
        background: #fffdf9; border-left: 3px solid #8eb69a;
        padding: .72rem .8rem; margin: .65rem 0; border-radius: 0 10px 10px 0;
    }
    .mini-step b { color: var(--ink); font-size: .88rem; }
    .mini-step small { color: var(--muted); display: block; margin-top: .12rem; }
    .library-upload-intro { text-align:center; color:var(--muted); margin:-.2rem 0 .65rem; font-size:.9rem; }
    .st-key-library_upload_panel { background:rgba(255,253,249,.78); border:1px dashed #aabcae; border-radius:18px; padding:1.15rem 1.3rem .95rem; margin:.25rem 0 1.25rem; }
    .st-key-library_upload_panel [data-testid="stFileUploader"] { min-height:150px; display:flex; align-items:center; padding:1rem; }
    .st-key-library_upload_panel [data-testid="stFileUploaderDropzone"] { min-height:118px; justify-content:center; }
    .st-key-library_upload_panel [data-testid="stFileUploaderDropzoneInstructions"] { text-align:center; }
    .st-key-library_upload_panel [data-testid="stFileUploaderDropzone"] button { min-height:2.8rem; padding:0 1.35rem; font-weight:750; }
    .st-key-library_upload_panel [data-testid="stButton"] > button { min-height:3.1rem; font-size:.98rem; }
    .library-toolbar { display:flex; justify-content:space-between; align-items:end; margin:.2rem 0 .55rem; }
    .library-toolbar h3 { margin:0; font-size:1.02rem; }
    .library-toolbar span { color:var(--muted); font-size:.8rem; }
    .note-item { background:rgba(255,253,249,.9); border:1px solid var(--line); border-radius:13px; padding:.78rem .9rem; margin:.35rem 0; box-shadow:0 4px 14px rgba(62,74,63,.035); }
    .note-row { display:grid; grid-template-columns:minmax(0,2.4fr) .62fr .8fr 1.05fr; align-items:center; gap:.8rem; }
    .note-name { min-width:0; font-size:.9rem; font-weight:700; color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .note-type { justify-self:start; padding:.22rem .5rem; border-radius:6px; border:1px solid #bfd2c1; background:#edf5ee; color:#356148; font-size:.72rem; font-weight:750; }
    .note-chunks, .note-time { color:var(--muted); font-size:.78rem; white-space:nowrap; }
    .library-summary { text-align:center; color:var(--muted); font-size:.82rem; margin:.8rem 0 .2rem; }
    .source-label { color: var(--sage-strong); font-size: .82rem; font-weight: 700; }
    .privacy-note { color: var(--muted); font-size: .86rem; text-align: center; margin-top: .8rem; }
    .balance-strip {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        margin-top:.7rem; padding:.82rem 1rem; border:1px solid #cfe0d3; border-radius:12px;
        background:linear-gradient(135deg, #f3f8f2 0%, #edf5ef 100%);
    }
    .balance-label { display:flex; align-items:center; gap:.5rem; color:#52665a; font-size:.9rem; font-weight:650; }
    .balance-label::before { content:""; width:7px; height:7px; border-radius:50%; background:#4f876b; box-shadow:0 0 0 4px rgba(79,135,107,.12); }
    .balance-value { color:#184d38; font-size:1.15rem; line-height:1; font-weight:750; letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
    .about-card {
        max-width: 820px; margin: 1.4rem auto; background: rgba(255,253,249,.86);
        border: 1px solid var(--line); border-radius: 22px; padding: 2rem 2.2rem;
        box-shadow: 0 14px 36px rgba(62,74,63,.07);
    }
    .tech-chip { display:inline-block; padding:.38rem .65rem; margin:.2rem; border-radius:99px; background:#edf4ed; color:#315d45; font-size:.82rem; }
    .github-link { display:inline-flex; align-items:center; gap:.42rem; margin-top:.45rem; color:#205d43; font-weight:750; text-decoration:none; }
    .github-link:hover { color:#153f2f; text-decoration:underline; }
    @media (max-width: 900px) {
        .app-brand span { display: none; }
        [data-testid="stToolbar"] .rc-overflow { padding-left: 180px !important; }
    }
    @media (prefers-color-scheme: dark) {
        .st-key-chat_image_popover [data-testid="stPopover"] > button {
            background: #2b302d; color: #e6eee8; border-color: #56645a;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def get_notes() -> tuple[list[dict], str | None]:
    """从 FastAPI 获取已导入的笔记；短暂重试，避免后端刚启动时误判为空库。"""
    last_error = ""
    for attempt in range(3):
        try:
            # 本项目的前后端都在本机。关闭环境代理读取，避免 127.0.0.1 被错误转发。
            with httpx.Client(timeout=15, trust_env=False) as client:
                response = client.get(f"{API_BASE_URL}/api/notes")
                response.raise_for_status()
                return response.json(), None
        except (httpx.HTTPError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < 2:
                time.sleep(0.5)

    print(f"读取笔记列表失败：{last_error}")
    return [], "暂时无法读取笔记库，请稍后刷新页面。"


def delete_note_request(note_id: str | None = None) -> None:
    """调用后端删除接口；Key 只放在本次请求头。"""
    url = f"{API_BASE_URL}/api/notes"
    if note_id:
        url += f"/{quote(note_id, safe=':')}"
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.delete(
            url,
            headers={DEEPSEEK_API_KEY_HEADER: st.session_state.deepseek_api_key},
        )
        response.raise_for_status()
    get_notes.clear()


def render_sources(sources: list[dict]) -> None:
    """展示后端 SSE meta 事件返回的来源片段。"""
    if not sources:
        return

    with st.expander("参考笔记", expanded=False):
        for source in sources:
            header_path = " > ".join(source["header_path"]) or "未标注标题"
            st.markdown(
                f"<div class='source-label'>📎 {source['file_name']} · {header_path}</div>",
                unsafe_allow_html=True,
            )
            st.write(source["content_preview"])
            if source.get("image_path"):
                st.image(source["image_path"])
            st.divider()


def has_valid_api_key() -> bool:
    """Key 只有通过验证且未被再次编辑时才可用于业务请求。"""
    current_key = st.session_state.deepseek_api_key.strip()
    return bool(current_key) and st.session_state.validated_api_key == current_key


def clear_api_key() -> None:
    st.session_state.deepseek_api_key = ""
    st.session_state.validated_api_key = ""
    if "api_key_input" in st.session_state:
        st.session_state.api_key_input = ""
    st.session_state.pop("deepseek_balance", None)


def render_balance(balance: dict) -> None:
    currency = str(balance.get("currency", ""))
    symbol = {"CNY": "¥", "USD": "$"}.get(currency, f"{escape(currency)} ")
    amount = escape(str(balance.get("total_balance", "--")))
    st.markdown(
        f"<div class='balance-strip'><span class='balance-label'>当前余额</span>"
        f"<span class='balance-value'>{symbol}{amount}</span></div>",
        unsafe_allow_html=True,
    )


def render_settings_page() -> None:
    st.markdown(
        "<div class='page-heading'><h1>设置</h1><p>配置当前浏览器会话使用的模型访问凭据。</p></div>",
        unsafe_allow_html=True,
    )
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = st.session_state.deepseek_api_key
    _, settings_column, _ = st.columns([1.1, 1.5, 1.1])
    with settings_column:
        with st.container(border=True):
            st.subheader("DeepSeek API Key")
            st.text_input(
                "API Key",
                type="password",
                key="api_key_input",
                placeholder="请输入 DeepSeek API Key",
                help="仅保存在当前浏览器会话，并随单次请求发送。",
            )
            save_column, clear_column = st.columns(2)
            if save_column.button("保存并验证", type="primary", use_container_width=True):
                candidate_key = st.session_state.api_key_input.strip()
                if not candidate_key:
                    st.session_state.validated_api_key = ""
                    st.warning("请先输入 API Key")
                else:
                    try:
                        response = httpx.post(
                            f"{API_BASE_URL}/api/key/validate",
                            headers={DEEPSEEK_API_KEY_HEADER: candidate_key},
                            timeout=30,
                        )
                        response.raise_for_status()
                        result = response.json()
                        if result.get("valid"):
                            st.session_state.deepseek_api_key = candidate_key
                            st.session_state.validated_api_key = candidate_key
                            st.session_state.pop("deepseek_balance", None)
                        else:
                            st.session_state.validated_api_key = ""
                            st.error(result.get("message", "API Key 验证失败"))
                    except (httpx.HTTPError, ValueError):
                        st.session_state.validated_api_key = ""
                        st.error("无法连接 DeepSeek，请稍后重试")
            clear_column.button("清除 Key", use_container_width=True, on_click=clear_api_key)
            if has_valid_api_key():
                st.success("已保存，当前会话内有效")
                try:
                    response = httpx.get(
                        f"{API_BASE_URL}/api/key/balance",
                        headers={DEEPSEEK_API_KEY_HEADER: st.session_state.deepseek_api_key},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result.get("success") is False:
                        st.error(result.get("message", "余额查询失败，请稍后重试"))
                    else:
                        st.session_state.deepseek_balance = result
                except (httpx.HTTPError, ValueError):
                    st.error("余额查询失败，请稍后重试")
                balance = st.session_state.get("deepseek_balance")
                if balance:
                    render_balance(balance)
        st.markdown(
            "<div class='privacy-note'>🔒 Key 仅保存在当前浏览器会话中，不会写入数据库或日志。刷新或关闭会话后可能清空。</div>",
            unsafe_allow_html=True,
        )


def render_about_page() -> None:
    st.markdown(
        """
        <div class='page-heading'><h1>关于</h1><p>了解这个项目解决什么问题，以及它是如何构建的。</p></div>
        <div class='about-card'>
            <div class='eyebrow'>PERSONAL KNOWLEDGE SPACE</div>
            <h2>笔记复习助手</h2>
            <p>基于 RAG 与视觉理解的个人笔记复习工具，帮助你从自己的资料中提问、回顾并追溯答案来源。</p>
            <hr>
            <h4>技术栈</h4>
            <div>
                <span class='tech-chip'>FastAPI</span><span class='tech-chip'>Streamlit</span>
                <span class='tech-chip'>LangChain</span><span class='tech-chip'>Chroma</span>
                <span class='tech-chip'>BM25</span><span class='tech-chip'>DeepSeek Vision</span>
            </div>
            <hr>
            <p><strong>开源地址</strong><br>
            <a class='github-link' href='https://github.com/yuyuyuoo55/Personal_Notes_Review_Assistant' target='_blank'>
                <svg width='17' height='17' viewBox='0 0 16 16' fill='currentColor' aria-hidden='true'><path d='M8 0C3.58 0 0 3.64 0 8.13c0 3.59 2.29 6.64 5.47 7.71.4.08.55-.18.55-.39 0-.19-.01-.83-.01-1.51-2.01.38-2.53-.5-2.69-.96-.09-.24-.48-.96-.82-1.15-.28-.15-.68-.53-.01-.54.63-.01 1.08.59 1.23.83.72 1.23 1.87.88 2.33.67.07-.53.28-.88.51-1.08-1.78-.21-3.64-.91-3.64-4.02 0-.89.31-1.62.82-2.19-.08-.21-.36-1.04.08-2.16 0 0 .67-.22 2.2.84A7.5 7.5 0 0 1 8 3.89c.68 0 1.36.09 2 .27 1.53-1.06 2.2-.84 2.2-.84.44 1.12.16 1.95.08 2.16.51.57.82 1.3.82 2.19 0 3.12-1.87 3.81-3.65 4.02.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.47.55.39A8.04 8.04 0 0 0 16 8.13C16 3.64 12.42 0 8 0Z'/></svg>
                GitHub ↗
            </a></p>
            <small>感谢每一位使用并提出反馈的朋友。</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 导入成功后递增 key，使 Streamlit 重建上传控件并清空刚才选中的文件。
# 否则页面 rerun 后仍保留该文件，会马上被“重复导入”校验命中，容易造成误解。
def render_library_page() -> None:
    has_api_key = has_valid_api_key()
    notes, notes_load_error = get_notes()
    existing_note_names = {note["file_name"] for note in notes}
    st.markdown(
        "<div class='page-heading'><h1>知识库</h1><p>集中导入、查看和管理用于检索的学习资料。</p></div>",
        unsafe_allow_html=True,
    )
    if not has_api_key:
        st.warning("请先在「设置」页填写并验证 DeepSeek API Key。")
    st.markdown("<div class='library-upload-intro'>支持 Markdown / ZIP / JPG / PNG / WEBP。文档含图片时，请将图片放入同级 images 文件夹，与 Markdown 一起压缩为 ZIP 后上传。</div>", unsafe_allow_html=True)
    if "note_import_success" in st.session_state:
        st.success(st.session_state.pop("note_import_success"))
    if "note_import_error" in st.session_state:
        st.error(st.session_state.pop("note_import_error"))

    with st.container(key="library_upload_panel"):
        uploaded_file = st.file_uploader(
            "选择 Markdown、ZIP 或图片文件", type=["md", "zip", "jpg", "jpeg", "png", "webp"],
            disabled=not has_api_key, label_visibility="collapsed",
            key=f"note_uploader_{st.session_state.note_uploader_version}",
        )
        is_duplicate_file = bool(
            uploaded_file and uploaded_file.name in existing_note_names
        )
        if is_duplicate_file:
            st.info(f"{uploaded_file.name} 已在笔记库中，无需重复导入。")
        is_zip_file = bool(uploaded_file and uploaded_file.name.lower().endswith(".zip"))
        if is_zip_file:
            st.info("ZIP 中的图片需要逐张识别，导入可能需要几分钟。开始后请勿刷新页面或重复点击。")
        st.button(
            "正在导入…" if st.session_state.note_import_in_progress else "导入文件",
            use_container_width=True,
            disabled=(
                not has_api_key
                or uploaded_file is None
                or is_duplicate_file
                or st.session_state.note_import_in_progress
            ),
            on_click=begin_note_import,
        )
    if st.session_state.note_import_in_progress:
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            import_message = (
                "正在上传 ZIP、逐张识别图片并建立索引，请勿刷新页面…"
                if is_zip_file
                else "正在导入文档、识别图片并建立索引，请勿关闭页面…"
            )
            with st.spinner(
                import_message,
                show_time=True,
            ):
                response = httpx.post(
                    f"{API_BASE_URL}/api/notes/import",
                    files=files,
                    headers={DEEPSEEK_API_KEY_HEADER: st.session_state.deepseek_api_key},
                    timeout=120,
                )
            response.raise_for_status()
            result = response.json()
            st.session_state.note_import_success = (
                f"已导入 {result['file_name']} · {result['chunk_count']} 个片段"
            )
            if result.get("image_processed"):
                st.session_state.note_import_success += (
                    f"；{result['image_processed']} 张图片已识别"
                )
            if result.get("warnings"):
                st.session_state.note_import_success += (
                    f"；{result.get('image_processed', 0)} 张图片已识别，"
                    f"{result.get('image_skipped', 0)} 张已跳过"
                )
            get_notes.clear()
            st.session_state.note_uploader_version += 1
            st.session_state.note_import_in_progress = False
            st.rerun()
        except httpx.HTTPStatusError as error:
            # 后端可能返回 JSON 业务错误，也可能在异常时返回空响应或 HTML。
            # 前端不能再直接 .json()，否则会把 JSONDecodeError 暴露给用户。
            try:
                detail = error.response.json().get("detail", "笔记导入失败")
            except ValueError:
                detail = f"笔记导入失败（后端状态码：{error.response.status_code}）"
            st.session_state.note_import_error = detail
            st.session_state.note_import_in_progress = False
            st.rerun()
        except httpx.HTTPError:
            st.session_state.note_import_error = "无法连接后端，请先启动项目"
            st.session_state.note_import_in_progress = False
            st.rerun()

    st.markdown(f"<div class='library-toolbar'><h3>已导入文件</h3><span>{len(notes)} 个文件</span></div>", unsafe_allow_html=True)

    if notes_load_error:
        st.warning(notes_load_error)
    elif not notes:
        st.caption("还没有导入笔记")
    else:
        if st.session_state.confirm_delete_all:
            st.warning("确认清空全部笔记？此操作无法撤销。")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("确认清空", type="primary", use_container_width=True):
                try:
                    delete_note_request()
                    st.session_state.confirm_delete_all = False
                    st.session_state.pending_note_delete = None
                    st.rerun()
                except httpx.HTTPError:
                    st.error("清空失败，请确认后端服务正常")
            if cancel_col.button("取消", use_container_width=True):
                st.session_state.confirm_delete_all = False
                st.rerun()
        elif st.button("🗑 清空全部", use_container_width=True, disabled=not has_api_key):
            st.session_state.confirm_delete_all = True
            st.rerun()

        for note in notes:
            note_col, delete_col = st.columns([6.2, 1.05], gap="small", vertical_alignment="center")
            file_type = Path(note["file_name"]).suffix.lstrip(".").upper() or "FILE"
            note_col.markdown(
                "<div class='note-item'><div class='note-row'>"
                f"<div class='note-name'>{escape(note['file_name'])}</div>"
                f"<span class='note-type'>{escape(file_type)}</span>"
                f"<span class='note-chunks'>{note['chunk_count']} 个片段</span>"
                f"<span class='note-time'>{escape(note['imported_at'])}</span></div></div>",
                unsafe_allow_html=True,
            )
            if delete_col.button(
                "删除",
                key=f"delete_{note['note_id']}",
                disabled=not has_api_key,
                use_container_width=True,
            ):
                st.session_state.pending_note_delete = note["note_id"]
                st.rerun()
            if st.session_state.pending_note_delete == note["note_id"]:
                st.warning(f"确认删除「{note['file_name']}」？")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "确认删除",
                    key=f"confirm_{note['note_id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        delete_note_request(note["note_id"])
                        st.session_state.pending_note_delete = None
                        st.rerun()
                    except httpx.HTTPError:
                        st.error("删除失败，请确认笔记仍存在且后端服务正常")
                if cancel_col.button(
                    "取消",
                    key=f"cancel_{note['note_id']}",
                    use_container_width=True,
                ):
                    st.session_state.pending_note_delete = None
                    st.rerun()

        st.markdown(
            f"<div class='library-summary'>共 {len(notes)} 个文件 · {sum(note['chunk_count'] for note in notes)} 个片段</div>",
            unsafe_allow_html=True,
        )


current_page = "智能问答"


def select_page(page_name: str):
    def activate_page() -> None:
        global current_page
        current_page = page_name

    return activate_page


st.markdown(
    "<div class='app-brand'>笔记复习助手 <span>会话内安全连接</span></div>",
    unsafe_allow_html=True,
)
navigation = st.navigation(
    [
        st.Page(select_page("智能问答"), title="智能问答", url_path="chat", default=True),
        st.Page(select_page("知识库"), title="知识库", url_path="library"),
        st.Page(select_page("设置"), title="设置", url_path="settings"),
        st.Page(select_page("关于"), title="关于", url_path="about"),
    ],
    position="top",
)
navigation.run()

if current_page == "知识库":
    render_library_page()
    st.stop()
if current_page == "设置":
    render_settings_page()
    st.stop()
if current_page == "关于":
    render_about_page()
    st.stop()

has_api_key = has_valid_api_key()
notes, notes_load_error = get_notes()
if not has_api_key:
    st.warning("请先在顶部导航的「设置」页填写并验证 DeepSeek API Key，完成后即可开始提问。")

note_count = len(notes)
chunk_count = sum(note["chunk_count"] for note in notes)

st.markdown(
    f"<div class='page-heading'><h1>智能问答</h1><p>当前检索范围 · {note_count} 份笔记 · {chunk_count} 个片段</p></div>",
    unsafe_allow_html=True,
)

chat_column, focus_column = st.columns([2.1, 1], gap="large")

with chat_column:
    st.markdown("<div class='section-title'>围绕笔记提问</div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = "fast"
    if "conversation_id" not in st.session_state:
        # 仅作为本次浏览器会话的内存键；刷新并新建会话或重启后端都会清空记忆。
        st.session_state.conversation_id = uuid4().hex

    # 模式选择位于问答区顶部；每次提问都把当前模式一起发送给 FastAPI。
    fast_column, accurate_column = st.columns(2, gap="small")
    with fast_column:
        if st.button(
            "快速模式",
            key="mode_fast",
            type="primary" if st.session_state.retrieval_mode == "fast" else "secondary",
            use_container_width=True,
        ):
            if st.session_state.retrieval_mode != "fast":
                # 不拆分聊天记录；仅在真正切换时插入一条模式分隔线。
                if st.session_state.messages:
                    st.session_state.messages.append(
                        {"role": "mode", "content": "已切换到：快速模式（Agentic RAG）"}
                    )
                st.session_state.retrieval_mode = "fast"
            st.rerun()
    with accurate_column:
        if st.button(
            "精确查找",
            key="mode_accurate",
            type="primary" if st.session_state.retrieval_mode == "accurate" else "secondary",
            use_container_width=True,
        ):
            if st.session_state.retrieval_mode != "accurate":
                # 精确查找与快速模式共用历史，但历史中会保留清晰的模式边界。
                if st.session_state.messages:
                    st.session_state.messages.append(
                        {"role": "mode", "content": "已切换到：精确查找（Step RAG）"}
                    )
                st.session_state.retrieval_mode = "accurate"
            st.rerun()

    mode_descriptions = {
        "fast": "当前链路：Agent 判断 →（直接回答 / 向量检索 Top-3）→ 基于片段回答",
        "accurate": "当前链路：原问题 → 查询改写 → 向量 + BM25 → RRF → Cross-Encoder → 回答",
    }
    st.markdown(
        f"<div class='mode-flow'>{mode_descriptions[st.session_state.retrieval_mode]}</div>",
        unsafe_allow_html=True,
    )

    # 只有问答历史固定在这个独立滚动区域中；侧边栏和右侧复习卡不会跟着滚动。
    chat_history = st.container(height=420, border=True)

    with chat_history:
        if not notes:
            st.markdown(
                """
                <div class="empty-card">
                    <strong>先导入第一份 Markdown 笔记</strong>
                    <span>导入后可以询问概念、术语或跨章节问题；回答会附带对应的原文来源。</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message in st.session_state.messages:
            if message["role"] == "mode":
                st.markdown(
                    f"<div class='mode-history-divider'><span>{message['content']}</span></div>",
                    unsafe_allow_html=True,
                )
                continue

            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))
                    if "elapsed_ms" in message:
                        st.caption(f"本次回答耗时：{message['elapsed_ms'] / 1000:.2f} 秒")

    input_column, image_column = st.columns([6, 1.15], gap="small", vertical_alignment="bottom")
    with input_column:
        question = st.chat_input(
            "例如：RRF 和加权融合有什么区别？",
            disabled=not has_api_key,
        )
    with image_column:
        with st.container(key="chat_image_popover"):
            with st.popover("添加图片", use_container_width=True):
                uploaded_chat_image = st.file_uploader(
                    "选择参与本次问答的图片",
                    type=["jpg", "jpeg", "png", "gif", "webp"],
                    disabled=not has_api_key,
                    label_visibility="collapsed",
                    key=f"chat_image_{st.session_state.chat_image_uploader_version}",
                )
                if uploaded_chat_image:
                    st.caption(uploaded_chat_image.name)

    if question:
        if not uploaded_chat_image and not notes:
            st.warning("请先导入 Markdown 笔记，或在对话区上传一张图片。")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": question})

        with chat_history:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                answer = ""
                sources = []
                elapsed_ms = 0
                reranker_progress = None

                try:
                    # 请求刚发出就显示状态；模型生成第一个 token 前不会再像页面卡住。
                    with st.status("正在理解问题并检索笔记…", expanded=True) as request_status:
                        headers = {
                            DEEPSEEK_API_KEY_HEADER: st.session_state.deepseek_api_key
                        }
                        if uploaded_chat_image:
                            endpoint = f"{API_BASE_URL}/api/chat/image"
                            request_kwargs = {
                                "data": {
                                    "query": question,
                                    "mode": st.session_state.retrieval_mode,
                                    "conversation_id": st.session_state.conversation_id,
                                },
                                "files": {
                                    "image": (
                                        uploaded_chat_image.name,
                                        uploaded_chat_image.getvalue(),
                                        uploaded_chat_image.type,
                                    )
                                },
                            }
                            request_status.update(
                                label="正在使用 DeepSeek Vision 理解图片…",
                                state="running",
                            )
                        else:
                            endpoint = f"{API_BASE_URL}/api/chat"
                            request_kwargs = {
                                "json": {
                                    "query": question,
                                    "mode": st.session_state.retrieval_mode,
                                    "conversation_id": st.session_state.conversation_id,
                                }
                            }
                        with httpx.stream(
                            "POST",
                            endpoint,
                            headers=headers,
                            **request_kwargs,
                            timeout=180,
                        ) as response:
                            response.raise_for_status()
                            event_name = ""

                            for line in response.iter_lines():
                                if line.startswith("event:"):
                                    event_name = line.removeprefix("event:").strip()
                                elif line.startswith("data:"):
                                    payload = json.loads(line.removeprefix("data:").strip())

                                    if event_name == "meta":
                                        sources = payload.get("sources", [])
                                        status_label = (
                                            "已找到相关资料，正在生成回答…"
                                            if sources
                                            else "没有找到对应片段，正在整理回复…"
                                        )
                                        request_status.update(label=status_label, state="running")
                                    elif event_name == "stage":
                                        if reranker_progress is None:
                                            reranker_progress = st.progress(0)
                                        reranker_progress.progress(
                                            int(payload.get("progress", 0)),
                                            text=payload.get("label", "正在准备精排模型…"),
                                        )
                                    elif event_name == "token":
                                        answer += payload["content"]
                                        answer_placeholder.markdown(f"{answer}▌")
                                    elif event_name == "done":
                                        elapsed_ms = int(payload.get("elapsed_ms", 0))

                        request_status.update(label="回答完成", state="complete", expanded=False)

                    if reranker_progress is not None:
                        reranker_progress.empty()

                    answer_placeholder.markdown(answer)
                    render_sources(sources)
                    st.caption(f"本次回答耗时：{elapsed_ms / 1000:.2f} 秒")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
                    if uploaded_chat_image:
                        st.session_state.chat_image_uploader_version += 1
                    # 本次消息完成后重新运行，所有历史消息会回到滚动区，输入框仍在最下方。
                    st.rerun()
                except (httpx.HTTPError, RuntimeError):
                    answer_placeholder.warning("本次问答暂时无法完成，请稍后重试。")

with focus_column:
    mode_now = "快速模式（Agentic RAG）" if st.session_state.get("retrieval_mode", "fast") == "fast" else "精确查找（Step RAG）"
    mode_summary = (
        "Agent 自主判断是否检索；需要资料时调用向量检索，适合日常复习。"
        if st.session_state.get("retrieval_mode", "fast") == "fast"
        else "执行查询改写、双路召回、RRF 与精排，适合准确查找。"
    )
    st.markdown(
        f"<div class='focus-card'><strong>本次复习 · {mode_now}</strong><span>{mode_summary}</span></div>",
        unsafe_allow_html=True,
    )
