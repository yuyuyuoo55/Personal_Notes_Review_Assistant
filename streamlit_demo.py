# -*- coding: utf-8 -*-
"""个人笔记复习助手 · 单进程 Streamlit 演示版（可直接部署到 Streamlit Community Cloud）。

为什么有这个文件：
  原项目是「Streamlit 前端 + FastAPI 后端」两个进程，走 HTTP 通信。
  免费托管平台（Streamlit Community Cloud）只能跑一个 Streamlit 进程，
  因此这里把后端 RAG 逻辑用 import 直接复用（不重写），改成单进程直调，
  UI 与原版保持一致，方便当作在线 Demo 展示。

用法（本地验证）：
  uv run streamlit run streamlit_demo.py
  访问 http://127.0.0.1:8501

复用说明：
  所有检索/生成逻辑来自 backend.app.services.rag_service / note_splitter /
  storage.vector_store 等，本项目未改动这些模块，只替换了「前端→后端」的调用方式。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path
from time import perf_counter
from uuid import uuid4

# 让脚本能从项目根目录 import backend.app.* 模块（config.py 也会加载根目录 .env）。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# ---------------------------------------------------------------------------
# 部署者密钥注入：仅加载 DashScope/OSS 等后端配置。
# 用户 DeepSeek Key 必须在页面自行输入，不从 Streamlit Secrets 代填。
# ---------------------------------------------------------------------------
try:
    _secrets = st.secrets
    if _secrets:
        for _key in (
            "DASHSCOPE_API_KEY",
            "OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET",
            "OSS_ENDPOINT",
            "OSS_BUCKET_NAME",
            "OSS_PUBLIC_BASE_URL",
        ):
            _val = _secrets.get(_key)
            if _val and not os.environ.get(_key):
                os.environ[_key] = str(_val)
except Exception:  # st.secrets 在无 Secrets 文件时抛异常，忽略即可（本地用 .env）
    pass

# 后端模块（复用，不重写）。
from backend.app.core.config import UPLOAD_DIRECTORY  # noqa: E402
from backend.app.services.markdown_image_service import enrich_markdown_images  # noqa: E402
from backend.app.services.multimodal_service import (  # noqa: E402
    ImageProcessingError,
    build_image_chunk,
    describe_image_url,
    image_data_url,
    validate_image,
)
# validate_deepseek_api_key 用于"验证 Key"按钮；若云端该版本暂缺此函数，
# 降级为"验证 Key"按钮不可用，但 app 本体仍能正常启动和展示，不整体崩溃。
try:
    from backend.app.services.multimodal_service import validate_deepseek_api_key  # noqa: E402
except ImportError:  # pragma: no cover
    validate_deepseek_api_key = None
from backend.app.services.image_chunk_store import (  # noqa: E402
    load_standalone_image_chunks,
    manifest_path,
    remove_image_doc_dir,
    save_image_chunks,
    standalone_image_names,
)
from backend.app.services.note_loader import load_notes  # noqa: E402
from backend.app.services.note_splitter import split_documents  # noqa: E402
from backend.app.services.note_delete_service import (  # noqa: E402
    delete_all_notes,
    delete_note,
)
from backend.app.services.rag_service import (  # noqa: E402
    get_reranker_model,
    invalidate_rag_cache,
    is_reranker_cached,
    prepare_rag_answer,
)
from backend.app.services.reranker import _HAS_SENTENCE_TRANSFORMERS  # noqa: E402
from backend.app.storage.vector_store import knowledge_to_vector, vector_store  # noqa: E402


# ---------------------------------------------------------------------------
# 数据层（替代原 FastAPI /api/notes，直接在本进程操作）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def list_notes() -> list[dict]:
    """返回已导入笔记及其已写入 Chroma 的片段数（等价后端 list_notes）。"""
    if not UPLOAD_DIRECTORY.exists():
        return []
    notes: list[dict] = []
    for file_path in UPLOAD_DIRECTORY.glob("*.md"):
        stored = vector_store.get(where={"source": str(file_path)})
        notes.append(
            {
                "note_id": f"md:{file_path.name}",
                "file_name": file_path.name,
                "chunk_count": len(stored["ids"]),
                "kind": "md",
                "source": str(file_path),
                "doc_id": None,
            }
        )
    seen_doc_ids: set[str] = set()
    for chunk in load_standalone_image_chunks(UPLOAD_DIRECTORY):
        source = Path(str(chunk.metadata["source"]))
        doc_id = str(chunk.metadata.get("doc_id", source.stem))
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        notes.append(
            {
                "note_id": f"image:{doc_id}",
                "file_name": source.name,
                "chunk_count": 1,
                "kind": "image",
                "source": str(source),
                "doc_id": doc_id,
            }
        )
    return notes


def import_note(
    file_name: str,
    file_content: bytes,
    api_key: str,
    content_type: str | None = None,
) -> dict:
    """保存笔记、切分、写入 Chroma，并让 RAG 缓存失效（等价后端 import_note）。"""
    suffix = Path(file_name).suffix.lower()
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    if suffix not in {".md", *image_suffixes}:
        raise ValueError("当前只支持 .md、.jpg、.jpeg、.png 或 .webp 格式笔记")
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIRECTORY / file_name
    if file_path.exists() or file_name in standalone_image_names(UPLOAD_DIRECTORY):
        raise ValueError("同名笔记已存在；当前版本不重复导入")
    if not file_content:
        raise ValueError("上传文件不能为空")
    if suffix in image_suffixes:
        safe_stem = re.sub(r"[^0-9A-Za-z_-]+", "-", Path(file_name).stem).strip("-") or "image"
        doc_id = f"{safe_stem}-{sha256(file_name.encode('utf-8')).hexdigest()[:8]}"
        doc_dir = UPLOAD_DIRECTORY / doc_id
        try:
            image_chunk = asyncio.run(build_image_chunk(
                file_content,
                content_type or guess_type(file_name)[0] or "",
                doc_dir,
                file_path,
                api_key,
                doc_id,
            ))
            save_image_chunks(doc_dir, [image_chunk])
            if not knowledge_to_vector([image_chunk]):
                raise RuntimeError("图片描述没有可写入向量库的内容")
            invalidate_rag_cache()
            return {
                "file_name": file_name,
                "chunk_count": 1,
                "image_processed": 1,
                "image_skipped": 0,
            }
        except Exception:
            if doc_dir.exists():
                remove_image_doc_dir(doc_dir, UPLOAD_DIRECTORY)
            raise

    try:
        markdown = file_content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Markdown 文件必须使用 UTF-8 编码") from error
    image_result = asyncio.run(enrich_markdown_images(
        markdown,
        api_key,
        source_path=str(file_path),
        doc_dir=UPLOAD_DIRECTORY / file_path.stem,
    ))
    file_path.write_text(image_result.markdown, encoding="utf-8")

    docs = load_notes(str(file_path))
    chunks = split_documents(docs)
    chunks.extend(image_result.image_chunks)
    save_image_chunks(file_path, image_result.image_chunks)
    if not knowledge_to_vector(chunks):
        file_path.unlink(missing_ok=True)
        manifest_path(file_path).unlink(missing_ok=True)
        raise RuntimeError("笔记切分后没有可写入向量库的内容")
    invalidate_rag_cache()
    return {
        "file_name": file_name,
        "chunk_count": len(chunks),
        "image_processed": image_result.image_processed,
        "image_skipped": image_result.image_skipped,
    }


# ---------------------------------------------------------------------------
# 页面样式（与原前端一致，保持演示观感）
# ---------------------------------------------------------------------------
st.set_page_config(page_title="笔记复习助手", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #203047; --muted: #718096; --cream: #faf8f3; --paper: #fffdf9;
        --sage: #dce9df; --sage-strong: #4f7a63; --peach: #f5d9c5;
        --peach-strong: #c76d4a; --line: #e9e2d7;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 82% 10%, rgba(220,233,223,.75), transparent 25rem),
            radial-gradient(circle at 55% 88%, rgba(245,217,197,.48), transparent 28rem),
            var(--cream);
    }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 {
        color: var(--ink); letter-spacing: -0.035em;
    }
    .block-container { max-width: 1260px; padding-top: .35rem; padding-bottom: 1rem; }
    /* 压缩 st.navigation 顶部导航与内容之间的留白，让页面更紧凑 */
    [data-testid="stNavigation"] { padding: 0 !important; margin: 0 !important; }
    [data-testid="stToolbar"] .rc-overflow {
        justify-content: center !important; padding-left: 250px !important; box-sizing: border-box;
    }
    [data-testid="stMain"] { padding-top: 0.15rem !important; }
    .app-brand {
        position: fixed; top: .72rem; left: max(1.4rem, calc(50vw - 620px)); z-index: 1000001;
        display: flex; align-items: center; gap: .5rem;
        color: #184d38; font-size: 1.25rem; line-height: 2rem; font-weight: 800;
    }
    .brand-logo { flex: 0 0 auto; }
    .brand-text { color: #184d38; }
    .page-heading { margin: .2rem 0 .45rem; }
    .page-heading h1 { font-size: 1.75rem; margin: 0 0 .12rem; }
    .page-heading p { color: var(--muted); margin: 0; }
    .dashboard-stat {
        padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 14px;
        background: rgba(255,253,249,.76); box-shadow: 0 8px 24px rgba(51,67,54,.05);
    }
    .dashboard-stat span { display: block; color: var(--muted); margin-bottom: .2rem; }
    .dashboard-stat strong { color: var(--sage-strong); font-size: 2rem; line-height: 1.15; }
    [data-testid="stFileUploader"] {
        background: #ffffffb8; border: 1px dashed #9db9a6; border-radius: 12px;
        padding: .3rem .45rem; max-width: 340px; margin-left: auto; margin-right: auto;
    }
    [data-testid="stFileUploader"] section { padding: .1rem; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { border: 0; background: transparent; padding: .1rem; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button { min-height: 1.8rem; font-size: .8rem; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small { display: none; }
    .stButton > button {
        background: var(--peach-strong); color: white; border: 0;
        border-radius: 10px; font-weight: 650; min-height: 2.7rem;
    }
    .stButton > button:disabled { background: #d8d9d3; color: #8b928c; }
    .stButton > button p { white-space: nowrap; }
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"] {
        background: var(--sage-strong); border-color: var(--sage-strong); color: #ffffff;
    }
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #426b55; border-color: #426b55;
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
    .st-key-chat_image_popover [data-testid="stPopoverButton"] {
        height: 3.25rem !important; min-height: 3.25rem !important;
        background: #4e72b8 !important; color: #ffffff !important;
        border: 1px solid #4e72b8 !important; border-radius: 14px !important;
    }
    .st-key-chat_image_popover [data-testid="stPopoverButton"]:hover {
        background: #3f62a5 !important; color: #ffffff !important;
        border-color: #3f62a5 !important;
    }
    [data-testid="stChatInput"] {
        height: 3.25rem; min-height: 3.25rem;
        background: #fffdf9; border: 1px solid var(--line); border-radius: 14px;
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
        color: var(--sage-strong); font-size: .76rem; font-weight: 750; letter-spacing: .11em;
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
    .note-item { background: #ffffff9e; border-radius: 10px; padding: .65rem .7rem; margin: .45rem 0; }
    .note-item b { font-size: .88rem; color: var(--ink); }
    .note-item span { display: block; color: var(--muted); font-size: .78rem; margin-top: .12rem; }
    .source-label { color: var(--sage-strong); font-size: .82rem; font-weight: 700; }
    .privacy-note { color: var(--muted); font-size: .86rem; text-align: center; margin-top: .8rem; }
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
        .st-key-chat_image_popover [data-testid="stPopoverButton"] {
            background: #4e72b8 !important; color: #ffffff !important;
            border-color: #4e72b8 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "note_uploader_version" not in st.session_state:
    st.session_state.note_uploader_version = 0
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

def has_valid_api_key() -> bool:
    current_key = st.session_state.get("deepseek_api_key", "").strip()
    validated_key = st.session_state.get("validated_api_key", "")
    return bool(current_key) and validated_key == current_key


def clear_api_key() -> None:
    st.session_state.deepseek_api_key = ""
    st.session_state.validated_api_key = ""
    if "api_key_input" in st.session_state:
        st.session_state.api_key_input = ""


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
                elif validate_deepseek_api_key is None:
                    st.session_state.validated_api_key = ""
                    st.error("当前环境缺少 Key 验证组件，请先部署最新代码后重试。")
                else:
                    try:
                        asyncio.run(validate_deepseek_api_key(candidate_key))
                        st.session_state.deepseek_api_key = candidate_key
                        st.session_state.validated_api_key = candidate_key
                        st.success("已保存，当前会话内有效")
                    except ImageProcessingError as error:
                        st.session_state.validated_api_key = ""
                        st.error(str(error))
            clear_column.button("清除 Key", use_container_width=True, on_click=clear_api_key)
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
            <hr><h4>技术栈</h4>
            <div><span class='tech-chip'>FastAPI</span><span class='tech-chip'>Streamlit</span>
            <span class='tech-chip'>LangChain</span><span class='tech-chip'>Chroma</span>
            <span class='tech-chip'>BM25</span><span class='tech-chip'>DeepSeek Vision</span></div>
            <hr><p><strong>开源地址</strong><br>
            <a class='github-link' href='https://github.com/yuyuyuoo55/Personal_Notes_Review_Assistant' target='_blank'>
                <svg width='17' height='17' viewBox='0 0 16 16' fill='currentColor' aria-hidden='true'><path d='M8 0C3.58 0 0 3.64 0 8.13c0 3.59 2.29 6.64 5.47 7.71.4.08.55-.18.55-.39 0-.19-.01-.83-.01-1.51-2.01.38-2.53-.5-2.69-.96-.09-.24-.48-.96-.82-1.15-.28-.15-.68-.53-.01-.54.63-.01 1.08.59 1.23.83.72 1.23 1.87.88 2.33.67.07-.53.28-.88.51-1.08-1.78-.21-3.64-.91-3.64-4.02 0-.89.31-1.62.82-2.19-.08-.21-.36-1.04.08-2.16 0 0 .67-.22 2.2.84A7.5 7.5 0 0 1 8 3.89c.68 0 1.36.09 2 .27 1.53-1.06 2.2-.84 2.2-.84.44 1.12.16 1.95.08 2.16.51.57.82 1.3.82 2.19 0 3.12-1.87 3.81-3.65 4.02.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.47.55.39A8.04 8.04 0 0 0 16 8.13C16 3.64 12.42 0 8 0Z'/></svg>
                GitHub ↗
            </a></p>
            <small>感谢每一位使用并提出反馈的朋友。</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_library_page() -> None:
    st.markdown(
        "<div class='page-heading'><h1>知识库</h1><p>集中导入、查看和管理用于检索的学习资料。</p></div>",
        unsafe_allow_html=True,
    )
    if not has_valid_api_key():
        st.warning("请先在「设置」页填写并验证 DeepSeek API Key。")
        return
    has_api_key = True
    notes = list_notes()
    existing_note_names = {note["file_name"] for note in notes}
    st.markdown("### 导入资料")
    if "note_import_success" in st.session_state:
        st.success(st.session_state.pop("note_import_success"))

    uploaded_file = st.file_uploader(
        "选择 Markdown 或图片文件",
        type=["md", "jpg", "jpeg", "png", "webp"],
        disabled=not has_api_key,
        label_visibility="collapsed",
        key=f"note_uploader_{st.session_state.note_uploader_version}",
    )
    is_duplicate_file = bool(uploaded_file and uploaded_file.name in existing_note_names)

    if is_duplicate_file:
        st.info(f"{uploaded_file.name} 已在笔记库中，无需重复导入。")

    if st.button(
        "导入到笔记库",
        use_container_width=True,
        disabled=not has_api_key or uploaded_file is None or is_duplicate_file,
    ):
        if uploaded_file is None:
            st.warning("请先选择一个文件。")
        else:
            try:
                result = import_note(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    st.session_state.deepseek_api_key,
                    uploaded_file.type,
                )
                image_note = ""
                if result["image_processed"]:
                    image_note = f" · 已识别 {result['image_processed']} 张图片"
                elif result["image_skipped"]:
                    image_note = f" · {result['image_skipped']} 张图片未识别，已跳过"
                st.session_state.note_import_success = (
                    f"已导入 {result['file_name']} · {result['chunk_count']} 个片段{image_note}"
                )
                list_notes.clear()
                st.session_state.note_uploader_version += 1
                st.rerun()
            except (ValueError, RuntimeError, ImageProcessingError) as error:
                st.error(str(error))

    st.divider()
    st.markdown("#### 笔记库")
    notes = list_notes()
    if not notes:
        st.caption("还没有导入笔记")
    else:
        if st.session_state.confirm_delete_all:
            st.warning("确认清空全部笔记？此操作无法撤销。")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("确认清空", type="primary", use_container_width=True):
                delete_all_notes()
                list_notes.clear()
                st.session_state.confirm_delete_all = False
                st.session_state.pending_note_delete = None
                st.rerun()
            if cancel_col.button("取消", use_container_width=True):
                st.session_state.confirm_delete_all = False
                st.rerun()
        elif st.button(
            "🗑 清空全部",
            use_container_width=True,
            disabled=not has_api_key,
        ):
            st.session_state.confirm_delete_all = True
            st.rerun()

        for note in notes:
            note_col, delete_col = st.columns(
                [3.6, 1.7], gap="small", vertical_alignment="center"
            )
            note_col.markdown(
                f"<div class='note-item'><b>📄 {note['file_name']}</b>"
                f"<span>{note['chunk_count']} 个知识片段</span></div>",
                unsafe_allow_html=True,
            )
            if delete_col.button(
                "🗑 删除",
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
                    identifier = note["doc_id"] if note["kind"] == "image" else Path(note["source"]).name
                    delete_note(identifier, note["kind"])
                    list_notes.clear()
                    st.session_state.pending_note_delete = None
                    st.rerun()
                if cancel_col.button(
                    "取消",
                    key=f"cancel_{note['note_id']}",
                    use_container_width=True,
                ):
                    st.session_state.pending_note_delete = None
                    st.rerun()


def render_dashboard_page() -> None:
    st.markdown(
        "<div class='page-heading'><h1>数据看板</h1><p>查看知识库的笔记规模、片段分布与资料类型。</p></div>",
        unsafe_allow_html=True,
    )
    if not has_valid_api_key():
        st.warning("请先在「设置」页填写并验证 DeepSeek API Key。")
        return
    notes = list_notes()
    note_count = len(notes)
    chunk_count = sum(int(note.get("chunk_count", 0)) for note in notes)

    note_column, chunk_column = st.columns(2, gap="medium")
    note_column.markdown(
        f"<div class='dashboard-stat'><span>总笔记数</span><strong>{note_count}</strong></div>",
        unsafe_allow_html=True,
    )
    chunk_column.markdown(
        f"<div class='dashboard-stat'><span>总片段数</span><strong>{chunk_count}</strong></div>",
        unsafe_allow_html=True,
    )

    if not notes:
        st.info("暂无数据，导入笔记后即可查看统计图表。")
        return

    chart_left, chart_right = st.columns([1.7, 1], gap="large")
    sorted_notes = sorted(notes, key=lambda note: int(note.get("chunk_count", 0)), reverse=True)
    file_names = [str(note.get("file_name", "未命名笔记")) for note in sorted_notes]
    chunk_counts = [int(note.get("chunk_count", 0)) for note in sorted_notes]

    with chart_left:
        st.markdown("### 各笔记片段数")
        bar_figure = go.Figure(
            go.Bar(
                x=file_names,
                y=chunk_counts,
                marker_color="#4f7a63",
                hovertemplate="%{x}<br>%{y} 个片段<extra></extra>",
            )
        )
        bar_figure.update_layout(
            height=360,
            margin=dict(l=12, r=12, t=12, b=72),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,253,249,.62)",
            showlegend=False,
            xaxis_title=None,
            yaxis_title="片段数",
            font=dict(color="#203047"),
        )
        bar_figure.update_xaxes(tickangle=-25, gridcolor="rgba(0,0,0,0)")
        bar_figure.update_yaxes(rangemode="tozero", gridcolor="#e9e2d7")
        st.plotly_chart(bar_figure, use_container_width=True, config={"displayModeBar": False})

    type_labels = {"md": "Markdown", "image": "图片"}
    type_counts: dict[str, int] = {}
    for note in notes:
        kind = str(note.get("kind") or "other")
        type_counts[kind] = type_counts.get(kind, 0) + 1

    with chart_right:
        st.markdown("### 文档类型占比")
        kinds = list(type_counts)
        pie_figure = go.Figure(
            go.Pie(
                labels=[type_labels.get(kind, kind.upper()) for kind in kinds],
                values=[type_counts[kind] for kind in kinds],
                hole=.58,
                marker=dict(colors=["#4f7a63", "#c76d4a", "#d4b56a", "#718096"]),
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value} 份 · %{percent}<extra></extra>",
            )
        )
        pie_figure.update_layout(
            height=360,
            margin=dict(l=12, r=12, t=12, b=12),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(color="#203047"),
            annotations=[dict(text=f"{note_count} 份", x=.5, y=.5, showarrow=False)],
        )
        st.plotly_chart(pie_figure, use_container_width=True, config={"displayModeBar": False})


current_page = "智能问答"


def select_page(page_name: str):
    def activate_page() -> None:
        global current_page
        current_page = page_name

    return activate_page


st.markdown(
    "<div class='app-brand'>"
    "<svg class='brand-logo' width='26' height='26' viewBox='0 0 32 32' fill='none' xmlns='http://www.w3.org/2000/svg'>"
    "<rect x='6' y='5' width='20' height='24' rx='3' fill='#184d38'/>"
    "<path d='M11 12h10M11 16h10M11 20h6' stroke='#fff' stroke-width='1.6' stroke-linecap='round'/>"
    "</svg>"
    "<span class='brand-text'>笔记复习助手</span>"
    "</div>",
    unsafe_allow_html=True,
)
navigation = st.navigation(
    [
        st.Page(select_page("智能问答"), title="智能问答", icon="💬", url_path="chat", default=True),
        st.Page(select_page("知识库"), title="知识库", icon="📚", url_path="library"),
        st.Page(select_page("数据看板"), title="数据看板", icon="📊", url_path="dashboard"),
        st.Page(select_page("设置"), title="设置", icon="⚙️", url_path="settings"),
        st.Page(select_page("关于"), title="关于", icon="ℹ️", url_path="about"),
    ],
    position="top",
)
navigation.run()

if current_page == "知识库":
    render_library_page()
    st.stop()
if current_page == "数据看板":
    render_dashboard_page()
    st.stop()
if current_page == "设置":
    render_settings_page()
    st.stop()
if current_page == "关于":
    render_about_page()
    st.stop()

has_api_key = has_valid_api_key()
if not has_api_key:
    st.markdown(
        "<div class='page-heading'><h1>智能问答</h1><p>围绕已导入的笔记进行检索与问答。</p></div>",
        unsafe_allow_html=True,
    )
    st.warning("请先在「设置」页填写并验证 DeepSeek API Key。")
    st.stop()

notes = list_notes()

# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
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
        st.session_state.conversation_id = uuid4().hex

    fast_column, accurate_column = st.columns(2, gap="small")
    with fast_column:
        if st.button(
            "快速模式",
            key="mode_fast",
            type="primary" if st.session_state.retrieval_mode == "fast" else "secondary",
            use_container_width=True,
        ):
            if st.session_state.retrieval_mode != "fast":
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
                if st.session_state.messages:
                    st.session_state.messages.append(
                        {"role": "mode", "content": "已切换到：精确查找（Step RAG）"}
                    )
                st.session_state.retrieval_mode = "accurate"
            st.rerun()

    mode_descriptions = {
        "fast": "当前链路：Agent 判断 →（直接回答 / 向量检索 Top-3）→ 基于片段回答　【线上主力 · 稳定】",
        "accurate": ("当前链路：原问题 → 查询改写 → 向量 + BM25 → RRF → 回答　"
                     "【云端无精排模型时自动降级：Cross-Encoder 精排一步会跳过】"),
    }
    st.markdown(
        f"<div class='mode-flow'>{mode_descriptions[st.session_state.retrieval_mode]}</div>",
        unsafe_allow_html=True,
    )

    chat_history = st.container(height=330, border=True)

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
                    sources = message.get("sources", [])
                    if sources:
                        with st.expander("参考笔记", expanded=False):
                            for source in sources:
                                header_path = " > ".join(source.header_path) or "未标注标题"
                                st.markdown(
                                    f"<div class='source-label'>📎 {source.file_name} · {header_path}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.write(source.content_preview)
                                if getattr(source, "image_path", None):
                                    st.image(source.image_path)
                                st.divider()
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

                started_at = perf_counter()
                preparing_reranker = False

                try:
                    with st.status("正在理解问题并检索笔记…", expanded=True) as request_status:
                        # 精确查找首次使用：提示精排模型准备。
                        # 部署环境若未安装 sentence_transformers / torch，精排会自动跳过，
                        # 精确模式降级为「查询改写 + 向量/BM25 双路召回 + RRF 融合」，不阻塞。
                        if st.session_state.retrieval_mode == "accurate":
                            if _HAS_SENTENCE_TRANSFORMERS:
                                preparing_reranker = True
                                request_status.update(
                                    label=(
                                        "正在从本机缓存加载精排模型…"
                                        if is_reranker_cached()
                                        else "首次使用：正在准备精排模型…"
                                    ),
                                    state="running",
                                )
                                get_reranker_model()
                                preparing_reranker = False
                                request_status.update(
                                    label="精排模型已就绪，正在执行完整检索…",
                                    state="running",
                                )
                            else:
                                request_status.update(
                                    label="精排模型未安装，精确模式将降级为混合检索+RRF 融合…",
                                    state="running",
                                )

                        if uploaded_chat_image:
                            request_status.update(
                                label="正在使用 DeepSeek Vision 理解图片…",
                                state="running",
                            )
                            image_bytes = uploaded_chat_image.getvalue()
                            mime = validate_image(image_bytes, uploaded_chat_image.type)
                            description = asyncio.run(describe_image_url(
                                image_data_url(image_bytes, mime),
                                st.session_state.deepseek_api_key,
                            ))
                            rag_query = f"图片内容：{description}\n\n用户问题：{question}"
                            preparation = prepare_rag_answer(
                                original_query=rag_query,
                                mode=st.session_state.retrieval_mode,
                                conversation_id=st.session_state.conversation_id,
                                api_key=st.session_state.deepseek_api_key,
                            )
                        else:
                            preparation = prepare_rag_answer(
                                original_query=question,
                                mode=st.session_state.retrieval_mode,
                                conversation_id=st.session_state.conversation_id,
                                api_key=st.session_state.deepseek_api_key,
                            )

                        # 精确查找在生成前已有最终来源；快速模式在工具节点后才有来源。
                        if preparation is not None and preparation.sources:
                            sources = preparation.sources
                            request_status.update(
                                label="已找到相关资料，正在生成回答…",
                                state="running",
                            )

                        for stream_event in preparation.answer_stream if preparation is not None else []:
                            if stream_event.event == "sources":
                                sources = stream_event.sources or []
                                request_status.update(
                                    label="已找到相关资料，正在生成回答…",
                                    state="running",
                                )
                            elif stream_event.event == "token":
                                answer += stream_event.content
                                answer_placeholder.markdown(f"{answer}▌")

                    request_status.update(label="回答完成", state="complete", expanded=False)
                except ImageProcessingError as error:
                    answer = str(error)
                    answer_placeholder.warning(answer)
                except Exception:
                    preparing_reranker = False
                    detail = (
                        "精排模型准备失败，请检查网络后重试。"
                        if preparing_reranker
                        else "本次问答暂时无法完成，请稍后重试。"
                    )
                    answer = detail
                    answer_placeholder.warning(detail)

                elapsed_ms = round((perf_counter() - started_at) * 1000)

                answer_placeholder.markdown(answer)
                if sources:
                    with st.expander("参考笔记", expanded=False):
                        for source in sources:
                            header_path = " > ".join(source.header_path) or "未标注标题"
                            st.markdown(
                                f"<div class='source-label'>📎 {source.file_name} · {header_path}</div>",
                                unsafe_allow_html=True,
                            )
                            st.write(source.content_preview)
                            if getattr(source, "image_path", None):
                                st.image(source.image_path)
                            st.divider()
                if answer and not answer.startswith("本次问答"):
                    st.caption(f"本次回答耗时：{elapsed_ms / 1000:.2f} 秒")
                # 无论是否带图，回答都应存入历史消息，否则 rerun 后纯文字回答会丢失。
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
                st.rerun()

with focus_column:
    mode_now = "快速模式（Agentic RAG）" if st.session_state.get("retrieval_mode", "fast") == "fast" else "精确查找（Step RAG）"
    mode_summary = (
        "Agent 自主判断是否检索；需要资料时调用向量检索，适合日常复习。"
        if st.session_state.get("retrieval_mode", "fast") == "fast"
        else "执行查询改写、双路召回与 RRF 融合；云端自动跳过本地精排模型。"
    )
    st.markdown(
        f"<div class='focus-card'><strong>本次复习 · {mode_now}</strong><span>{mode_summary}</span></div>",
        unsafe_allow_html=True,
    )
