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
import sys
import time
from pathlib import Path
from time import perf_counter
from uuid import uuid4

# 让脚本能从项目根目录 import backend.app.* 模块（config.py 也会加载根目录 .env）。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

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
from backend.app.services.image_chunk_store import save_image_chunks  # noqa: E402
from backend.app.services.note_loader import load_notes  # noqa: E402
from backend.app.services.note_splitter import split_documents  # noqa: E402
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
def list_notes() -> list[dict]:
    """返回已导入笔记及其已写入 Chroma 的片段数（等价后端 list_notes）。"""
    if not UPLOAD_DIRECTORY.exists():
        return []
    notes: list[dict] = []
    for file_path in UPLOAD_DIRECTORY.glob("*.md"):
        stored = vector_store.get(where={"source": str(file_path)})
        notes.append(
            {
                "file_name": file_path.name,
                "chunk_count": len(stored["ids"]),
            }
        )
    return notes


def import_note(file_name: str, file_content: bytes, api_key: str) -> dict:
    """保存笔记、切分、写入 Chroma，并让 RAG 缓存失效（等价后端 import_note）。"""
    if not file_name.lower().endswith(".md"):
        raise ValueError("当前只支持 .md 格式笔记")
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIRECTORY / file_name
    if file_path.exists():
        raise ValueError("同名笔记已存在；当前版本不重复导入")
    if not file_content:
        raise ValueError("上传文件不能为空")
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
    [data-testid="stSidebar"] { background: #eef2ed; border-right: 1px solid #d9e2da; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2.6rem; }
    /* 暗黑主题下侧边栏为浅色底，强制标题/说明文字用深色，避免看不清 */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, h1, h2, h3 {
        color: var(--ink); letter-spacing: -0.035em;
    }
    [data-testid="stSidebar"] h1 { font-size: 1.55rem; }
    .block-container { max-width: 1260px; padding-top: 3rem; padding-bottom: 2rem; }
    [data-testid="stFileUploader"] {
        background: #ffffffb8; border: 1px dashed #9db9a6; border-radius: 16px;
        padding: .7rem .85rem;
    }
    [data-testid="stFileUploader"] section { padding: .2rem; }
    [data-testid="stFileUploaderDropzone"] { border: 0; background: transparent; }
    [data-testid="stSidebar"] .stButton > button {
        background: var(--peach-strong); color: white; border: 0;
        border-radius: 10px; font-weight: 650; min-height: 2.7rem;
    }
    [data-testid="stSidebar"] .stButton > button:disabled { background: #d8d9d3; color: #8b928c; }
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"] {
        background: var(--sage-strong); border-color: var(--sage-strong); color: #ffffff;
    }
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #426b55; border-color: #426b55;
    }
    [data-testid="stChatInput"] {
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
    .section-title { color: var(--ink); font-size: 1.12rem; font-weight: 750; margin: .8rem 0 .75rem; }
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 侧边栏：导入笔记 + 笔记库
# ---------------------------------------------------------------------------
notes = list_notes()
existing_note_names = {note["file_name"] for note in notes}

if "note_uploader_version" not in st.session_state:
    st.session_state.note_uploader_version = 0
if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = ""
if "validated_api_key" not in st.session_state:
    st.session_state.validated_api_key = ""
if "chat_image_uploader_version" not in st.session_state:
    st.session_state.chat_image_uploader_version = 0

with st.sidebar:
    st.title("📚 笔记复习助手")
    st.caption("把课堂与技术笔记，变成可追溯的复习资料。")
    st.text_input(
        "请输入您的DeepSeek API Key",
        type="password",
        key="deepseek_api_key",
        placeholder="sk-...",
        help="用于问答的 DeepSeek API Key（在 platform.deepseek.com 申请）。仅保存在当前浏览器会话，不会写入数据库、日志或项目文件。",
    )
    if st.button("验证 Key", use_container_width=True):
        if validate_deepseek_api_key is None:
            st.error("当前环境缺少 Key 验证组件，请先部署最新代码后重试。")
        else:
            try:
                asyncio.run(validate_deepseek_api_key(st.session_state.deepseek_api_key.strip()))
                st.session_state.validated_api_key = st.session_state.deepseek_api_key.strip()
                st.success("您的 DeepSeek API Key 有效，可以使用")
            except ImageProcessingError as error:
                st.session_state.validated_api_key = ""
                st.error(str(error))
    has_api_key = bool(st.session_state.deepseek_api_key.strip()) and (
        st.session_state.validated_api_key == st.session_state.deepseek_api_key.strip()
    )
    if not has_api_key:
        st.info("请先输入您的 DeepSeek API Key，再进行提问或导入笔记。")
    st.markdown("#### 导入笔记")
    if "note_import_success" in st.session_state:
        st.success(st.session_state.pop("note_import_success"))

    uploaded_file = st.file_uploader(
        "选择 Markdown 文件",
        type=["md"],
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
        try:
            result = import_note(
                uploaded_file.name,
                uploaded_file.getvalue(),
                st.session_state.deepseek_api_key,
            )
            image_note = ""
            if result["image_processed"]:
                image_note = f" · 已识别 {result['image_processed']} 张图片"
            elif result["image_skipped"]:
                image_note = f" · {result['image_skipped']} 张图片未识别，已跳过"
            st.session_state.note_import_success = (
                f"已导入 {result['file_name']} · {result['chunk_count']} 个片段{image_note}"
            )
            st.session_state.note_uploader_version += 1
            st.rerun()
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

    st.divider()
    st.markdown("#### 笔记库")
    notes = list_notes()
    if not notes:
        st.caption("还没有导入笔记")
    else:
        for note in notes:
            st.markdown(
                f"<div class='note-item'><b>📄 {note['file_name']}</b>"
                f"<span>{note['chunk_count']} 个知识片段</span></div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
note_count = len(notes)
chunk_count = sum(note["chunk_count"] for note in notes)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="eyebrow">PERSONAL KNOWLEDGE SPACE</div>
        <h1>从你的笔记里，<em>重新理解知识。</em></h1>
        <p>提出问题，系统只依据已导入的学习资料回答，并保留可回看的来源。</p>
        <div class="scope-pill">当前检索范围 · {note_count} 份笔记 · {chunk_count} 个片段</div>
    </div>
    """,
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
            "⚡ 快速模式\n\nAgentic RAG · Agent 自主检索",
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
            "◎ 精确查找\n\nStep RAG · 固定完整链路",
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

    chat_history = st.container(height=500, border=True)

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
                                header_path = " > ".join(source["header_path"]) or "未标注标题"
                                st.markdown(
                                    f"<div class='source-label'>📎 {source['file_name']} · {header_path}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.write(source["content_preview"])
                                if source.get("image_path"):
                                    st.image(source["image_path"])
                                st.divider()
                    if "elapsed_ms" in message:
                        st.caption(f"本次回答耗时：{message['elapsed_ms'] / 1000:.2f} 秒")

    question = st.chat_input(
        "例如：RRF 和加权融合有什么区别？",
        disabled=not has_api_key,
    )
    uploaded_chat_image = st.file_uploader(
        "可选：上传图片，图片描述会随问题一起参与 RAG 检索",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        disabled=not has_api_key,
        key=f"chat_image_{st.session_state.chat_image_uploader_version}",
    )

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
                            header_path = " > ".join(source["header_path"]) or "未标注标题"
                            st.markdown(
                                f"<div class='source-label'>📎 {source['file_name']} · {header_path}</div>",
                                unsafe_allow_html=True,
                            )
                            st.write(source["content_preview"])
                            if source.get("image_path"):
                                st.image(source["image_path"])
                            st.divider()
                if answer and not answer.startswith("本次问答"):
                    st.caption(f"本次回答耗时：{elapsed_ms / 1000:.2f} 秒")
                if uploaded_chat_image:
                    st.session_state.chat_image_uploader_version += 1
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
                st.rerun()

with focus_column:
    st.markdown("<div class='section-title'>本次复习</div>", unsafe_allow_html=True)
    mode_now = "快速模式（Agentic RAG）" if st.session_state.get("retrieval_mode", "fast") == "fast" else "精确查找（Step RAG）"
    st.markdown(
        f"""
        <div class="focus-card">
            <strong>当前模式：{mode_now}</strong>
            <span>{'Quick · 线上主力：Agent 会自行判断是否检索；需要资料时只调用向量检索，稳定快速。'
                 if st.session_state.get('retrieval_mode', 'fast') == 'fast'
                 else 'Accurate · 云端提示：未内置精排模型（torch 等未部署），此模式自动降级为「查询改写 + 向量/BM25 双路召回 + RRF 融合」，Cross-Encoder 精排一步会跳过。'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='mini-step'><b>01 · 导入资料</b><small>按 Markdown 标题切分并建立索引</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-step'><b>02 · 基于来源问答</b><small>混合检索、精排后再生成回答</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-step'><b>03 · 章节小测</b><small>下一阶段开放</small></div>", unsafe_allow_html=True)
