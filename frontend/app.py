"""个人笔记复习助手 Streamlit 前端。"""

import json
import os
import time
from uuid import uuid4

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEEPSEEK_API_KEY_HEADER = "X-DeepSeek-API-Key"

if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = ""
if "chat_image_uploader_version" not in st.session_state:
    st.session_state.chat_image_uploader_version = 0

st.set_page_config(page_title="笔记复习助手", page_icon="📚", layout="wide")

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
    [data-testid="stSidebar"] {
        background: #eef2ed;
        border-right: 1px solid #d9e2da;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2.6rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
    [data-testid="stSidebar"] h1 { font-size: 1.55rem; }
    .block-container { max-width: 1260px; padding-top: 3rem; padding-bottom: 2rem; }
    [data-testid="stFileUploader"] {
        background: #ffffffb8;
        border: 1px dashed #9db9a6;
        border-radius: 16px;
        padding: .7rem .85rem;
    }
    [data-testid="stFileUploader"] section { padding: .2rem; }
    [data-testid="stFileUploaderDropzone"] { border: 0; background: transparent; }
    [data-testid="stSidebar"] .stButton > button {
        background: var(--peach-strong);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
    }
    [data-testid="stSidebar"] .stButton > button:disabled {
        background: #d8d9d3;
        color: #8b928c;
    }
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
            st.divider()


notes, notes_load_error = get_notes()
existing_note_names = {note["file_name"] for note in notes}

# 导入成功后递增 key，使 Streamlit 重建上传控件并清空刚才选中的文件。
# 否则页面 rerun 后仍保留该文件，会马上被“重复导入”校验命中，容易造成误解。
if "note_uploader_version" not in st.session_state:
    st.session_state.note_uploader_version = 0

with st.sidebar:
    st.title("📚 笔记复习助手")
    st.caption("把课堂与技术笔记，变成可追溯的复习资料。")
    st.text_input(
        "请输入您的DeepSeek API Key",
        type="password",
        key="deepseek_api_key",
        help="仅保存在当前浏览器会话，并随单次请求发送；不会写入数据库或日志。",
    )
    has_api_key = bool(st.session_state.deepseek_api_key.strip())
    if not has_api_key:
        st.info("请先输入API Key")
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
    is_duplicate_file = bool(
        uploaded_file and uploaded_file.name in existing_note_names
    )

    if is_duplicate_file:
        st.info(f"{uploaded_file.name} 已在笔记库中，无需重复导入。")

    if st.button(
        "导入到笔记库",
        use_container_width=True,
        disabled=not has_api_key or uploaded_file is None or is_duplicate_file,
    ):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/markdown",
                )
            }
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
            if result.get("warnings"):
                st.session_state.note_import_success += (
                    f"；{result.get('image_processed', 0)} 张图片已识别，"
                    f"{result.get('image_skipped', 0)} 张已跳过"
                )
            st.session_state.note_uploader_version += 1
            st.rerun()
        except httpx.HTTPStatusError as error:
            # 后端可能返回 JSON 业务错误，也可能在异常时返回空响应或 HTML。
            # 前端不能再直接 .json()，否则会把 JSONDecodeError 暴露给用户。
            try:
                detail = error.response.json().get("detail", "笔记导入失败")
            except ValueError:
                detail = f"笔记导入失败（后端状态码：{error.response.status_code}）"
            st.error(detail)
        except httpx.HTTPError:
            st.error("无法连接后端，请先启动项目")

    st.divider()
    st.markdown("#### 笔记库")

    if notes_load_error:
        st.warning(notes_load_error)
    elif not notes:
        st.caption("还没有导入笔记")
    else:
        for note in notes:
            st.markdown(
                f"<div class='note-item'><b>📄 {note['file_name']}</b><span>{note['chunk_count']} 个知识片段</span></div>",
                unsafe_allow_html=True,
            )


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
        # 仅作为本次浏览器会话的内存键；刷新并新建会话或重启后端都会清空记忆。
        st.session_state.conversation_id = uuid4().hex

    # 模式选择位于问答区顶部；每次提问都把当前模式一起发送给 FastAPI。
    fast_column, accurate_column = st.columns(2, gap="small")
    with fast_column:
        if st.button(
            "⚡ 快速模式\n\nAgentic RAG · Agent 自主检索",
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
            "◎ 精确查找\n\nStep RAG · 固定完整链路",
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
                    render_sources(message.get("sources", []))
                    if "elapsed_ms" in message:
                        st.caption(f"本次回答耗时：{message['elapsed_ms'] / 1000:.2f} 秒")

    question = st.chat_input(
        "例如：RRF 和加权融合有什么区别？",
        disabled=not has_api_key,
    )
    uploaded_chat_image = st.file_uploader(
        "可选：上传图片并随问题一起发送（图片问答不走 RAG）",
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
                                "data": {"query": question},
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
    st.markdown("<div class='section-title'>本次复习</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="focus-card">
            <strong>{'当前模式：快速模式（Agentic RAG）' if st.session_state.get('retrieval_mode', 'fast') == 'fast' else '当前模式：精确查找（Step RAG）'}</strong>
            <span>{'Agent 会自行判断是否检索；需要资料时只调用向量检索，适合日常复习。' if st.session_state.get('retrieval_mode', 'fast') == 'fast' else '固定执行改写、双路召回、RRF 与精排，适合术语和需要准确来源的问题。'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='mini-step'><b>01 · 导入资料</b><small>按 Markdown 标题切分并建立索引</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-step'><b>02 · 基于来源问答</b><small>混合检索、精排后再生成回答</small></div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-step'><b>03 · 章节小测</b><small>下一阶段开放</small></div>", unsafe_allow_html=True)
