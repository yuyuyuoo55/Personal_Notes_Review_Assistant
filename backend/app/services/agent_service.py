"""快速模式 Agent：create_agent、按需检索与对话摘要裁剪。"""

from collections.abc import Generator
from dataclasses import dataclass, field
import json
import re
from hashlib import sha256
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from backend.app.core.config import UPLOAD_DIRECTORY
from backend.app.schemas.note import SourceChunk
from backend.app.services.vector_retriever import vector_retriever
from backend.app.storage.vector_store import vector_store


# 1. 为每个 conversation_id 保存 Agent 状态。
# 这是纯内存短期记忆；后端进程重启后会清空。
checkpointer = InMemorySaver()


# 2. 历史对话超过阈值时的压缩规则。
MEMORY_SUMMARY_PROMPT = """
你负责压缩个人笔记复习助手的旧对话。

请只保留后续追问需要的信息：
- 用户姓名、偏好、学习目标；
- 当前讨论的技术主题；
- 已经确认过的结论；
- “它、刚才那个、这个命令”等指代对象。

不要编造信息。
不要保留无关寒暄。
用简洁中文输出摘要。

旧对话：
{messages}
""".strip()


# 3. Agent 行为规则。
FAST_AGENT_SYSTEM_PROMPT = """
你是个人笔记复习助手的快速模式 Agent。

规则：
1. 问候、身份介绍、感谢等普通聊天，可以直接回答。
2. 技术概念、命令、代码、笔记内容相关的问题，必须调用 search_personal_notes。
3. 用户说“刚才那个”“它”“这个命令”时，要结合历史对话理解指代；若属于技术追问，也必须调用 SearchPersonalNotes。
4. 工具返回 found 为 true 时，只能依据返回的笔记片段回答；不要补充片段以外的知识。
5. 工具返回 found 为 false 时，第一句明确回复“没有找到对应片段”，不要编造答案。
6. 回答简洁、自然；使用笔记时说明来源文件名。
7. 若检索片段含图片且与问题相关，引用该图片中的信息；无关图片不要引用。
""".strip()


# 4. 兼容部分 DeepSeek 响应把工具调用保留为 DSML 文本的情况。
# 原项目已经遇到过这个问题；中间件负责把 DSML 文本改回标准 tool_calls，
# 这样 create_agent 才能继续自动执行工具。
RAW_SEARCH_TOOL_PATTERN = re.compile(
    r'<[^>]*invoke name="search_personal_notes"[^>]*>(?P<body>.*?)<[^>]*?/invoke>',
    re.DOTALL,
)
RAW_QUERY_PARAMETER_PATTERN = re.compile(
    r'<[^>]*parameter name="query"[^>]*>(?P<query>.*?)<[^>]*?/parameter>',
    re.DOTALL,
)


class DeepSeekToolCallMiddleware(AgentMiddleware):
    """把未被 LangChain 解析的 DeepSeek DSML 工具调用转成标准 AIMessage。"""

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        last_message = messages[-1]
        if last_message.tool_calls:
            return None

        raw_content = str(last_message.content or "")
        tool_match = RAW_SEARCH_TOOL_PATTERN.search(raw_content)
        if tool_match is None:
            return None

        query_match = RAW_QUERY_PARAMETER_PATTERN.search(tool_match.group("body"))
        search_query = query_match.group("query").strip() if query_match else ""

        # 使用相同消息 ID 覆盖 DSML 原消息，避免 DSML 文本进入后续上下文或前端。
        normalized_message = AIMessage(
            id=last_message.id,
            content="",
            tool_calls=[
                {
                    "name": "search_personal_notes",
                    "args": {"query": search_query},
                    "id": "deepseek_dsml_search_call",
                }
            ],
        )
        return {"messages": [normalized_message]}


# 5. 定义 Agent 可以调用的检索工具。
@tool
def search_personal_notes(query: str) -> str:
    """在用户已导入的个人笔记中检索与问题相关的 Top-3 语义片段。"""

    # 5.1 没有笔记时不访问可能遗留旧数据的向量库。
    if not UPLOAD_DIRECTORY.exists() or not any(UPLOAD_DIRECTORY.glob("*.md")):
        return json.dumps(
            {"found": False, "chunks": []},
            ensure_ascii=False,
        )

    # 5.2 保留现有 vector_retriever 的返回格式：(Document, L2 距离)。
    results = vector_retriever(
        query=query,
        vector_store=vector_store,
        top_k=3,
    )

    if not results:
        return json.dumps(
            {"found": False, "chunks": []},
            ensure_ascii=False,
        )

    # 5.3 排名来自列表顺序；distance 是 Chroma 返回的 L2 距离。
    # 不在快速模式硬编码距离阈值，是否足够回答交给最终模型按提示词判断。
    # 这里
    chunks: list[dict[str, Any]] = []
    # 注意:L2 距离本质是在向量空间里算两点相隔多远：
    # rank 越小 → 排名越靠前
    # distance 越小 → 语义越接近
    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        chunks.append(
            {
                "id": str(metadata.get("chunk_id", "unknown")),
                "rank": rank,
                "distance": round(float(distance), 4),# round:确保它是普通 Python 浮点数 保留四位
                "content": document.page_content,
                "metadata": metadata,
            }
        )

    return json.dumps(
        {"found": True, "chunks": chunks},
        ensure_ascii=False,
    )


@dataclass
class FastAgentEvent:
    """快速模式交给 SSE 接口的两类事件：来源或文本 token。"""

    event: Literal["sources", "token"]
    content: str = ""
    sources: list[SourceChunk] = field(default_factory=list)
    """
    default_factory=list 表示：
    default_factory=list 表示：
    每创建一个 FastAgentEvent
    → 都新建自己的空列表
    这和 Java 中每个对象各自：
    new ArrayList<>()
    是一个意思。
    """

def _build_source_chunks(chunks: list[dict[str, Any]]) -> list[SourceChunk]:
    """将工具返回的 Chunk 转成前端已有的 SourceChunk DTO。"""
    sources: list[SourceChunk] = []

    for chunk in chunks:
        # 组装标题路径
        metadata = chunk.get("metadata", {})
        header_path = []
        for header_name in ["Header 1", "Header 2", "Header 3"]:
            header = metadata.get(header_name)
            if header:
                header_path.append(header)

        # 组装文件来源
        source_path = str(metadata.get("source", "未知文件"))
        file_name = source_path.replace("\\", "/").split("/")[-1]
        sources.append(
            SourceChunk(
                file_name=file_name,
                header_path=header_path,
                chunk_id=str(chunk.get("id", "unknown")),
                content_preview=str(chunk.get("content", ""))[:200],
                image_path=metadata.get("image_path"),
            )
        )

    return sources


def get_fast_agent(chat_model):
    """按请求创建 Agent，避免首个用户模型及 Key 被全局单例长期持有。"""
    return create_agent(
        model=chat_model,
        tools=[search_personal_notes],
        system_prompt=FAST_AGENT_SYSTEM_PROMPT,
        middleware=[
            DeepSeekToolCallMiddleware(),
            SummarizationMiddleware(
                model=chat_model,
                trigger=("messages", 14),
                keep=("messages", 12),
                trim_tokens_to_summarize=1200,
                summary_prompt=MEMORY_SUMMARY_PROMPT,
            ),
        ],
        checkpointer=checkpointer,
    )


def _parse_tool_payload(message: ToolMessage) -> dict[str, Any] | None:
    """只识别本项目检索工具返回的 JSON；其他工具消息忽略。"""
    try:
        payload = json.loads(str(message.content))
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict) or "found" not in payload or "chunks" not in payload:
        return None
    return payload


def stream_fast_agent_answer(
    *,
    query: str,
    conversation_id: str,
    chat_model,
    api_key: str,
) -> Generator[FastAgentEvent, None, None]:
    """运行 Agent，并同时输出“工具来源事件”和“模型文本 token 事件”。"""
    fast_agent = get_fast_agent(chat_model)
    # 相同 conversation_id 在不同 BYOK 用户之间也必须隔离；只保存不可逆摘要。
    key_namespace = sha256(api_key.encode()).hexdigest()[:16]
    thread_id = f"{key_namespace}:{conversation_id}"

    # 6. 明确使用 HumanMessage 表示当前用户问题。
    # 同一个 thread_id 会自动续接此前的原始消息或摘要。
    stream = fast_agent.stream(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["updates", "messages"],
    )

    tool_has_run = False
    pending_direct_answer_tokens: list[str] = []
    # 暂存第一次模型输出的文字。 在还没确认 Agent 是否会调工具之前，不能立刻把文字发给页面。

    for stream_mode, data in stream:
        # 6.1 tools 节点结束后会产生 ToolMessage。
        # 这时先把来源传给 SSE，再继续输出后续模型回答。
        if stream_mode == "updates":
            for node_update in data.values():
                # 某些中间件节点只表示“没有状态更新”，值会是 None。
                # 这种节点不包含 ToolMessage，直接跳过即可。
                if not isinstance(node_update, dict):
                    continue
                for message in node_update.get("messages", []):
                    if not isinstance(message, ToolMessage):
                        continue

                    tool_payload = _parse_tool_payload(message)
                    if tool_payload is None:
                        continue

                    tool_has_run = True
                    pending_direct_answer_tokens.clear()
                    yield FastAgentEvent(
                        event="sources",
                        sources=_build_source_chunks(tool_payload.get("chunks", [])),
                    )

        # 6.2 messages 模式提供模型逐 token 输出。
        elif stream_mode == "messages":
            message_chunk, metadata = data
            if metadata.get("langgraph_node") != "model":
                continue

            content = message_chunk.content
            if not isinstance(content, str) or not content:
                continue

            # 首次模型调用既可能是“直接回答”，也可能是 DeepSeek DSML 工具调用。
            # 因此先缓存，只有工具结束后才持续转发，避免 DSML 内部文本泄露到页面。
            if tool_has_run:
                yield FastAgentEvent(event="token", content=content)
            else:
                pending_direct_answer_tokens.append(content)

    # 6.3 普通聊天没有工具节点：图结束后才确认这些 token 是直接回答。
    # 这类回答很短，优先保证不会把内部工具文本显示给用户。
    if not tool_has_run:
        for content in pending_direct_answer_tokens:
            yield FastAgentEvent(event="token", content=content)
