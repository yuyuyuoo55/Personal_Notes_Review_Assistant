"""问答接口：以 SSE 依次发送来源、回答 token 与耗时。"""

import json
from collections.abc import Generator
from time import perf_counter

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.app.schemas.chat import ChatRequest
from backend.app.services.rag_service import (
    get_reranker_model,
    is_reranker_cached,
    prepare_rag_answer,
)


router = APIRouter(prefix="/api/chat", tags=["chat"])


def sse_event(event_name: str, data: dict) -> str:
    """将 Python 字典包装成浏览器可识别的 SSE 事件。"""
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
def chat(request: ChatRequest) -> StreamingResponse:
    """保持原有 SSE 协议；快速模式的来源在 Agent 工具执行后发送。"""

    def event_stream() -> Generator[str, None, None]:
        started_at = perf_counter()
        preparing_reranker = False

        try:
            # 精确查找首次使用时，先通知前端 Cross-Encoder 的准备状态。
            if request.mode == "accurate":
                preparing_reranker = True
                if is_reranker_cached():
                    yield sse_event(
                        "stage",
                        {"progress": 25, "label": "正在从本机缓存加载精排模型…"},
                    )
                else:
                    yield sse_event(
                        "stage",
                        {"progress": 5, "label": "首次使用：正在下载精排模型到本机缓存…"},
                    )

                get_reranker_model()
                preparing_reranker = False
                yield sse_event(
                    "stage",
                    {"progress": 70, "label": "精排模型已就绪，正在执行完整检索…"},
                )

            preparation = prepare_rag_answer(
                original_query=request.query,
                mode=request.mode,
                conversation_id=request.conversation_id,
            )

            # 精确查找在开始生成前就已有最终来源；快速模式要等工具节点结束。
            meta_sent = False
            if request.mode == "accurate":
                yield sse_event(
                    "meta",
                    {
                        "rewritten_query": preparation.rewritten_query,
                        "mode": preparation.mode,
                        "sources": [source.model_dump() for source in preparation.sources],
                    },
                )
                meta_sent = True

            for stream_event in preparation.answer_stream:
                if stream_event.event == "sources":
                    yield sse_event(
                        "meta",
                        {
                            "rewritten_query": preparation.rewritten_query,
                            "mode": preparation.mode,
                            "sources": [source.model_dump() for source in stream_event.sources or []],
                        },
                    )
                    meta_sent = True

                elif stream_event.event == "token":
                    # 普通聊天不调用工具，因此首次输出 token 前补一个空来源 meta。
                    if not meta_sent:
                        yield sse_event(
                            "meta",
                            {
                                "rewritten_query": preparation.rewritten_query,
                                "mode": preparation.mode,
                                "sources": [],
                            },
                        )
                        meta_sent = True

                    yield sse_event("token", {"content": stream_event.content})

            elapsed_ms = round((perf_counter() - started_at) * 1000)
            yield sse_event("done", {"elapsed_ms": elapsed_ms})

        except Exception:
            # SSE 已开启后无法改 HTTP 状态码，只返回用户可理解的信息。
            detail = (
                "精排模型准备失败，请检查网络后重试。"
                if preparing_reranker
                else "本次问答暂时无法完成，请稍后重试。"
            )
            yield sse_event("token", {"content": detail})
            elapsed_ms = round((perf_counter() - started_at) * 1000)
            yield sse_event("done", {"elapsed_ms": elapsed_ms})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
