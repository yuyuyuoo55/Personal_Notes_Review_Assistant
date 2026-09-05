"""问答接口：以 SSE 依次发送来源、回答 token 与耗时。"""

import json
from collections.abc import Generator
from time import perf_counter

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.schemas.chat import ChatRequest
from backend.app.core.auth import require_user_deepseek_api_key
from backend.app.services.multimodal_service import (
    ImageProcessingError,
    describe_image_url,
    image_data_url,
    validate_image,
)
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
def chat(
    request: ChatRequest,
    api_key: str = Depends(require_user_deepseek_api_key),
) -> StreamingResponse:
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
                api_key=api_key,
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

        except Exception as error:
            # SSE 已开启后无法改 HTTP 状态码，只返回用户可理解的信息。
            error_name = type(error).__name__.lower()
            if "authentication" in error_name or "permission" in error_name:
                detail = "API Key无效，请检查后重试。"
            elif preparing_reranker:
                detail = "精排模型准备失败，请检查网络后重试。"
            else:
                detail = "本次问答暂时无法完成，请稍后重试。"
            yield sse_event("token", {"content": detail})
            elapsed_ms = round((perf_counter() - started_at) * 1000)
            yield sse_event("done", {"elapsed_ms": elapsed_ms})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/image")
async def chat_with_image(
    query: str = Form(..., min_length=1, max_length=500),
    mode: str = Form("fast"),
    conversation_id: str = Form(""),
    image: UploadFile = File(...),
    api_key: str = Depends(require_user_deepseek_api_key),
) -> StreamingResponse:
    """Describe the uploaded image, then use description plus question for RAG."""
    image_bytes = await image.read()
    content_type = image.content_type or "application/octet-stream"

    try:
        mime = validate_image(image_bytes, content_type)
        description = await describe_image_url(image_data_url(image_bytes, mime), api_key)
        rag_query = f"图片内容：{description}\n\n用户问题：{query}"
        preparation = prepare_rag_answer(
            original_query=rag_query,
            mode=mode,
            conversation_id=conversation_id,
            api_key=api_key,
        )
        preparation_error = None
    except ImageProcessingError as error:
        preparation = None
        preparation_error = str(error)

    def event_stream() -> Generator[str, None, None]:
        started_at = perf_counter()
        if preparation_error:
            yield sse_event("meta", {"rewritten_query": query, "mode": mode, "sources": []})
            yield sse_event("token", {"content": preparation_error})
        else:
            meta_sent = False
            if mode == "accurate":
                yield sse_event("meta", {"rewritten_query": preparation.rewritten_query, "mode": mode, "sources": [source.model_dump() for source in preparation.sources]})
                meta_sent = True
            for item in preparation.answer_stream:
                if item.event == "sources":
                    yield sse_event("meta", {"rewritten_query": preparation.rewritten_query, "mode": mode, "sources": [source.model_dump() for source in item.sources or []]})
                    meta_sent = True
                elif item.event == "token":
                    if not meta_sent:
                        yield sse_event("meta", {"rewritten_query": preparation.rewritten_query, "mode": mode, "sources": []})
                        meta_sent = True
                    yield sse_event("token", {"content": item.content})
        yield sse_event(
            "done",
            {"elapsed_ms": round((perf_counter() - started_at) * 1000)},
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
