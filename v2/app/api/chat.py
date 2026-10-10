"""RAG/聊天流接口，Token暂存，最终校验成功才发送done。"""
import asyncio
import json
from contextlib import suppress

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from ..graph import build_graph
from ..rag.rag_answer import cited_answer_sources
from .runtime import ApiError, DeferredGraphStore, run_job, runtime_from
from .schemas import ChatRequest
from .streaming import StreamingAnswerModel

router = APIRouter(prefix="/chat", tags=["chat"])


def event(name, data):
    return "event: " + name + "\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n"


async def chat_events(request, runtime, payload):
    """【业务需求】输出Graph真实回答增量、已校验来源与明确结束/错误。
    【输入】HTTP请求、运行时、问题/历史/联网开关。
    【执行步骤】线程创建Graph；异步读取custom/updates；断开时关闭流，失败不发送成功done。
    【输出】status/token/sources/done/error SSE；token在done前属于未完成草稿。
    """
    stream = None
    stream_lock = False
    try:
        yield event("status", {"stage": "processing"})
        def prepare():
            model = runtime.model_factory()
            return build_graph(model, DeferredGraphStore(runtime), answer_model=StreamingAnswerModel(model),
                               web_search=runtime.web_search)
        graph = await run_job(runtime, prepare)
        # 整段流都占用业务锁，避免生成中途更换 Key 或同时启动导入。
        if not runtime.job_lock.acquire(blocking=False):
            raise ApiError(409, "busy", "已有业务任务正在处理。")
        stream_lock = True
        history = [HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
                   for m in payload.history]
        state = {"messages": history + [HumanMessage(content=payload.question)], "web_enabled": payload.web_enabled}
        answer = None
        sources = []
        route = None
        async with asyncio.timeout(runtime.timeout):
            stream = graph.astream(state, stream_mode=["custom", "updates"])
            async for kind, data in stream:
                if await request.is_disconnected():
                    return
                if kind == "custom" and data.get("event") == "token":
                    yield event("token", {"text": data["text"]})
                elif kind == "updates":
                    for node, update in data.items():
                        if not isinstance(update, dict):
                            continue
                        route = update.get("route", route)
                        yield event("status", {"stage": node, "state": "completed"})
                        if update.get("web_error"):
                            yield event("status", {"stage": "web_search", "warning": update["web_error"]})
                        # Daily 联网仍由 daily 结束，复用真实 Token 和相同来源/完成校验。
                        if node in {"daily", "rag_answer"}:
                            answer = update["messages"][-1].content
                            # 防止候选/旧来源进入公共 SSE；只对应本轮已校验的最终回答。
                            sources = cited_answer_sources(update.get("answer_sources", []),answer)
            if answer is None:
                raise ValueError("工作流未生成最终回答。")
            yield event("sources", {"items": sources})
            yield event("done", {"status": "completed", "answer": answer, "route": route})
    except asyncio.CancelledError:
        raise
    except ApiError as error:
        yield event("error", {"status": "failed", "code": error.code, "message": error.message})
    except Exception:
        yield event("error", {"status": "failed", "message": "回答未完成，请检查配置或稍后重试；请丢弃本轮未完成草稿。"})
    finally:
        if stream is not None:
            with suppress(Exception):
                await stream.aclose()
        if stream_lock:
            runtime.job_lock.release()


@router.post("/stream", response_class=StreamingResponse, responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}})
async def chat_stream(request: Request, payload: ChatRequest):
    return StreamingResponse(chat_events(request, runtime_from(request), payload), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
