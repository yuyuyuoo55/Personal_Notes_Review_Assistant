"""RAG 业务编排：快速 Agentic RAG 与精确 Step RAG。"""

from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document

from backend.app.core.config import DEEPSEEK_TEXT_MODEL, UPLOAD_DIRECTORY
from backend.app.schemas.note import SourceChunk
from backend.app.services.agent_service import FastAgentEvent, stream_fast_agent_answer
from backend.app.services.bm25_retriever import bm25_retriever, _HAS_PKUSEG
from backend.app.services.chat_service import generate_responses_based_on_the_data
from backend.app.services.note_loader import load_notes
from backend.app.services.image_chunk_store import load_image_chunks
from backend.app.services.note_splitter import split_documents
from backend.app.services.query_rewriter import query_rewrite
from backend.app.services.reranker import cross_encoder_reranker_index, _HAS_SENTENCE_TRANSFORMERS
from backend.app.services.rrf_fusion import rrf_fusion
from backend.app.services.vector_retriever import vector_retriever
from backend.app.storage.vector_store import vector_store


@dataclass
class RagStreamEvent:
    """RAG 层交给 SSE 接口的事件。"""

    event: Literal["sources", "token"]
    content: str = ""
    sources: list[SourceChunk] | None = None


@dataclass
class RagPreparation:
    """问答接口开始 SSE 输出前准备好的数据。"""

    rewritten_query: str
    mode: str
    sources: list[SourceChunk]
    answer_stream: Generator[RagStreamEvent, None, None]


# 新导入笔记后设为 True；下一次精确查找才重建 BM25。
_bm25_rebuild_required = True

# 精确查找仍保留该阈值：向量很远且 BM25 未命中时，不进入后续精排。
MAX_VECTOR_DISTANCE = 1.10
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def text_stream(text: str) -> Generator[RagStreamEvent, None, None]:
    """将普通文本包装成与模型流一致的 token 事件。"""
    yield RagStreamEvent(event="token", content=text)


def wrap_text_stream(stream: Generator[str, None, None]) -> Generator[RagStreamEvent, None, None]:
    """将精确查找现有的 str 流转换为统一 RAG 事件。"""
    for content in stream:
        yield RagStreamEvent(event="token", content=content)


def wrap_fast_agent_stream(
    stream: Generator[FastAgentEvent, None, None],
) -> Generator[RagStreamEvent, None, None]:
    """将快速 Agent 的事件转换为 RAG 层统一事件。"""
    for item in stream:
        yield RagStreamEvent(
            event=item.event,
            content=item.content,
            sources=item.sources,
        )


def no_material_preparation(original_query: str, mode: str) -> RagPreparation:
    """资料为空或不相关时，返回用户可理解的拒答，不抛内部异常。"""
    return RagPreparation(
        rewritten_query=original_query,
        mode=mode,
        sources=[],
        answer_stream=text_stream(
            "我没有在当前笔记库中找到足够可靠的资料，无法基于笔记回答这个问题。"
            "你可以导入相关笔记后再问。"
        ),
    )


def create_chat_model(api_key: str) -> ChatDeepSeek:
    """按请求创建模型，避免把任一用户 Key 留在跨请求缓存中。"""
    if not api_key:
        raise RuntimeError("请先输入API Key")
    return ChatDeepSeek(model=DEEPSEEK_TEXT_MODEL, api_key=api_key)


@lru_cache
def get_reranker_model():
    """优先加载本机 Cross-Encoder 缓存；缺失时才首次下载。

    部署环境未安装 sentence_transformers / torch 时返回 None，精排一步会自动跳过。
    """
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL_NAME, local_files_only=True)
    except OSError:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL_NAME)


def is_reranker_cached() -> bool:
    """检查 Hugging Face 本地缓存是否具备精排模型权重。"""
    snapshots_directory = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--BAAI--bge-reranker-base"
        / "snapshots"
    )
    if not snapshots_directory.exists():
        return False

    for snapshot in snapshots_directory.iterdir():
        has_config = (snapshot / "config.json").exists()
        has_weights = (
            (snapshot / "model.safetensors").exists()
            or (snapshot / "pytorch_model.bin").exists()
        )
        if has_config and has_weights:
            return True
    return False


@lru_cache
def load_all_chunks() -> tuple[Document, ...]:
    """读取 data/uploads 的所有 Markdown，并按既有规则切分。"""
    if not UPLOAD_DIRECTORY.exists():
        return ()

    all_chunks: list[Document] = []
    for file_path in UPLOAD_DIRECTORY.glob("*.md"):
        docs = load_notes(str(file_path))
        all_chunks.extend(split_documents(docs))
        all_chunks.extend(load_image_chunks(file_path))
    return tuple(all_chunks)


def invalidate_rag_cache() -> None:
    """导入新笔记后清空 Chunk 缓存，并标记 BM25 下次重建。"""
    global _bm25_rebuild_required
    load_all_chunks.cache_clear()
    _bm25_rebuild_required = True


def build_source_chunks(reranked_results: list[dict]) -> list[SourceChunk]:
    """将精确查找最终候选转成前端展示的来源 DTO。"""
    sources: list[SourceChunk] = []
    for result in reranked_results:
        metadata = result.get("metadata", {})
        header_path = []
        for header_name in ["Header 1", "Header 2", "Header 3"]:
            header = metadata.get(header_name)
            if header:
                header_path.append(header)

        sources.append(
            SourceChunk(
                file_name=Path(metadata.get("source", "未知文件")).name,
                header_path=header_path,
                chunk_id=str(result.get("id", "unknown")),
                content_preview=str(result.get("content", ""))[:200],
                image_path=metadata.get("image_path"),
            )
        )
    return sources


def prepare_rag_answer(
    original_query: str,
    mode: str = "fast",
    conversation_id: str = "",
    api_key: str = "",
) -> RagPreparation:
    """按模式准备流式 RAG 回答；快速模式由 Agent 自主决定是否检索。"""
    global _bm25_rebuild_required

    if mode not in {"fast", "accurate"}:
        raise ValueError("不支持的检索模式")

    # 0. 快速模式：create_agent 负责“直接答 / 调工具 / 根据工具结果再答”。
    if mode == "fast":
        chat_model = create_chat_model(api_key)
        return RagPreparation(
            rewritten_query=original_query,
            mode=mode,
            sources=[],
            answer_stream=wrap_fast_agent_stream(
                stream_fast_agent_answer(
                    query=original_query,
                    conversation_id=conversation_id,
                    chat_model=chat_model,
                    api_key=api_key,
                )
            ),
        )

    # 1. 精确查找：固定执行完整 Step RAG，不使用会话记忆。
    all_chunks = list(load_all_chunks())
    if not all_chunks:
        return no_material_preparation(original_query, mode)

    # 2. 查询改写只执行一次，随后两路检索共用改写后的问题。
    chat_model = create_chat_model(api_key)
    rewritten_query = query_rewrite(original_query, chat_model)

    # 3. 双路召回。
    vector_results = vector_retriever(
        query=rewritten_query,
        vector_store=vector_store,
        top_k=6,
    )
    bm25_results = bm25_retriever(
        query=rewritten_query,
        chunks_list=all_chunks,
        top_k=6,
        force_rebuild=_bm25_rebuild_required,
    )
    _bm25_rebuild_required = False

    # 4. 向量很远且 BM25 无命中时，认为没有可靠资料。
    best_vector_distance = vector_results[0][1] if vector_results else None
    has_keyword_hit = any(score > 0 for _, score in bm25_results)
    has_relevant_material = (
        best_vector_distance is not None and best_vector_distance <= MAX_VECTOR_DISTANCE
    ) or has_keyword_hit
    if not has_relevant_material:
        return no_material_preparation(original_query, mode)

    # 5. RRF 按排名融合两路结果，随后 Cross-Encoder 精排 Top-3。
    rrf_results = rrf_fusion([vector_results, bm25_results])
    reranked_results = cross_encoder_reranker_index(
        query=rewritten_query,
        rrf_results=rrf_results,
        cross_encoder=get_reranker_model(),
        top_k=3,
    )

    # 6. 只将最终片段交给生成模块，不回到 Agent。
    return RagPreparation(
        rewritten_query=rewritten_query,
        mode=mode,
        sources=build_source_chunks(reranked_results),
        answer_stream=wrap_text_stream(
            generate_responses_based_on_the_data(
                query=original_query,
                chat_model=chat_model,
                reranked_results=reranked_results,
            )
        ),
    )
