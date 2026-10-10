"""Phase 6：从同一个 Chroma 集合读取笔记，使用 pkuseg + bm25s 检索。"""

from functools import lru_cache
from typing import Any, TypedDict

import bm25s
from langchain_chroma import Chroma


class BM25Result(TypedDict):
    """关键词检索结果；score 越大越相关，不是向量距离。"""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float

# 创建缓存分词器
@lru_cache(maxsize=1)
def get_segmenter():
    """首次关键词检索时创建中文分词器，后续复用，日常聊天无需初始化。"""
    try:
        import pkuseg
    except ModuleNotFoundError as error:
        if error.name != "pkuseg":
            raise
        # Cloud 的 Python 3.12 wheel 使用维护中的同算法 fork；不跳过 BM25。
        import spacy_pkuseg as pkuseg
    return pkuseg.pkuseg()


def retrieve_bm25(query: str, vector_store: Chroma, top_k: int = 3) -> list[BM25Result]:
    """
    【业务需求】
    用改写后的问题进行关键词检索，与向量检索共用笔记和片段 ID。
    输入：rewritten_query、当前 Chroma 对象、TopK。

    【执行步骤】
    1. 检查问题和 TopK。
    2. 从同一个 Chroma 集合读取全部片段，空库直接返回空列表。
    3. 整理片段 ID、正文和元数据，建立中文分词语料。
    4. 使用 bm25s 在内存中建立 BM25 索引。
    5. 对问题分词，检索不超过片段总数的 TopK 结果。
    6. 返回片段及原始 BM25 分数。
    输出：[{id, content, metadata, score}]，不生成答案。
    """
    # 1. 检查改写后的问题和 TopK。
    if not isinstance(query, str) or not query.strip():
        raise ValueError("BM25 检索需要非空的 rewritten_query。")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数。")

    # 2. 读取全部片段，不以向量检索的 TopK 作为 BM25 语料。
    # 而 ids 是 Chroma 的 get() 默认就会返回的字段，不需要放进 include。
    # include告诉 Chroma，我这次想从数据库里取出哪些信息。
    records = vector_store.get(include=["documents", "metadatas"])
    if not records["ids"]:
        return []

    # 3. 使用数据库 ID，与向量结果中的 id 一致；正文使用同一个分词器。
    segmenter = get_segmenter() # 创建分词器对象
    # corpus_tokens 负责帮助检索，corpus 负责提供检索命中后的完整笔记信息。
    corpus = [] # 保存完整笔记信息
    corpus_tokens = [] # 保存每条笔记切分后的关键词
    for index in range(len(records["ids"])):
        chunk_id = records["ids"][index]
        content = records["documents"][index]
        metadata = dict(records["metadatas"][index] or {})
        metadata.setdefault("chunk_id", chunk_id)
        corpus.append({"id": chunk_id, "content": content, "metadata": metadata})
        corpus_tokens.append(segmenter.cut(content))

    # 4. 沿用 V1 的成熟库，默认 Lucene BM25 评分；每次读取当前语料建索引。
    bm25_index = bm25s.BM25(corpus=corpus)
    bm25_index.index(corpus_tokens, show_progress=False)

    # 5. bm25s 接收批量查询，因此将当前问题的词列表放入外层列表。
    query_tokens = [segmenter.cut(query.strip())]
    limit = min(top_k, len(corpus))
    # matches 保存命中的原始笔记，scores 保存它们各自的 BM25 分数。
    matches, scores = bm25_index.retrieve(query_tokens, k=limit, show_progress=False)

    # 6. 保留实际 BM25 分数，用 score 与向量结果的 distance 明确区分。
    results = []
    for rank, item in enumerate(matches[0]):
        result = dict(item)
        result["score"] = float(scores[0, rank])
        results.append(result)
    return results
