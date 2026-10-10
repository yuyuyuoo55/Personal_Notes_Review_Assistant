"""Phase 7：仅按排名融合 Vector 和 BM25，不使用原始距离或分数。"""

from typing import Any, TypedDict

from .bm25_retrieval import BM25Result
from .vector_retrieval import VectorResult


class FusedResult(TypedDict):
    """融合后的片段及排名，未命中的检索器对应排名为 None。"""

    id: str
    content: str
    metadata: dict[str, Any]
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None


def rrf_fusion(
    vector_results: list[VectorResult],
    bm25_results: list[BM25Result],
    rrf_k: int = 60,
) -> list[FusedResult]:
    """
    【业务需求】
    将两路检索结果按数据库 ID 去重，再按排名累加 RRF 分数。
    【输入】
    vector_results、bm25_results：各自已排序的结果；rrf_k 沿用 V1 默认值 60。
    【执行步骤】
    1. 创建按片段 ID 保存的融合字典，并标记两路排名字段。
    2. 遍历每一路结果，排名从 1 开始，获取数据库 ID。
    3. 首次出现时复制正文与元数据，初始化分数和排名。
    4. 累加 1 / (rrf_k + rank)，记录对应排名。
    5. 按融合分数降序返回全部片段。
    【输出】
    [{id, content, metadata, rrf_score, vector_rank, bm25_rank}]，不修改输入。
    """
    # 1. 字典的 key 是数据库 ID；两路结果分别写入自己的排名字段。
    fused_chunks = {}
    retrieval_lists = [
        ("vector_rank", vector_results),
        ("bm25_rank", bm25_results),
    ]

    # 2. 每一路都从第 1 名开始，使用数据库 ID 识别同一片段。
    for rank_name, results in retrieval_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("RRF 融合需要有效的数据库片段 ID。")

            # 3. 同一 ID 只保存一份片段；复制元数据，保留原始检索结果。
            if chunk_id not in fused_chunks:
                fused_chunks[chunk_id] = {
                    "id": chunk_id,
                    "content": chunk["content"],
                    "metadata": dict(chunk["metadata"]),
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                }

            # 4. 只使用排名加分，不读取 Vector distance 或 BM25 score。
            fused_chunks[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)
            fused_chunks[chunk_id][rank_name] = rank

    # 5. 返回按 RRF 分数降序排列的列表；两路为空时自然返回 []。
    fused_results = sorted(
        fused_chunks.values(),
        key=lambda chunk: chunk["rrf_score"],
        reverse=True,
    )
    return fused_results
