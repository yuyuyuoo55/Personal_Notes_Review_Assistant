from hashlib import sha256
from typing import Any

from langchain_core.documents import Document

from backend.app.core.logger import get_logger

logger = get_logger(__name__)


def rrf_fusion(
    results_list: list,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    results_list 是一个二维列表：
    results_list[0]：向量检索器返回的结果
    results_list[1]：BM25 检索器返回的结果
    """

    # 1. 创建融合结果字典。
    # key 是 chunk_id，value 是融合后的 Chunk 信息。
    fused_chunks = {}

    # 2. 第一层循环：依次遍历“向量结果列表”和“BM25 结果列表”。
    for result in results_list:
        # 每一路检索结果的排名都要从第 1 名重新开始。
        rank = 1

        # 3. 第二层循环：遍历当前检索器返回的每一个结果。
        for chunk,score in result:
            # 两种检索结果都是：(片段, 分数)，RRF 不使用原始分数。

            # 4. 判断当前片段来自向量检索还是 BM25。
            # 因为
            # 两路返回格式不一样：
            # 向量： (Document, 距离)
            # BM25： (dict, 分数)
            if isinstance(chunk, Document):
                # 向量检索返回 Document。
                content = chunk.page_content
                metadata = chunk.metadata
                # 兼容早期写入 Chroma、尚未带 chunk_id 的旧片段。
                # 新导入的片段始终带 chunk_id；旧片段则根据来源和正文生成稳定兜底 ID。
                chunk_id = metadata.get("chunk_id")
                if not chunk_id:
                    source = str(metadata.get("source", "unknown"))
                    chunk_id = sha256(f"{source}:{content}".encode("utf-8")).hexdigest()[:16]
                rank_name = "vector_rank"

            else:
                # BM25 检索返回字典。
                chunk_id = chunk["id"]
                content = chunk["content"]
                metadata = chunk["metadata"]
                rank_name = "bm25_rank"

            # 5. 如果这个 chunk_id 第一次出现，保存它的基础信息。
            if chunk_id not in fused_chunks:
                fused_chunks[chunk_id] = {
                    "id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                }

            # 6. 按当前排名累加 RRF 分数。
            fused_chunks[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

            # 7. 记录它在当前检索器中的排名。
            fused_chunks[chunk_id][rank_name] = rank

            # 8. 当前结果处理完成，排名加一。
            rank += 1

    # 9. 将字典中的 Chunk 按 RRF 总分从高到低排序。
    rrf_results = sorted(
        fused_chunks.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    logger.info(f"RRF 融合后共有 {len(rrf_results)} 个 Chunk")
    return rrf_results


if __name__ == "__main__":
    # 1. 模拟向量检索结果，格式为：(Document, Chroma 距离)。
    vector_results = [
        (
            Document(
                page_content="A 的正文",
                metadata={"chunk_id": "A", "source": "vector_test.md"},
            ),
            0.10,
        ),
        (
            Document(
                page_content="B 的正文",
                metadata={"chunk_id": "B", "source": "vector_test.md"},
            ),
            0.20,
        ),
        (
            Document(
                page_content="C 的正文",
                metadata={"chunk_id": "C", "source": "vector_test.md"},
            ),
            0.30,
        ),
    ]

    # 2. 模拟 BM25 检索结果，格式为：(dict, BM25 分数)。
    bm25_results = [
        (
            {
                "id": "B",
                "content": "B 的正文",
                "metadata": {"source": "bm25_test.md"},
            },
            8.2,
        ),
        (
            {
                "id": "D",
                "content": "D 的正文",
                "metadata": {"source": "bm25_test.md"},
            },
            6.5,
        ),
        (
            {
                "id": "A",
                "content": "A 的正文",
                "metadata": {"source": "bm25_test.md"},
            },
            4.1,
        ),
    ]

    # 3. 将两路检索器的返回值组成二维列表，传给 RRF。
    results_list = [vector_results, bm25_results]
    rrf_results = rrf_fusion(results_list)

    # 4. 打印融合结果，预期排序为 B、A、D、C。
    for rank, result in enumerate(rrf_results, start=1):
        print(f"\n===== Top {rank} =====")
        print(f"chunk_id：{result['id']}")
        print(f"RRF 分数：{result['rrf_score']:.6f}")
        print(f"向量排名：{result['vector_rank']}")
        print(f"BM25 排名：{result['bm25_rank']}")
        print(f"内容：{result['content']}")
