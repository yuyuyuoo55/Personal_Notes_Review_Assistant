from pathlib import Path
from typing import Any

import bm25s
import pkuseg
from langchain_core.documents import Document

from backend.app.core.config import BM25_INDEX_PATH
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


def bm25_retriever(
    query: str,
    chunks_list: list[Document],
    index_path: str | Path = BM25_INDEX_PATH,
    top_k: int = 5,
    # 导入了新笔记或切分规则变化时传 True，避免加载与当前 Chunk 不一致的旧索引。
    force_rebuild: bool = False,
) -> list[tuple[dict[str, Any], float]]:
    """按关键词检索，返回 [(保存的 Chunk 字典, BM25 分数)]。"""
    # 1. 获取 BM25 索引路径。
    index_file = Path(index_path)

    # 2. 初始化中文切词工具。
    segmenter = pkuseg.pkuseg()

    # 3. 判断本地是否已有索引；有且不要求重建时，直接加载。
    if index_file.exists() and not force_rebuild:
        logger.info(f"BM25 索引已存在，正在加载：{index_file}")
        bm25_index = bm25s.BM25.load(index_file, load_corpus=True)
    else:
        if not chunks_list:
            logger.warning("没有可用于创建 BM25 索引的 Chunk")
            return []

        logger.info("正在创建 BM25 索引")

        # 4. 复制原始 Chunk 的必要信息，组成 BM25 的 corpus。
        #    BM25 只会返回这里保存的 corpus，不会沿 index_path 回读原 Document。
        bm25_corpus = [
            {
                "id": chunk.metadata["chunk_id"],
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks_list
        ]

        # 5. 遍历 corpus，对每个 Chunk 的正文进行切词。
        corpus_tokens = [
            segmenter.cut(item["content"])
            for item in bm25_corpus
        ]

        # 6. 创建 BM25 对象，并保留未切词的 corpus。
        bm25_index = bm25s.BM25(corpus=bm25_corpus)

        # 7. 使用切词后的 corpus_tokens 建立倒排索引。
        bm25_index.index(corpus_tokens)

        # 8. 保存索引和 corpus，下次可直接加载。
        index_file.parent.mkdir(parents=True, exist_ok=True)
        bm25_index.save(index_file)
        logger.info(f"BM25 索引已创建并保存：{index_file}")

    # 9. 对用户问题切词，并包装成 list[list[str]]。
    #    例如两个问题时：[["RRF", "优点"], ["项目", "目标"]]。
    query_tokens = [segmenter.cut(query)]

    # 10. 检索 Top-K，分别得到命中的 corpus 和对应 BM25 分数。
    results, scores = bm25_index.retrieve(query_tokens, k=top_k)

    ranked_results: list[tuple[dict[str, Any], float]] = []

    # 11. 将第一个问题的结果和分数配对，组成 [(字典, 分数)]。
    #     results[0]、scores[0] 表示批量查询中的第一个问题。
    for rank, item in enumerate(results[0]):
        ranked_results.append((dict(item), float(scores[0, rank])))

    logger.info(f"BM25 检索到 {len(ranked_results)} 个片段")
    return ranked_results


if __name__ == "__main__":
    from backend.app.core.config import PROJECT_ROOT
    from backend.app.services.note_loader import load_notes
    from backend.app.services.note_splitter import split_documents

    # 1. 指定一份已有的 Markdown 笔记。
    file_path = PROJECT_ROOT / "task_plan.md"

    # 2. 加载原始文档，并按标题切分为 Chunk。
    docs = load_notes(str(file_path))
    chunks = split_documents(docs)

    # 3. 准备一个适合关键词检索的问题。
    query = "个人笔记复习助手 项目目标"

    # 4. 调用 BM25 检索。
    #    测试索引单独保存，不影响以后正式使用的 data/bm25/notes.bm25。
    results = bm25_retriever(
        query=query,
        chunks_list=chunks,
        index_path=PROJECT_ROOT / "data/bm25/bm25_test.bm25",
        top_k=2,
        force_rebuild=True,
    )

    # 5. 打印检索结果。
    print(f"\n问题：{query}")
    print(f"切分后的 Chunk 数量：{len(chunks)}")
    print(f"BM25 返回结果数量：{len(results)}")

    for rank, (item, score) in enumerate(results, start=1):
        print(f"\n===== Top {rank} =====")
        print(f"chunk_id：{item['id']}")
        print(f"BM25 分数：{score:.4f}")
        print(f"内容：{item['content'][:200]}")
        print(f"元数据：{item['metadata']}")
