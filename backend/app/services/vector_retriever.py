from langchain_chroma import Chroma
from backend.app.core.logger import get_logger
from langchain_core.documents import Document

logger = get_logger(__name__)

def vector_retriever(
    query: str,
    vector_store: Chroma,
    top_k: int,
) -> list[tuple[Document, float]]:
    """使用调用方传入的查询文本，执行纯向量检索。"""

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    if not results:
        logger.warning("没有检索到片段")
        return []

    logger.info(f"检索到片段有：{len(results)}")
    return results

if __name__ == "__main__":
    from backend.app.services.note_loader import load_notes
    from backend.app.services.note_splitter import split_documents
    from backend.app.storage.vector_store import (
        knowledge_to_vector,
        vector_store,
    )

    file_path = (
        r"D:\codex\project\vibe-coding\Personal_Notes_Review_Assistant\task_plan.md"
    )

    docs = load_notes(file_path)
    chunks = split_documents(docs)

    current_count = vector_store._collection.count()

    if current_count == 0:
        knowledge_to_vector(chunks)
        print(f"首次写入 {len(chunks)} 个 Chunk")
    else:
        print(f"Chroma 已有 {current_count} 个 Chunk，直接测试检索")

    query = "个人笔记复习助手的项目目标是什么？"

    results = vector_retriever(
        query=query,
        vector_store=vector_store,
        top_k=2,
    )

    print(f"\n问题：{query}")
    print(f"检索到 {len(results)} 个结果")

    for index, (doc, distance) in enumerate(results, start=1):
        print(f"\n===== Top {index} =====")
        print(f"距离：{distance:.4f}")
        print(f"内容：{doc.page_content[:200]}")
        print(f"元数据：{doc.metadata}")
