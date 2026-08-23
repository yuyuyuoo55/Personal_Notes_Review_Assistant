from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from backend.app.core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_EMBEDDING_MODEL,
)
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# 1. 初始化 Embedding 模型：负责把文本转换成向量。
embedding = DashScopeEmbeddings(
    dashscope_api_key=DASHSCOPE_API_KEY,
    model=DASHSCOPE_EMBEDDING_MODEL,
)

# 2. 初始化 Chroma 向量库，并绑定刚才的 Embedding 模型。
vector_store = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=str(CHROMA_PERSIST_DIRECTORY),
    embedding_function=embedding,
)


def knowledge_to_vector(docs: list[Document]) -> bool:
    """将已切分的 Chunk 写入 Chroma；Embedding 由 Chroma 自动调用。"""
    if not docs:
        logger.warning("没有可写入向量库的文档")
        return False

    # 3. 写入完整 Document；Chroma 会自动向量化正文，并保存 metadata。
    vector_store.add_documents(docs)
    logger.info(f"已写入 {len(docs)} 个 Chunk 到 Chroma")
    return True


if __name__ == "__main__":
    from backend.app.services.note_loader import load_notes
    from backend.app.services.note_splitter import split_documents

    file_path = (
        r"D:\codex\project\vibe-coding\Personal_Notes_Review_Assistant\task_plan.md"
    )

    # 1. 加载原始 Markdown 文档。
    docs = load_notes(file_path)

    # 2. 按 Markdown 标题切分为多个 Chunk。
    chunks = split_documents(docs)

    # 3. 将 Chunk 写入 Chroma。
    success = knowledge_to_vector(chunks)

    # 4. 读取当前向量库的记录数，验证是否写入成功。
    count = vector_store._collection.count()

    print(f"本次切分 Chunk 数量：{len(chunks)}")
    print(f"写入是否成功：{success}")
    print(f"Chroma 当前总记录数：{count}")
