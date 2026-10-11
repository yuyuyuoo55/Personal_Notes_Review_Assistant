"""公开示例直接复用真实预计算向量，不在访客进入时调用 Embedding。"""
import json
import math
import os
from hashlib import sha256
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings

from ..rag.vector_retrieval import EMBEDDING_MODEL, EMBEDDING_DIMENSION
from .runtime import ApiError

ASSETS = Path(__file__).resolve().parents[2] / "deploy" / "example"


class CloudEmbedding(Embeddings):
    """读取示例无需 Key；只有实际导入或问题检索才解析当前会话百炼凭据。"""
    def __init__(self, keys):
        self.keys = keys

    def _model(self):
        key = self.keys.effective("dashscope")
        if not key:
            raise ApiError(422, "not_configured", "请配置百炼服务后再执行向量检索或导入。")
        return DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=key)

    def embed_documents(self, texts):
        return self._model().embed_documents(texts)

    def embed_query(self, text):
        return self._model().embed_query(text)


def load_example(runtime, assets=ASSETS):
    """【业务需求】新访客独立拥有公开示例，首次加载不产生模型费用。
    【输入】只指向当前访客集合与 uploads 的 Runtime、仓库公开资产。
    【执行步骤】核对正文/片段哈希与向量空间；复制 Markdown；直接写入当前集合。
    【输出】六个真实预计算片段；资产无效时拒绝，不回退付费向量化。
    """
    raw = (assets / "ai_application.md").read_bytes()
    # Git/Windows 行尾转换不改变正文与向量的对应关系。
    raw = raw.replace(b"\r\n", b"\n")
    package = json.loads((assets / "embeddings.json").read_text(encoding="utf-8"))
    if package.get("version") != 1 or package.get("model") != EMBEDDING_MODEL or package.get("dimension") != EMBEDDING_DIMENSION or sha256(raw).hexdigest() != package.get("note_sha256"):
        raise ValueError("公开示例资产不匹配。")
    chunks = package["chunks"]
    if len(chunks) != 6:
        raise ValueError("公开示例片段数量不正确。")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for chunk in chunks:
        content, vector = chunk["content"], chunk["embedding"]
        if not content or content not in text or sha256(content.encode()).hexdigest() != chunk["content_sha256"] or len(vector) != EMBEDDING_DIMENSION or any(type(x) not in {int, float} or not math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError("公开示例正文或向量无效。")
    runtime.upload_root.mkdir(parents=True, exist_ok=True)
    path = runtime.upload_root / "AI应用开发公开示例.md"
    source = os.path.normcase(str(path.resolve()))
    ids, metadatas = [], []
    for index, chunk in enumerate(chunks, 1):
        chunk_id = sha256((source + "\n" + str(index) + "\n" + chunk["content"]).encode()).hexdigest()
        ids.append(chunk_id)
        metadatas.append({**chunk["headings"], "source": source, "filename": path.name,
                          "chunk_index": index, "chunk_id": chunk_id, "public_example": True})
    path.write_bytes(raw)
    # 提供 embeddings，绕过 LangChain add_documents 和 Chroma 默认 embedding 函数。
    runtime.store()._collection.upsert(ids=ids, documents=[c["content"] for c in chunks],
                                      metadatas=metadatas, embeddings=[c["embedding"] for c in chunks])
