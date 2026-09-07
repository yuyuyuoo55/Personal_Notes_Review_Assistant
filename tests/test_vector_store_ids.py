from langchain_core.documents import Document

from backend.app.storage import vector_store as vector_store_module


def test_knowledge_to_vector_uses_stable_chunk_ids(monkeypatch):
    calls = []

    class FakeVectorStore:
        def add_documents(self, docs, **kwargs):
            calls.append((docs, kwargs))

    monkeypatch.setattr(vector_store_module, "vector_store", FakeVectorStore())
    docs = [
        Document(page_content="A", metadata={"chunk_id": "chunk-a"}),
        Document(page_content="B", metadata={"chunk_id": "chunk-b"}),
    ]

    assert vector_store_module.knowledge_to_vector(docs) is True
    assert calls[0][1]["ids"] == ["chunk-a", "chunk-b"]
