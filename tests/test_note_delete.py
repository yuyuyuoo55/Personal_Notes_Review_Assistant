from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.api import notes as notes_api
from backend.app.main import app
from backend.app.services import note_delete_service
from backend.app.services.image_chunk_store import manifest_path


client = TestClient(app)
TEST_KEY = "test-delete-key"


class FakeVectorStore:
    def __init__(self, ids=None):
        self.where_calls = []
        self.deleted_ids = []
        self._ids = ids or []
        self._collection = SimpleNamespace(delete=self.delete_where)

    def delete_where(self, *, where):
        self.where_calls.append({"where": where})

    def get(self, include=None):
        return {"ids": self._ids}

    def delete(self, ids):
        self.deleted_ids.extend(ids)


def test_delete_markdown_removes_file_manifest_vectors_and_cache(monkeypatch, tmp_path):
    note = tmp_path / "课程笔记.md"
    note.write_text("# 测试", encoding="utf-8")
    manifest_path(note).write_text("[]", encoding="utf-8")
    image_dir = tmp_path / note.stem
    image_dir.mkdir()
    (image_dir / "saved.png").write_bytes(b"image")
    fake_store = FakeVectorStore()
    invalidated = []
    monkeypatch.setattr(note_delete_service, "vector_store", fake_store)
    monkeypatch.setattr(note_delete_service, "invalidate_rag_cache", lambda: invalidated.append(True))

    note_delete_service.delete_note(note.name, "md", tmp_path)

    assert fake_store.where_calls == [{"where": {"source": str(note)}}]
    assert not note.exists()
    assert not manifest_path(note).exists()
    assert not image_dir.exists()
    assert invalidated == [True]


def test_delete_image_removes_only_matching_doc_directory(monkeypatch, tmp_path):
    target = tmp_path / "diagram-1234"
    other = tmp_path / "other-5678"
    target.mkdir()
    other.mkdir()
    fake_store = FakeVectorStore()
    monkeypatch.setattr(note_delete_service, "vector_store", fake_store)
    monkeypatch.setattr(note_delete_service, "invalidate_rag_cache", lambda: None)

    note_delete_service.delete_note(target.name, "image", tmp_path)

    assert fake_store.where_calls == [{"where": {"doc_id": target.name}}]
    assert not target.exists()
    assert other.exists()


def test_delete_all_keeps_upload_root_and_clears_vectors(monkeypatch, tmp_path):
    (tmp_path / "note.md").write_text("# note", encoding="utf-8")
    (tmp_path / "doc-id").mkdir()
    fake_store = FakeVectorStore(ids=["chunk-1", "chunk-2"])
    monkeypatch.setattr(note_delete_service, "vector_store", fake_store)
    monkeypatch.setattr(note_delete_service, "invalidate_rag_cache", lambda: None)

    note_delete_service.delete_all_notes(tmp_path)

    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []
    assert fake_store.deleted_ids == ["chunk-1", "chunk-2"]


def test_delete_endpoints_require_key_and_dispatch(monkeypatch):
    assert client.delete("/api/notes/md:note.md").status_code == 401
    assert client.delete("/api/notes").status_code == 401

    calls = []
    monkeypatch.setattr(notes_api, "delete_note", lambda identifier, kind: calls.append((identifier, kind)))
    monkeypatch.setattr(notes_api, "delete_all_notes", lambda: calls.append(("all", None)))

    response = client.delete(
        "/api/notes/md:note.md",
        headers={"X-DeepSeek-API-Key": TEST_KEY},
    )
    assert response.status_code == 204
    response = client.delete(
        "/api/notes",
        headers={"X-DeepSeek-API-Key": TEST_KEY},
    )
    assert response.status_code == 204
    assert calls == [("note.md", "md"), ("all", None)]


def test_delete_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(note_delete_service, "vector_store", FakeVectorStore())
    try:
        note_delete_service.delete_note("../outside.md", "md", tmp_path)
        raise AssertionError("路径穿越标识未被拒绝")
    except ValueError as error:
        assert "标识无效" in str(error)
