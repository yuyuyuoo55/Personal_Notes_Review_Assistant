import asyncio
import io
from zipfile import ZipFile

from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api import notes as notes_api
from backend.app.services import markdown_image_service
from backend.app.services.markdown_bundle_service import read_markdown_bundle


client = TestClient(app)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 12), "white").save(output, format="PNG")
    return output.getvalue()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_bundle_matches_relative_and_absolute_markdown_image_paths(monkeypatch, tmp_path):
    markdown = (
        "# 图文笔记\n\n"
        "![相对路径](assets/example.png)\n\n"
        "![绝对路径](C:\\Users\\demo\\Pictures\\example.png)\n"
    )
    bundle = read_markdown_bundle(_zip_bytes({
        "day01/note.md": markdown.encode("utf-8"),
        "day01/assets/example.png": _png_bytes(),
    }))

    async def fake_describe(image_url, api_key):
        assert image_url.startswith("data:image/png;base64,")
        return "图片中的知识点"

    monkeypatch.setattr(markdown_image_service, "describe_image_url", fake_describe)
    result = asyncio.run(markdown_image_service.enrich_markdown_images(
        bundle.markdown,
        "test-key",
        source_path=str(tmp_path / bundle.markdown_name),
        doc_dir=tmp_path / "note",
        local_images=bundle.local_images,
    ))

    assert bundle.markdown_name == "note.md"
    assert result.image_total == 2
    assert result.image_processed == 2
    assert result.image_skipped == 0
    assert len(result.image_chunks) == 2


def test_bundle_rejects_unsafe_member_path():
    content = _zip_bytes({"note.md": b"# note", "../secret.png": _png_bytes()})
    with pytest.raises(ValueError, match="不安全"):
        read_markdown_bundle(content)


def test_zip_bundle_can_be_imported_through_notes_api(monkeypatch, tmp_path):
    markdown = "# HTTP\n\n![请求图](images/http.png)\n"
    content = _zip_bytes({
        "HTTP.md": markdown.encode("utf-8"),
        "images/http.png": _png_bytes(),
    })

    async def fake_describe(image_url, api_key):
        return "HTTP 请求结构图"

    monkeypatch.setattr(notes_api, "UPLOAD_DIRECTORY", tmp_path)
    monkeypatch.setattr(markdown_image_service, "describe_image_url", fake_describe)
    monkeypatch.setattr(notes_api, "knowledge_to_vector", lambda chunks: True)
    monkeypatch.setattr(notes_api, "invalidate_rag_cache", lambda: None)

    response = client.post(
        "/api/notes/import",
        headers={"X-DeepSeek-API-Key": "test-key"},
        files={"file": ("HTTP.zip", content, "application/zip")},
    )

    assert response.status_code == 201
    assert response.json()["file_name"] == "HTTP.md"
    assert response.json()["image_processed"] == 1
    assert response.json()["image_skipped"] == 0
    assert (tmp_path / "HTTP.md").exists()


def test_markdown_image_chunk_id_is_stable_across_repeated_processing(monkeypatch, tmp_path):
    markdown = "# HTTP\n\n![请求图](images/http.png)\n"
    bundle = read_markdown_bundle(_zip_bytes({
        "HTTP.md": markdown.encode("utf-8"),
        "images/http.png": _png_bytes(),
    }))

    async def fake_describe(image_url, api_key):
        return "HTTP 请求结构图"

    monkeypatch.setattr(markdown_image_service, "describe_image_url", fake_describe)
    first = asyncio.run(markdown_image_service.enrich_markdown_images(
        bundle.markdown, "test-key", source_path=str(tmp_path / "HTTP.md"),
        doc_dir=tmp_path / "first", local_images=bundle.local_images,
    ))
    second = asyncio.run(markdown_image_service.enrich_markdown_images(
        bundle.markdown, "test-key", source_path=str(tmp_path / "HTTP.md"),
        doc_dir=tmp_path / "second", local_images=bundle.local_images,
    ))

    assert first.image_chunks[0].metadata["image_path"] != second.image_chunks[0].metadata["image_path"]
    assert first.image_chunks[0].metadata["chunk_id"] == second.image_chunks[0].metadata["chunk_id"]
