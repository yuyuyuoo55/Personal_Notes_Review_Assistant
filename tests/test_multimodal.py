import asyncio
import base64
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import markdown_image_service
from backend.app.services.multimodal_service import ImageProcessingError, InvalidApiKeyError
from backend.app.services.rag_service import create_chat_model


client = TestClient(app)
TEST_KEY = "test-user-key-never-log"


def test_chat_and_import_require_user_api_key():
    chat_response = client.post(
        "/api/chat",
        json={"query": "测试", "mode": "fast", "conversation_id": "test-session"},
    )
    assert chat_response.status_code == 401
    assert chat_response.json()["detail"] == "请先输入API Key"

    import_response = client.post(
        "/api/notes/import",
        files={"file": ("note.md", b"# test", "text/markdown")},
    )
    assert import_response.status_code == 401


def test_image_question_description_participates_in_rag(monkeypatch, caplog):
    async def fake_describe(image_url, api_key):
        assert image_url.startswith("data:image/png;base64,")
        assert api_key == TEST_KEY
        return "会议时间为周五下午三点"

    monkeypatch.setattr(
        "backend.app.api.chat.describe_image_url",
        fake_describe,
    )
    def fake_prepare(**kwargs):
        assert "会议时间为周五下午三点" in kwargs["original_query"]
        assert "这张图片讲了什么" in kwargs["original_query"]
        return SimpleNamespace(
            rewritten_query=kwargs["original_query"],
            sources=[],
            answer_stream=iter([SimpleNamespace(event="token", content="命中笔记内容", sources=[])]),
        )
    monkeypatch.setattr("backend.app.api.chat.prepare_rag_answer", fake_prepare)
    response = client.post(
        "/api/chat/image",
        headers={"X-DeepSeek-API-Key": TEST_KEY},
        data={"query": "这张图片讲了什么"},
        files={"image": ("meeting.png", b"\x89PNG\r\n\x1a\nmock", "image/png")},
    )

    assert response.status_code == 200
    assert "命中笔记内容" in response.text
    assert "event: done" in response.text
    assert TEST_KEY not in caplog.text


def test_invalid_image_type_degrades_to_chinese_message():
    response = client.post(
        "/api/chat/image",
        headers={"X-DeepSeek-API-Key": TEST_KEY},
        data={"query": "这张图片讲了什么"},
        files={"image": ("note.txt", b"not-image", "text/plain")},
    )
    assert response.status_code == 200
    assert "仅支持 JPEG、PNG、GIF 或 WebP 图片" in response.text
    assert "event: done" in response.text


def test_invalid_text_api_key_degrades_to_chinese_sse(monkeypatch):
    class AuthenticationError(Exception):
        pass

    def reject_request(**kwargs):
        raise AuthenticationError("do not expose provider details")

    monkeypatch.setattr("backend.app.api.chat.prepare_rag_answer", reject_request)
    response = client.post(
        "/api/chat",
        headers={"X-DeepSeek-API-Key": "invalid-test-key"},
        json={"query": "测试", "mode": "fast", "conversation_id": "test-session"},
    )
    assert response.status_code == 200
    assert "API Key无效，请检查后重试" in response.text
    assert "provider details" not in response.text
    assert "event: done" in response.text


def test_markdown_image_description_is_inserted(monkeypatch, tmp_path):
    tiny_image = base64.b64encode(b"\x89PNG\r\n\x1a\nmock").decode("ascii")
    markdown = f"# 会议通知\n\n![通知](data:image/png;base64,{tiny_image})\n"

    async def fake_describe(image_url, api_key):
        assert image_url.startswith("data:image/png;base64,")
        assert api_key == TEST_KEY
        return "图片写着会议时间为周五下午三点"

    monkeypatch.setattr(markdown_image_service, "describe_image_url", fake_describe)
    result = asyncio.run(markdown_image_service.enrich_markdown_images(
        markdown, TEST_KEY, source_path=str(tmp_path / "note.md"), doc_dir=tmp_path
    ))

    assert result.image_total == 1
    assert result.image_processed == 1
    assert result.image_skipped == 0
    assert "> 图片内容：图片写着会议时间为周五下午三点" in result.markdown
    assert len(result.image_chunks) == 1
    assert result.image_chunks[0].metadata["is_image_chunk"] is True
    assert result.image_chunks[0].metadata["image_path"]


def test_markdown_local_save_failure_skips_only_image(monkeypatch, tmp_path):
    tiny_image = base64.b64encode(b"\x89PNG\r\n\x1a\nmock").decode("ascii")
    markdown = f"# 正文\n\n正文仍需保留。\n\n![图](data:image/png;base64,{tiny_image})"

    def fail_save(*args):
        raise ImageProcessingError("本地保存失败，图片已跳过")

    monkeypatch.setattr(markdown_image_service, "save_image_to_local", fail_save)
    result = asyncio.run(markdown_image_service.enrich_markdown_images(
        markdown, TEST_KEY, source_path=str(tmp_path / "note.md"), doc_dir=tmp_path
    ))

    assert result.image_processed == 0
    assert result.image_skipped == 1
    assert "正文仍需保留" in result.markdown
    assert "本地保存失败" in result.warnings[0]


def test_different_keys_do_not_share_chat_model_instances():
    first = create_chat_model("first-test-key")
    second = create_chat_model("second-test-key")
    assert first is not second


def test_key_validate_endpoint_reports_valid_and_invalid(monkeypatch):
    async def accept_key(api_key):
        assert api_key == TEST_KEY

    monkeypatch.setattr("backend.app.api.key_validate.validate_deepseek_api_key", accept_key)
    response = client.post("/api/key/validate", headers={"X-DeepSeek-API-Key": TEST_KEY})
    assert response.status_code == 200
    assert response.json()["valid"] is True

    async def reject_key(api_key):
        raise InvalidApiKeyError("provider detail")

    monkeypatch.setattr("backend.app.api.key_validate.validate_deepseek_api_key", reject_key)
    response = client.post("/api/key/validate", headers={"X-DeepSeek-API-Key": "invalid"})
    assert response.json() == {"valid": False, "message": "API Key 无效，请检查后重试"}
    assert "provider detail" not in response.text


def test_invalid_api_key_aborts_markdown_import(monkeypatch, tmp_path):
    """Key 无效时必须中止整篇导入并抛出明确错误，而不是默默跳过图片。"""
    tiny_image = base64.b64encode(b"\x89PNG\r\n\x1a\nmock").decode("ascii")
    markdown = f"# 会议通知\n\n![通知](data:image/png;base64,{tiny_image})\n"

    async def reject_describe(image_url, api_key):
        raise InvalidApiKeyError("API Key无效，图片已跳过")

    monkeypatch.setattr(markdown_image_service, "describe_image_url", reject_describe)
    try:
        asyncio.run(markdown_image_service.enrich_markdown_images(
            markdown, TEST_KEY, source_path=str(tmp_path / "note.md"), doc_dir=tmp_path
        ))
        raise AssertionError("InvalidApiKeyError 未被抛出")
    except InvalidApiKeyError as error:
        assert "API Key 无效" in str(error)
        assert "导入已中止" in str(error)
