"""Phase17A 离线回归：真实图片文件/临时 Chroma，Fake Vision 与离线 Embedding。"""

import base64
import hashlib
import json
import os
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from PIL import Image
from pydantic import ValidationError
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.integrations.vision import VisionContent, describe_image
from app.knowledge.image_import import import_image, import_markdown_images, resolve_image_path
from app.knowledge.note_import import import_markdown
from app.quiz.chapter import list_notes, list_chapters, read_chapter
from app.quiz.service import generate_quiz
from app.quiz.store import QuizStore
from app.rag.vector_retrieval import retrieve_vectors
from app.rag.bm25_retrieval import retrieve_bm25
from app.rag.rrf_fusion import rrf_fusion
from smoke_quiz import FakeModel, generation, review


class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.writes = 0

    def get(self, where=None, include=None):
        rows = [(key, doc) for key, doc in self.rows.items()
                if not where or doc.metadata.get("source") == where["source"]]
        return {"ids": [key for key, _ in rows], "documents": [doc.page_content for _, doc in rows],
                "metadatas": [dict(doc.metadata) for _, doc in rows]}

    def add_documents(self, documents, ids):
        self.writes += 1
        self.rows.update({key: deepcopy(doc) for key, doc in zip(ids, documents)})

    def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)


class FakeVision:
    def __init__(self, content="RRF 是倒数排名融合，使用排名融合检索结果。", fail_at=None):
        self.calls = []
        self.content = content
        self.fail_at = fail_at

    def __call__(self, image_bytes, mime):
        assert image_bytes.startswith((b"\x89PNG", b"\xff\xd8", b"RIFF"))
        self.calls.append((image_bytes, mime))
        if self.fail_at == len(self.calls):
            raise RuntimeError("private-key-must-not-leak")
        return {"confirmed_content": self.content, "uncertain_details": ["右下角模糊信息需核对"]}


class OfflineEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count(word)) for word in ["RRF", "排名", "融合", "Git", "状态", "日常"]]


class ImageImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="v2-image-", ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.note = self.root / "notes.md"
        self.image = self.root / "diagram.png"
        Image.new("RGB", (32, 32), color="white").save(self.image)
        self.store = MemoryStore()
        self.vision = FakeVision()
        self.note.write_text("# 检索\n## RRF\n文字说明。\n![结构图](diagram.png)\n## Git\nGit 状态。", encoding="utf-8")

    def run_import(self, **kwargs):
        return import_markdown_images(self.note, self.store, vision=self.vision, **kwargs)

    def test_relative_path_chapter_source_uncertainty(self):
        original = self.note.read_bytes()
        result = self.run_import()
        self.assertEqual(result["status"], "imported")
        chapter = read_chapter(self.store, result["source"], {"Header 1": "检索", "Header 2": "RRF"})
        self.assertEqual(chapter.status, "ok")
        self.assertIn("倒数排名融合", chapter.content)
        self.assertNotIn("右下角模糊", chapter.content)
        images = [doc for doc in self.store.rows.values() if doc.metadata.get("is_image_chunk")]
        self.assertEqual(images[0].metadata["image_path"], str(self.image))
        self.assertEqual(images[0].metadata["filename"], "notes.md")
        self.assertEqual(images[0].metadata["image_reference"], "diagram.png")
        self.assertEqual(self.note.read_bytes(), original)
        self.assertEqual([doc.metadata["chunk_index"] for doc in self.store.rows.values()], list(range(1, len(self.store.rows) + 1)))

    def test_duplicate_uses_cache_no_embedding(self):
        self.run_import()
        old_ids = set(self.store.rows)
        writes = self.store.writes
        result = self.run_import()
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["vision_calls"], 0)
        self.assertEqual(len(self.vision.calls), 1)
        self.assertEqual(self.store.writes, writes)
        self.assertEqual(set(self.store.rows), old_ids)

    def test_image_update_and_removal_cleanup(self):
        self.run_import()
        old_image_ids = {i for i, d in self.store.rows.items() if d.metadata.get("is_image_chunk")}
        Image.new("RGB", (32, 32), "red").save(self.image)
        self.vision.content = "RRF 根据不同检索列表的排名融合结果。"
        self.assertEqual(self.run_import()["status"], "updated")
        self.assertTrue(old_image_ids.isdisjoint(self.store.rows))
        self.assertEqual(len(self.vision.calls), 2)
        self.note.write_text("# 检索\n## RRF\n现在只保留文字。", encoding="utf-8")
        result = self.run_import()
        self.assertEqual(result["status"], "updated")
        self.assertFalse(any(d.metadata.get("is_image_chunk") for d in self.store.rows.values()))
        self.assertEqual(read_chapter(self.store, result["source"], {"Header 1": "检索"}).status, "ok")

    def test_delete_one_image_preserve_other_and_contiguous_order(self):
        second = self.root / "second.png"
        Image.new("RGB", (32, 32), "blue").save(second)
        self.note.write_text("# 检索\n![一](diagram.png)\n![二](second.png)", encoding="utf-8")
        self.run_import()
        self.note.write_text("# 检索\n![二](second.png)", encoding="utf-8")
        result = self.run_import()
        self.assertEqual(result["vision_calls"], 0)
        images = [d.metadata["image_path"] for d in self.store.rows.values() if d.metadata.get("is_image_chunk")]
        self.assertEqual(images, [str(second)])
        self.assertEqual(read_chapter(self.store, result["source"], {"Header 1": "检索"}).status, "ok")

    def test_external_path_requires_explicit_root(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as external:
            picture = Path(external) / "external.png"
            Image.new("RGB", (32, 32)).save(picture)
            self.note.write_text(f"# 外部\n![图](<{picture}>)", encoding="utf-8")
            self.assertEqual(self.run_import()["status"], "failed")
            self.assertEqual(self.vision.calls, [])
            self.assertEqual(self.run_import(allowed_image_roots=[external])["status"], "imported")

    def test_traversal_urls_and_unc_rejected_without_api(self):
        for reference in ["../outside.png", "%2e%2e/outside.png", "https://example.com/a.png", "file:///secret.png", "data:image/png;base64,xx", "\\\\server\\share\\a.png"]:
            with self.subTest(reference=reference):
                self.note.write_text(f"# 检索\n![图]({reference})", encoding="utf-8")
                self.assertEqual(self.run_import()["status"], "failed")
        self.assertEqual(self.vision.calls, [])
        self.assertEqual(self.store.rows, {})

    def test_symlink_escape_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as external:
            outside = Path(external) / "outside.png"
            Image.new("RGB", (32, 32)).save(outside)
            link = self.root / "link.png"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("当前 Windows 账户没有创建符号链接权限")
            with self.assertRaises(ValueError):
                resolve_image_path("link.png", self.note)

    def test_missing_image_preserves_old_whole_note(self):
        self.run_import()
        old = deepcopy(self.store.rows)
        self.note.write_text("# 更新\n新文字\n![图](missing.png)", encoding="utf-8")
        result = self.run_import()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["warnings"])
        self.assertEqual(self.store.rows, old)

    def test_bad_format_corruption_size_pixels_and_count(self):
        self.image.write_bytes(b"not-an-image")
        self.assertEqual(self.run_import()["status"], "failed")
        Image.new("RGB", (32, 32)).save(self.image)
        with patch("app.knowledge.image_import.MAX_IMAGE_BYTES", 8):
            self.assertEqual(self.run_import()["status"], "failed")
        with patch("app.knowledge.image_import.MAX_IMAGE_PIXELS", 10):
            self.assertEqual(self.run_import()["status"], "failed")
        self.note.write_text("# 大量\n" + "![图](diagram.png)\n" * 21, encoding="utf-8")
        self.assertEqual(self.run_import()["status"], "failed")
        self.assertEqual(self.vision.calls, [])

    def test_vision_failure_and_invalid_output_no_write(self):
        self.run_import()
        old = deepcopy(self.store.rows)
        Image.new("RGB", (32, 32), "green").save(self.image)
        for fake in [FakeVision(fail_at=1), lambda *_: {"confirmed_content": "", "uncertain_details": []}, lambda *_: {"invented": "unknown"}]:
            result = import_markdown_images(self.note, self.store, vision=fake)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(self.store.rows, old)
            self.assertNotIn("private-key", str(result))

    def test_markdown_limits_encoding_and_image_total_limit(self):
        with patch("app.knowledge.image_import.MAX_TOTAL_IMAGE_BYTES", 8):
            self.assertEqual(self.run_import()["status"], "failed")
        with patch("app.knowledge.image_import.MAX_MARKDOWN_BYTES", 8):
            self.assertEqual(self.run_import()["status"], "failed")
        self.note.write_bytes(b'\xff\xfe\xfa')
        self.assertEqual(self.run_import()["status"], "failed")
        self.note.write_text("", encoding="utf-8")
        self.assertEqual(self.run_import()["status"], "failed")
        self.assertEqual(self.vision.calls, [])

    def test_one_failed_image_does_not_write_partial_note(self):
        import_markdown(self.note, self.store)
        old = deepcopy(self.store.rows)
        second = self.root / "second.png"
        Image.new("RGB", (32, 32), "blue").save(second)
        self.note.write_text("# 更新\n![一](diagram.png)\n![二](second.png)", encoding="utf-8")
        result = import_markdown_images(self.note, self.store, vision=FakeVision(fail_at=2))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["image_processed"], 1)
        self.assertEqual(result["vision_calls"], 2)
        self.assertEqual(self.store.rows, old)

    def test_standalone_formats_chapter_and_update(self):
        for suffix, fmt in [(".png", "PNG"), (".jpg", "JPEG"), (".jpeg", "JPEG"), (".webp", "WEBP")]:
            with self.subTest(suffix=suffix):
                path = self.root / ("standalone" + suffix)
                Image.new("RGB", (32, 32)).save(path, format=fmt)
                result = import_image(path, self.store, vision=self.vision)
                self.assertEqual(result["status"], "imported")
                self.assertEqual(read_chapter(self.store, result["source"], {"Header 1": path.name}).status, "ok")
                self.assertEqual(import_image(path, self.store, vision=self.vision)["status"], "unchanged")
        self.assertEqual(len(list_notes(self.store)), 4)
        self.assertEqual(import_image(self.root / "missing.png", self.store, vision=self.vision)["status"], "failed")

    def test_code_fences_inline_code_and_caption_title(self):
        self.note.write_text('# 示例\n```md\n![示例](missing.png)\n```\n`![示例](missing.png)`\n![实际](diagram.png "标题")', encoding="utf-8")
        result = self.run_import()
        self.assertEqual(result["image_total"], 1)
        self.assertEqual(result["status"], "imported")

    def test_pure_markdown_legacy_ids_unchanged(self):
        self.note.write_text("# 纯文本\n## 概念\nGit 状态\n", encoding="utf-8-sig")
        result = self.run_import()
        path = self.note.resolve()
        sections = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")], strip_headers=False).split_text(path.read_text(encoding="utf-8-sig"))
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(sections)
        expected = []
        for i, chunk in enumerate(chunks, start=1):
            chunk.metadata.update(source=os.path.normcase(str(path)), filename=path.name, chunk_index=i)
            expected.append(hashlib.sha256((json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True) + "\n" + chunk.page_content).encode()).hexdigest())
        self.assertEqual(list(self.store.rows), expected)
        self.assertEqual(import_markdown(path, self.store)["status"], "unchanged")
        self.assertEqual(self.vision.calls, [])

    def test_storage_failure_preserves_old_records(self):
        self.run_import()
        old = deepcopy(self.store.rows)
        self.note.write_text(self.note.read_text(encoding="utf-8") + "\n新内容", encoding="utf-8")
        with patch.object(self.store, "add_documents", side_effect=RuntimeError("private-store-error")):
            result = self.run_import()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.store.rows, old)

    def test_http_payload_real_bytes_and_json_validation(self):
        image_bytes = self.image.read_bytes()
        def handler(request):
            payload = json.loads(request.content)
            blocks = payload["messages"][1]["content"]
            url = blocks[1]["image_url"]["url"]
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), image_bytes)
            self.assertNotIn(str(self.image), json.dumps(payload))
            self.assertEqual(payload["model"], "configured-vision")
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"confirmed_content": "文字", "uncertain_details": []})}}]})
        with patch.dict(os.environ, {"DEEPSEEK_VISION_MODEL": "configured-vision"}):
            with httpx.Client(base_url="https://offline.test/", transport=httpx.MockTransport(handler)) as client:
                self.assertEqual(describe_image(image_bytes, "image/png", client=client).confirmed_content, "文字")
        for status, body in [(401, {}), (200, {"choices": [{"message": {"content": "bad-json"}}]})]:
            with httpx.Client(base_url="https://offline.test/", transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body))) as client:
                with self.assertRaises(ValueError):
                    describe_image(image_bytes, "image/png", client=client)

    def saved_vision_response(self):
        # 公开夹具只保留通用流程标签和协议字段，不依赖本机验收元数据。
        evidence = Path(__file__).resolve().parent / "fixtures" / "vision_response.json"
        return json.loads(evidence.read_text(encoding="utf-8"))["vision_content"]

    def test_saved_real_response_replay_and_strict_validation(self):
        raw = json.loads(self.saved_vision_response())
        original = deepcopy(raw)
        parsed = VisionContent.model_validate(raw)
        self.assertEqual(parsed.confirmed_content.splitlines(), ["需求分析", "设计", "编码", "测试", "上线运维"])
        self.assertEqual(parsed.uncertain_details, [])
        self.assertEqual(raw, original)
        self.assertEqual(VisionContent.model_validate({**raw, "uncertain_details": ["箭头不清楚"]}).uncertain_details, ["箭头不清楚"])
        for changes in [{"type": "other"}, {"type": None}, {"type": 1}, {"unexpected": True},
                        {"confirmed_content": 123}, {"confirmed_content": None},
                        {"uncertain_details": "wrong"}, {"uncertain_details": [123]},
                        {"uncertain_details": ("tuple",)}]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                VisionContent.model_validate({**raw, **changes})
        with httpx.Client(base_url="https://offline.test/", transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({**raw, "type": "bad"})}}]}))) as client:
            with self.assertRaises(ValidationError):
                describe_image(self.image.read_bytes(), "image/png", client=client)

    def test_saved_real_response_import_and_reopen_mock_embedding(self):
        # 回放已保存响应的脱敏正文；不重新识别图片、不调用任何供应商。
        response = self.saved_vision_response()
        store = Chroma(collection_name="replay_test", persist_directory=str(self.root / "replay"), embedding_function=OfflineEmbedding())
        with httpx.Client(base_url="https://offline.test/", transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"choices": [{"message": {"content": response}}]}))) as client:
            result = import_image(self.image, store, vision=lambda data, mime: describe_image(data, mime, client=client))
        self.assertEqual(result["status"], "imported")
        self.assertEqual(result["chunk_count"], 1)
        reopened = Chroma(collection_name="replay_test", persist_directory=str(self.root / "replay"), embedding_function=None, create_collection_if_not_exists=False)
        data = reopened.get(include=["documents", "metadatas", "embeddings"])
        self.assertEqual(len(data["ids"]), 1)
        self.assertEqual(len(data["embeddings"][0]), 6)
        metadata = data["metadatas"][0]
        self.assertEqual(metadata["image_path"], str(self.image))
        self.assertEqual(metadata["source"], result["source"])
        self.assertEqual(json.loads(metadata["uncertain_details"]), [])
        self.assertTrue(metadata["is_image_chunk"])
        chapter = read_chapter(reopened, result["source"], {"Header 1": self.image.name})
        self.assertEqual(chapter.status, "ok")
        for step in ["需求分析", "设计", "编码", "测试", "上线运维"]:
            self.assertIn(step, chapter.content)

    def test_real_chroma_reopen_retrieval_and_quiz(self):
        database = str(self.root / "chroma")
        store = Chroma(collection_name="image_test", persist_directory=database, embedding_function=OfflineEmbedding())
        result = import_markdown_images(self.note, store, vision=self.vision)
        self.assertEqual(result["status"], "imported")
        reopened = Chroma(collection_name="image_test", persist_directory=database, embedding_function=OfflineEmbedding(), create_collection_if_not_exists=False)
        headings = {"Header 1": "检索", "Header 2": "RRF"}
        chapter = read_chapter(reopened, result["source"], headings)
        self.assertEqual(chapter.status, "ok")
        self.assertIn(headings, list_chapters(reopened, result["source"]))
        vector = retrieve_vectors("RRF 排名融合", reopened, 3)
        with patch("app.rag.bm25_retrieval.get_segmenter") as segmenter:
            segmenter.return_value.cut.side_effect = lambda text: re.findall(r"RRF|排名|融合|Git|状态|日常|文字", text)
            keyword = retrieve_bm25("RRF 排名融合", reopened, 3)
        self.assertTrue(any(item["metadata"].get("is_image_chunk") for item in vector))
        self.assertTrue(any(item["metadata"].get("is_image_chunk") for item in keyword))
        self.assertTrue(rrf_fusion(vector, keyword))
        image_chunk = next(c for c in chapter.chunks if "倒数排名融合" in c.content)
        question = {"question_type": "single_choice", "question": "图中的 RRF 是什么？", "knowledge_point": "图片RRF定义",
                    "source_chunk_ids": [image_chunk.chunk_id], "evidence": [{"chunk_id": image_chunk.chunk_id, "quote": "RRF 是倒数排名融合"}],
                    "options": {"A": "倒数排名融合", "B": "Git状态", "C": "暂存区", "D": "版本库"}, "correct_answer": "A", "explanation": "测试预设",
                    "reference_answer": "", "key_points": []}
        public = generate_quiz(reopened, result["source"], headings, 1, 0, model=FakeModel([generation([question]), review([True])]), quiz_store=QuizStore())
        self.assertEqual(public.status, "ready")
        self.assertNotIn("correct_answer", public.questions[0].model_dump())


if __name__ == "__main__":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    unittest.main(verbosity=2)
