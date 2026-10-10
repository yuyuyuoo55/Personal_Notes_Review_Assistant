"""Phase 16A 离线验收：Fake Model + 临时真实 Chroma；不调用付费 API。"""

import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException

from app.knowledge.note_import import import_markdown
from app.quiz.chapter import list_chapters, list_notes, normalize_source, read_chapter
from app.quiz.schemas import GeneratedQuiz, QualityReview, QuestionCounts
from app.quiz.service import generate_quiz
from app.quiz.store import QuizStore
from app.rag.vector_retrieval import create_vector_store


SOURCE = normalize_source("quiz-test-notes.md")
HEADINGS = {"Header 1": "检索", "Header 2": "RRF"}
BODY = "RRF 是倒数排名融合。RRF 使用排名，不需要统一不同分数尺度。"


class FakeStore:
    def __init__(self, records=None):
        self.records = records if records is not None else [
            ("chunk-1", BODY, {"source": SOURCE, "filename": "quiz-test-notes.md",
                              "chunk_index": 1, "chunk_id": "chunk-1", **HEADINGS})]
        self.calls = []

    def get(self, where=None, include=None):
        self.calls.append((where, include))
        rows = [row for row in self.records if where is None or row[2].get("source") == where["source"]]
        return {"ids": [row[0] for row in rows], "documents": [row[1] for row in rows],
                "metadatas": [deepcopy(row[2]) for row in rows]}


class FakeModel:
    """每次输出由测试明确提供，不声称模拟模型能保证事实正确。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.schema = None

    def with_structured_output(self, schema, method):
        assert method == "function_calling"
        self.schema = schema
        return self

    def bind(self, **kwargs):
        assert kwargs == {"extra_body": {"thinking": {"type": "disabled"}}}
        return self

    def invoke(self, messages):
        self.calls.append((self.schema, messages))
        if not self.responses:
            raise AssertionError("意外的额外模型调用")
        schema, response = self.responses.pop(0)
        assert schema is self.schema
        if isinstance(response, Exception):
            raise response
        return response


def choice(number=1):
    return {"question_type": "single_choice", "question": f"第{number}个知识点：RRF 是什么？",
            "knowledge_point": f"RRF定义{number}", "source_chunk_ids": ["chunk-1"],
            "evidence": [{"chunk_id": "chunk-1", "quote": "RRF 是倒数排名融合。"}],
            "options": {"A": "倒数排名融合", "B": "只比较分数", "C": "文本切分", "D": "模型初始化"},
            "correct_answer": "A", "explanation": "内部解析哨兵", "reference_answer": "", "key_points": []}


def short(number=1):
    return {"question_type": "short_answer", "question": f"第{number}个知识点：为什么不统一分数？",
            "knowledge_point": f"RRF分数尺度{number}", "source_chunk_ids": ["chunk-1"],
            "evidence": [{"chunk_id": "chunk-1", "quote": "RRF 使用排名，不需要统一不同分数尺度。"}],
            "options": {}, "correct_answer": "", "explanation": "",
            "reference_answer": "内部参考答案哨兵", "key_points": ["内部得分点哨兵"]}


def generation(questions, reason=""):
    return (GeneratedQuiz, {"questions": questions, "insufficient_reason": reason})


def review(approvals):
    return (QualityReview, {"decisions": [{"candidate_id": f"candidate-{i}", "approved": approved,
                                          "reason": "离线预设审核结果"}
                                         for i, approved in enumerate(approvals, start=1)]})


class QuizTests(unittest.TestCase):
    def run_quiz(self, model, single_choice=1, short_answer=0, store=None, source=SOURCE,
                 headings=None, vector_store=None):
        return generate_quiz(vector_store if vector_store is not None else FakeStore(), source,
                             HEADINGS if headings is None else headings, single_choice, short_answer,
                             model=model, quiz_store=store if store is not None else QuizStore())

    def test_count_boundaries_and_combinations(self):
        for counts in [(0, 0), (-1, 1), (6, 0), (0, 4), (True, 0), (0, True), (1.0, 0), ("1", 0)]:
            with self.subTest(counts=counts):
                model = FakeModel([])
                self.assertEqual(self.run_quiz(model, *counts).status, "invalid_request")
                self.assertEqual(model.calls, [])
        for counts in [(0, 1), (0, 3), (5, 0), (1, 0), (1, 1), (5, 3)]:
            with self.subTest(counts=counts):
                questions = [choice(i) for i in range(counts[0])] + [short(i) for i in range(counts[1])]
                model = FakeModel([generation(questions), review([True] * len(questions))])
                result = self.run_quiz(model, *counts)
                self.assertEqual(result.status, "ready")
                self.assertEqual(result.actual, QuestionCounts(single_choice=counts[0], short_answer=counts[1]).model_dump())

    def test_missing_notes_chapters_empty_and_metadata(self):
        for vector_store, source, headings, status in [
            (FakeStore(), normalize_source("missing.md"), HEADINGS, "note_not_found"),
            (FakeStore(), SOURCE, {"Header 1": "不存在"}, "chapter_not_found"),
            (FakeStore([("x", "# 检索\n## RRF", {"source": SOURCE, "chunk_id": "x", "chunk_index": 1, **HEADINGS})]), SOURCE, HEADINGS, "empty_chapter"),
            (FakeStore([("x", BODY, {"source": SOURCE, "chunk_id": "x", **HEADINGS})]), SOURCE, HEADINGS, "invalid_metadata"),
        ]:
            with self.subTest(status=status):
                model = FakeModel([])
                result = self.run_quiz(model, vector_store=vector_store, source=source, headings=headings)
                self.assertEqual(result.chapter_status, status)
                self.assertEqual(model.calls, [])

    def test_private_answers_and_storage_isolation(self):
        store = QuizStore()
        model = FakeModel([generation([choice(), short()]), review([True, True])])
        result = self.run_quiz(model, 1, 1, store)
        public = json.dumps(result.model_dump(), ensure_ascii=False)
        for forbidden in ["correct_answer", "reference_answer", "key_points", "explanation",
                          "knowledge_point", "evidence", "source_chunk_ids", "内部"]:
            self.assertNotIn(forbidden, public)
        saved = store.get(result.quiz_id)
        self.assertEqual(saved.questions[0].correct_answer, "A")
        self.assertEqual(saved.questions[1].reference_answer, "内部参考答案哨兵")
        saved.questions[0].correct_answer = "B"
        self.assertEqual(store.get(result.quiz_id).questions[0].correct_answer, "A")
        self.assertIsNone(QuizStore().get(result.quiz_id))
        second = self.run_quiz(FakeModel([generation([choice()]), review([True])]), store=store)
        self.assertNotEqual(result.quiz_id, second.quiz_id)
        self.assertGreaterEqual(len(result.quiz_id), 40)
        self.assertIsNone(store.get("unknown"))

    def test_insufficient_knowledge_and_no_questions(self):
        model = FakeModel([generation([choice()], "只有一个可独立出题的知识点。"), review([True])])
        result = self.run_quiz(model, 5, 3)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.requested.single_choice, 5)
        self.assertEqual(result.actual, {"single_choice": 1, "short_answer": 0})
        self.assertTrue(result.shortage_reason)
        self.assertEqual(len(model.calls), 2)
        store = QuizStore()
        result = self.run_quiz(FakeModel([generation([], "没有足够依据")]), store=store)
        self.assertEqual(result.status, "insufficient_content")
        self.assertIsNone(result.quiz_id)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(store._quizzes, {})

    def test_bad_json_missing_fields_and_limited_retry(self):
        for response in ["not-json", {"questions": [{}], "insufficient_reason": ""},
                         {"questions": [dict(choice(), correct_answer="E")], "insufficient_reason": ""},
                         {"questions": [dict(choice(), options={"A": "same", "B": "same", "C": "c", "D": "d"})], "insufficient_reason": ""},
                         {"questions": [dict(short(), key_points=[])], "insufficient_reason": ""}]:
            with self.subTest(response=response):
                model = FakeModel([(GeneratedQuiz, response), (GeneratedQuiz, response)])
                result = self.run_quiz(model)
                self.assertEqual(result.status, "invalid_model_output")
                self.assertEqual(result.model_calls, 2)
                self.assertIsNone(result.quiz_id)
        model = FakeModel([(GeneratedQuiz, OutputParserException("invalid JSON")), generation([choice()]), review([True])])
        self.assertEqual(self.run_quiz(model).status, "ready")

    def test_duplicate_unsupported_and_excess_questions(self):
        unsupported = dict(choice(3), source_chunk_ids=["outside"], evidence=[{"chunk_id": "outside", "quote": BODY}])
        fake_quote = dict(choice(4), evidence=[{"chunk_id": "chunk-1", "quote": "不在原文中的事实"}])
        model = FakeModel([generation([choice(), choice(), unsupported, fake_quote], "不凑题"), review([True])])
        result = self.run_quiz(model, 5, 0)
        self.assertEqual(result.actual["single_choice"], 1)
        for call in model.calls:
            if call[0] is QualityReview:
                self.assertEqual(len(json.loads(call[1][1].content)["candidates"]), 1)
        model = FakeModel([generation([choice(1), choice(2)]), review([True])])
        self.assertEqual(self.run_quiz(model, 1, 0).actual["single_choice"], 1)
        model = FakeModel([generation([unsupported, fake_quote]), generation([unsupported, fake_quote])])
        self.assertEqual(self.run_quiz(model).status, "invalid_model_output")
        self.assertEqual([call[0] for call in model.calls], [GeneratedQuiz, GeneratedQuiz])

    def test_heading_only_evidence_and_storage_error(self):
        rows = [("chunk-1", "## RRF\n" + BODY, {"source": SOURCE, "chunk_id": "chunk-1",
                                                 "chunk_index": 1, **HEADINGS})]
        candidate = dict(choice(), evidence=[{"chunk_id": "chunk-1", "quote": "## RRF"}])
        model = FakeModel([generation([candidate]), generation([candidate])])
        result = self.run_quiz(model, vector_store=FakeStore(rows))
        self.assertEqual(result.status, "invalid_model_output")
        self.assertEqual(result.model_calls, 2)
        class BrokenStore:
            def get(self, **kwargs):
                raise RuntimeError("private-storage-details")
        result = self.run_quiz(FakeModel([]), vector_store=BrokenStore())
        self.assertEqual(result.chapter_status, "storage_error")
        self.assertNotIn("private-storage-details", result.model_dump_json())

    def test_semantic_review_rejection_and_coverage(self):
        # 模拟审校拒绝同义重复或多正确选项；不是实际模型语义准确率证据。
        model = FakeModel([generation([choice(1), choice(2)], "不继续凑题"), review([True, False])])
        result = self.run_quiz(model, 2, 0)
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.questions), 1)
        bad_review = (QualityReview, {"decisions": [{"candidate_id": "unknown", "approved": True, "reason": ""}]})
        model = FakeModel([generation([choice()]), bad_review, generation([choice()]), bad_review])
        result = self.run_quiz(model)
        self.assertEqual(result.status, "invalid_model_output")
        self.assertEqual(result.model_calls, 4)

    def test_model_failure_safe_error_and_partial_retention(self):
        for responses in [[(GeneratedQuiz, RuntimeError("secret-request-body"))],
                          [generation([choice()]), (QualityReview, RuntimeError("secret-request-body"))]]:
            result = self.run_quiz(FakeModel(responses))
            self.assertEqual(result.status, "model_error")
            self.assertNotIn("secret-request-body", result.model_dump_json())
            self.assertIsNone(result.quiz_id)
        model = FakeModel([generation([choice()]), review([True]), (GeneratedQuiz, RuntimeError("secret"))])
        result = self.run_quiz(model, 2, 0)
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.questions), 1)
        with patch("app.quiz.service.create_model", side_effect=RuntimeError("secret-key")):
            result = generate_quiz(FakeStore(), SOURCE, HEADINGS, 1, 0, quiz_store=QuizStore())
        self.assertEqual(result.status, "model_error")
        self.assertNotIn("secret-key", result.model_dump_json())

    def test_metadata_selection_order_descendants_and_same_name(self):
        rows = []
        for index, headings in [(1, HEADINGS), (2, {**HEADINGS, "Header 3": "优点"}),
                                (3, {"Header 1": "检索", "Header 2": "BM25"})]:
            rows.append((f"chunk-{index}", f"正文{index}", {"source": SOURCE, "chunk_id": f"chunk-{index}",
                                                          "chunk_index": index, **headings}))
        other = normalize_source("other/quiz-test-notes.md")
        rows.append(("other", "不可混入", {"source": other, "chunk_id": "other", "chunk_index": 1, **HEADINGS}))
        store = FakeStore(list(reversed(rows)))
        result = read_chapter(store, SOURCE, HEADINGS)
        self.assertEqual([chunk.chunk_index for chunk in result.chunks], [1, 2])
        self.assertNotIn("正文3", result.content)
        self.assertNotIn("不可混入", result.content)
        self.assertEqual(len(list_notes(store)), 2)
        self.assertIn({**HEADINGS, "Header 3": "优点"}, list_chapters(store, SOURCE))
        rows[2] = ("chunk-3", "重复正文", {"source": SOURCE, "chunk_id": "chunk-3", "chunk_index": 3, **HEADINGS})
        rows[1] = ("chunk-2", "隔开的章节", {"source": SOURCE, "chunk_id": "chunk-2", "chunk_index": 2, "Header 1": "其他"})
        self.assertEqual(read_chapter(FakeStore(rows), SOURCE, HEADINGS).status, "ambiguous_chapter")

    def test_preamble_and_sparse_headings(self):
        rows = [("a", "无标题正文", {"source": SOURCE, "chunk_index": 1, "chunk_id": "a"}),
                ("b", "### 跳级\n正文", {"source": SOURCE, "chunk_index": 2, "chunk_id": "b", "Header 3": "跳级"})]
        store = FakeStore(rows)
        self.assertEqual(len(read_chapter(store, SOURCE, {}).chunks), 1)
        self.assertEqual(read_chapter(store, SOURCE, {"Header 3": "跳级"}).status, "ok")
        self.assertIn({"Header 3": "跳级"}, list_chapters(store, SOURCE))

    def test_temporary_real_chroma_metadata_and_multi_chunk(self):
        class OfflineEmbedding(Embeddings):
            def embed_documents(self, texts):
                return [[1.0, 0.0, 0.0] for _ in texts]

            def embed_query(self, text):
                raise AssertionError("章节读取不应执行相似度检索")

        with TemporaryDirectory(prefix="v2-quiz-", ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            with patch("app.rag.vector_retrieval.DashScopeEmbeddings", return_value=OfflineEmbedding()):
                store = create_vector_store(root / "chroma")
            note = root / "notes.md"
            note.write_text("# 检索\n## RRF\n" + BODY * 60 + "\n### 优点\n使用排名。\n## BM25\n关键词检索。", encoding="utf-8")
            imported = import_markdown(note, store)
            result = read_chapter(store, imported["source"], HEADINGS)
            self.assertEqual(result.status, "ok")
            self.assertGreater(len(result.chunks), 2)
            self.assertEqual([c.chunk_index for c in result.chunks], sorted(c.chunk_index for c in result.chunks))
            self.assertTrue(all(c.source == imported["source"] for c in result.chunks))
            self.assertNotIn("关键词检索", result.content)
            self.assertIn({**HEADINGS, "Header 3": "优点"}, list_chapters(store, imported["source"]))
            self.assertEqual(list_notes(store)[0]["filename"], "notes.md")
            # 相邻同名章节可能被旧 splitter 合并，依靠保留的正文标题检测歧义。
            note.write_text("# 检索\n## RRF\n第一处。\n## RRF\n第二处。", encoding="utf-8")
            import_markdown(note, store)
            self.assertEqual(read_chapter(store, imported["source"], HEADINGS).status, "ambiguous_chapter")
            print("临时真实 Chroma：metadata、顺序、多 Chunk、章节隔离和重复标题检查通过")


if __name__ == "__main__":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    unittest.main(verbosity=2)
