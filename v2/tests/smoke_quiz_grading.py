"""Phase 16B 离线流程验收；Fake 批改/报告不等于真实模型质量验证。"""

import json
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smoke_quiz import (BODY, HEADINGS, SOURCE, FakeModel, FakeStore, choice, generation,
                        review, short)
from app.quiz.grading import submit_answer, submit_quiz
from app.quiz.report import complete_quiz
from app.quiz.schemas import ReportContent, ShortAnswerAssessment
from app.quiz.service import generate_quiz
from app.quiz.store import QuizStore


def make_quiz(choice_count=1, short_count=1):
    questions = [choice(i) for i in range(choice_count)]
    for i in range(short_count):
        question = short(i)
        question["reference_answer"] = "RRF 使用排名，不需要统一不同分数尺度。"
        question["key_points"] = ["使用排名", "不需要统一分数尺度"]
        questions.append(question)
    store = QuizStore()
    model = FakeModel([generation(questions), review([True] * len(questions))])
    public = generate_quiz(FakeStore(), SOURCE, HEADINGS, choice_count, short_count,
                           model=model, quiz_store=store)
    assert public.status == "ready"
    return store, public, store.get(public.quiz_id)


def assessment(grade="优秀"):
    data = {"grade": grade, "feedback": "根据本次用户答案与原文评价。",
            "correct_points": ["能说明使用排名"] if grade != "不及格" else [],
            "missing_points": [] if grade == "优秀" else ["遗漏或误解：不需要统一分数尺度"],
            "suggestion": "对照原文，理解排名与分数尺度的区别。"}
    return (ShortAnswerAssessment, data)


def report_response(quiz, needs_review=()):
    point = {"text": "RRF 使用排名，而不是统一分数尺度。",
             "evidence": [{"chunk_id": "chunk-1", "quote": "RRF 使用排名，不需要统一不同分数尺度。"}]}
    return (ReportContent, {"core_points": [point], "possible_confusions": [deepcopy(point)],
        "memory_methods": [{"text": "对比记忆：关注名次，不是分数尺度。", "evidence": deepcopy(point["evidence"])}],
        "review_suggestions": [{"focus": "加强" if q.question_id in needs_review else "巩固",
                                "question_ids": [q.question_id], "suggestion": "复习 RRF 排名与分数尺度的关系。"}
                               for q in quiz.questions]})


def single_payload(quiz, question, answer, **extra):
    return {"quiz_id": quiz.quiz_id, "question_id": question.question_id, "user_answer": answer, **extra}


def batch_payload(quiz, answers):
    return {"quiz_id": quiz.quiz_id, "answers": [{"question_id": q.question_id, "user_answer": answer}
                                                 for q, answer in zip(quiz.questions, answers)]}


class GradingTests(unittest.TestCase):
    def test_choice_correct_wrong_without_model(self):
        for answer, expected in [("A", True), ("B", False), (" a ", True)]:
            with self.subTest(answer=answer):
                store, public, quiz = make_quiz(1, 0)
                model = FakeModel([])
                with patch("app.quiz.grading.create_model", side_effect=AssertionError("不能初始化模型")):
                    result = submit_answer(single_payload(quiz, quiz.questions[0], answer), model=model, quiz_store=store)
                self.assertEqual(result.status, "graded")
                self.assertIs(result.result.correct, expected)
                self.assertEqual(result.result.correct_answer, "A")
                self.assertEqual(result.result.explanation, quiz.questions[0].explanation)
                self.assertEqual(model.calls, [])

    def test_all_five_grades_and_paraphrased_answers(self):
        cases = [("优秀", "利用各候选的名次来融合，不必把不同打分统一尺度。"),
                 ("良好", "主要看名次。"), ("中等", "融合结果，但不清楚尺度。"),
                 ("及格", "可能与排名有关。"), ("不及格", "先把所有分数统一后才能使用 RRF。")]
        for grade, answer in cases:
            with self.subTest(grade=grade):
                store, public, quiz = make_quiz(0, 1)
                model = FakeModel([assessment(grade)])
                result = submit_answer(single_payload(quiz, quiz.questions[0], answer), model=model, quiz_store=store)
                self.assertEqual(result.status, "graded")
                self.assertEqual(result.result.grade, grade)
                self.assertEqual(result.result.user_answer, answer)
                self.assertEqual(result.result.reference_answer, quiz.questions[0].reference_answer)
                self.assertNotIn("score", result.model_dump_json())
                payload = json.loads(model.calls[0][1][1].content)
                self.assertEqual(payload["original_notes"][0]["content"], BODY)
                self.assertEqual(payload["key_points"], quiz.questions[0].key_points)
                self.assertIn("允许用户使用自己的话", model.calls[0][1][0].content)

    def test_request_ids_types_and_forged_answers(self):
        store, public, quiz = make_quiz()
        question = quiz.questions[0]
        for payload, status in [
            (single_payload(quiz, question, "A", quiz_id="absent"), "quiz_not_found"),
            (single_payload(quiz, question, "A", question_id="absent"), "question_not_found"),
            (single_payload(quiz, question, "E"), "invalid_request"),
            (single_payload(quiz, question, "A", question_type="short_answer"), "invalid_request"),
            (single_payload(quiz, question, "A", correct_answer="A"), "invalid_request"),
            (single_payload(quiz, question, "A", reference_answer="伪造"), "invalid_request"),
            (single_payload(quiz, question, " "), "invalid_request"),
            (single_payload(quiz, question, 3), "invalid_request"),
            (single_payload(quiz, quiz.questions[1], "x" * 10001), "invalid_request"),
        ]:
            with self.subTest(status=status, payload=payload):
                result = submit_answer(payload, model=FakeModel([]), quiz_store=store)
                self.assertEqual(result.status, status)
                self.assertIsNone(result.result)
        self.assertEqual(store.get_progress(quiz.quiz_id).answers, {})
        self.assertEqual(complete_quiz([], quiz_store=store).status, "invalid_request")
        self.assertEqual(complete_quiz("absent", quiz_store=store).status, "quiz_not_found")

    def test_repeated_answers_are_cached_and_changed_answer_rejected(self):
        store, public, quiz = make_quiz(0, 1)
        question = quiz.questions[0]
        payload = single_payload(quiz, question, "使用排名，不必统一分数尺度。")
        model = FakeModel([assessment()])
        first = submit_answer(payload, model=model, quiz_store=store)
        second = submit_answer(payload, model=model, quiz_store=store)
        self.assertEqual(second.status, "already_graded")
        self.assertEqual(first.result, second.result)
        self.assertEqual(len(model.calls), 1)
        changed = submit_answer(single_payload(quiz, question, "不同答案"), model=model, quiz_store=store)
        self.assertEqual(changed.status, "submission_conflict")
        first.result.feedback = "尝试修改副本"
        self.assertNotEqual(store.get_progress(quiz.quiz_id).grades[question.question_id].feedback, "尝试修改副本")

    def test_incomplete_has_no_report_or_unsubmitted_answers(self):
        store, public, quiz = make_quiz()
        model = FakeModel([])
        result = complete_quiz(quiz.quiz_id, model=model, quiz_store=store)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.results, [])
        self.assertEqual(len(result.pending_question_ids), 2)
        self.assertIsNone(result.report)
        submit_answer(single_payload(quiz, quiz.questions[0], "B"), model=model, quiz_store=store)
        result = complete_quiz(quiz.quiz_id, model=model, quiz_store=store)
        self.assertEqual(len(result.results), 1)
        self.assertNotIn("reference_answer", result.model_dump_json())
        self.assertEqual(model.calls, [])
        for key in ["correct_answer", "reference_answer", "explanation", "key_points"]:
            self.assertNotIn(key, public.model_dump_json())

    def test_invalid_batch_is_checked_before_any_state_change(self):
        store, public, quiz = make_quiz()
        valid = batch_payload(quiz, ["A", "使用排名"])
        cases = []
        cases.append({"quiz_id": "absent", "answers": valid["answers"]})
        cases.append({"quiz_id": quiz.quiz_id, "answers": valid["answers"][:1]})
        cases.append({"quiz_id": quiz.quiz_id, "answers": valid["answers"][:1] * 2})
        unknown = deepcopy(valid); unknown["answers"][1]["question_id"] = "unknown"; cases.append(unknown)
        wrong_type = deepcopy(valid); wrong_type["answers"][1]["question_type"] = "single_choice"; cases.append(wrong_type)
        forged = deepcopy(valid); forged["answers"][1]["reference_answer"] = "伪造"; cases.append(forged)
        wrong_choice = deepcopy(valid); wrong_choice["answers"][0]["user_answer"] = "E"; cases.append(wrong_choice)
        for payload in cases:
            self.assertNotEqual(submit_quiz(payload, model=FakeModel([]), quiz_store=store).status, "completed")
            self.assertEqual(store.get_progress(quiz.quiz_id).answers, {})

    def test_grade_failure_invalid_schema_and_retry_same_answer(self):
        for failure in [RuntimeError("secret-key-and-request"), {"grade": 90},
                        {**assessment()[1], "reference_answer": "不能由模型替换"},
                        {**assessment()[1], "missing_points": ["优秀但遗漏很多"]}]:
            with self.subTest(failure=failure):
                store, public, quiz = make_quiz(0, 1)
                question = quiz.questions[0]
                payload = single_payload(quiz, question, "依据名次融合。")
                model = FakeModel([(ShortAnswerAssessment, failure), assessment("良好")])
                result = submit_answer(payload, model=model, quiz_store=store)
                self.assertIn(result.status, {"model_error", "invalid_model_output"})
                self.assertIsNone(result.result)
                self.assertNotIn("secret-key", result.model_dump_json())
                progress = store.get_progress(quiz.quiz_id)
                self.assertEqual(progress.active_questions, [])
                self.assertEqual(progress.grades, {})
                changed = submit_answer(single_payload(quiz, question, "改变答案"), model=model, quiz_store=store)
                self.assertEqual(changed.status, "submission_conflict")
                retried = submit_answer(payload, model=model, quiz_store=store)
                self.assertEqual(retried.status, "graded")
                self.assertEqual(retried.result.grade, "良好")

    def test_model_initialization_failures_release_state(self):
        store, public, quiz = make_quiz(0, 1)
        payload = single_payload(quiz, quiz.questions[0], "回答")
        with patch("app.quiz.grading.create_model", side_effect=ValueError("private-api-key")):
            result = submit_answer(payload, quiz_store=store)
        self.assertEqual(result.status, "model_error")
        self.assertEqual(store.get_progress(quiz.quiz_id).active_questions, [])
        submit_answer(payload, model=FakeModel([assessment()]), quiz_store=store)
        with patch("app.quiz.report.create_model", side_effect=ValueError("private-api-key")):
            result = complete_quiz(quiz.quiz_id, quiz_store=store)
        self.assertEqual(result.status, "model_error")
        self.assertFalse(store.get_progress(quiz.quiz_id).report_active)
        self.assertNotIn("private-api-key", result.model_dump_json())

    def test_full_report_matches_actual_results_and_duplicate_batch(self):
        store, public, quiz = make_quiz(2, 1)
        wrong = [quiz.questions[1].question_id, quiz.questions[2].question_id]
        model = FakeModel([assessment("中等"), report_response(quiz, wrong)])
        payload = batch_payload(quiz, ["A", "B", "只知道融合结果"])
        result = submit_quiz(payload, model=model, quiz_store=store)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.results), 3)
        overview = result.report.overview
        self.assertEqual(overview.actual_question_count, 3)
        self.assertEqual(overview.single_choice_count, 2)
        self.assertEqual(overview.single_choice_correct, 1)
        self.assertEqual(overview.short_answer_ratings[0].grade, "中等")
        self.assertEqual(overview.headings, HEADINGS)
        self.assertEqual({m.question_id for m in result.report.actual_mistakes}, set(wrong))
        self.assertTrue(result.report.core_points)
        self.assertTrue(result.report.possible_confusions)
        self.assertTrue(result.report.memory_methods)
        self.assertEqual(len(model.calls), 2)
        report_input = json.loads(model.calls[1][1][1].content)
        self.assertEqual(report_input["overview"]["single_choice_correct"], 1)
        self.assertEqual(len(report_input["grades"]), 3)
        cached = submit_quiz(payload, model=FakeModel([]), quiz_store=store)
        self.assertEqual(cached.status, "completed")
        self.assertEqual(cached.report, result.report)
        changed = batch_payload(quiz, ["B", "B", "只知道融合结果"])
        self.assertEqual(submit_quiz(changed, model=FakeModel([]), quiz_store=store).status, "submission_conflict")

    def test_partial_failure_preserves_success_and_retries_only_failed_part(self):
        store, public, quiz = make_quiz(1, 2)
        payload = batch_payload(quiz, ["A", "用名次融合，无需统一分数", "先统一分数"])
        model = FakeModel([assessment(), (ShortAnswerAssessment, RuntimeError("private-failure"))])
        result = submit_quiz(payload, model=model, quiz_store=store)
        self.assertEqual(result.status, "partial_failure")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.failed_question_ids, [quiz.questions[2].question_id])
        self.assertIsNone(result.report)
        retry = FakeModel([assessment("不及格"), report_response(quiz, [quiz.questions[2].question_id])])
        result = submit_quiz(payload, model=retry, quiz_store=store)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(retry.calls), 2)
        self.assertEqual(result.report.overview.short_answer_ratings[0].grade, "优秀")
        self.assertEqual(result.report.overview.short_answer_ratings[1].grade, "不及格")

    def test_report_failure_invalid_evidence_mismatch_and_retry(self):
        for kind in ["call", "structure", "quote", "target", "wrong_focus", "fake_overview"]:
            with self.subTest(kind=kind):
                store, public, quiz = make_quiz(1, 0)
                valid_report = report_response(quiz)[1]
                bad = deepcopy(valid_report)
                if kind == "call": bad = RuntimeError("secret-report-token")
                if kind == "structure": bad = {"core_points": []}
                if kind == "quote": bad["core_points"][0]["evidence"][0]["quote"] = "伪造原文"
                if kind == "target": bad["review_suggestions"][0]["question_ids"] = ["unknown"]
                if kind == "wrong_focus": bad["review_suggestions"][0]["focus"] = "加强"
                if kind == "fake_overview": bad["overview"] = {"single_choice_correct": 999}
                model = FakeModel([(ReportContent, bad)])
                payload = batch_payload(quiz, ["A"])
                result = submit_quiz(payload, model=model, quiz_store=store)
                self.assertIn(result.status, {"model_error", "invalid_model_output"})
                self.assertIsNone(result.report)
                self.assertEqual(len(result.results), 1)
                self.assertFalse(store.get_progress(quiz.quiz_id).report_active)
                self.assertIsNone(store.get_progress(quiz.quiz_id).report)
                self.assertNotIn("secret-report-token", result.model_dump_json())
                retry = FakeModel([report_response(quiz)])
                result = submit_quiz(payload, model=retry, quiz_store=store)
                self.assertEqual(result.status, "completed")
                self.assertEqual(len(retry.calls), 1)

    def test_no_user_mistakes_when_all_correct(self):
        store, public, quiz = make_quiz()
        model = FakeModel([assessment(), report_response(quiz)])
        result = submit_quiz(batch_payload(quiz, ["A", "依靠排名，无需统一分数尺度"]), model=model, quiz_store=store)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.report.actual_mistakes, [])
        self.assertTrue(result.report.possible_confusions)

    def test_concurrent_duplicate_answer_is_busy_and_calls_model_once(self):
        store, public, quiz = make_quiz(0, 1)
        entered, release = Event(), Event()
        class BlockingModel(FakeModel):
            def invoke(self, messages):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timeout")
                return super().invoke(messages)
        model = BlockingModel([assessment()])
        payload = single_payload(quiz, quiz.questions[0], "回答")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(submit_answer, payload, model=model, quiz_store=store)
            try:
                self.assertTrue(entered.wait(5))
                duplicate = submit_answer(payload, model=model, quiz_store=store)
                self.assertEqual(duplicate.status, "in_progress")
                self.assertIsNone(duplicate.result)
                self.assertEqual(complete_quiz(quiz.quiz_id, model=model, quiz_store=store).status, "incomplete")
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5).status, "graded")
        self.assertEqual(len(model.calls), 1)

    def test_report_in_progress_reentrant_call_has_no_duplicate_model_call(self):
        store, public, quiz = make_quiz(1, 0)
        submit_answer(single_payload(quiz, quiz.questions[0], "A"), quiz_store=store)
        class ReentrantModel(FakeModel):
            def invoke(self, messages):
                duplicate = complete_quiz(quiz.quiz_id, model=self, quiz_store=store)
                self.duplicate_status = duplicate.status
                return super().invoke(messages)
        model = ReentrantModel([report_response(quiz)])
        result = complete_quiz(quiz.quiz_id, model=model, quiz_store=store)
        self.assertEqual(model.duplicate_status, "in_progress")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(model.calls), 1)

    def test_restart_loses_quiz_and_progress(self):
        store, public, quiz = make_quiz(1, 0)
        result = submit_answer(single_payload(quiz, quiz.questions[0], "A"), quiz_store=QuizStore())
        self.assertEqual(result.status, "quiz_not_found")


if __name__ == "__main__":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    unittest.main(verbosity=2)
