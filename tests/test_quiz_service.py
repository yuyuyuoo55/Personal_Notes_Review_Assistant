import json

import pytest

from backend.app.services import quiz_service
from backend.app.services.quiz_service import (
    generate_quiz_questions,
    grade_quiz_answers,
    parse_quiz_grades,
    parse_quiz_questions,
    parse_quiz_review,
)


CONTEXT = "【Git > 什么是Git】\nGit是一个分布式版本控制工具。\n\n【Git > Git的常用命令】\ngit status 查看文件状态。"


def make_question(question_type: str, question: str, *, correct_option: str = "") -> dict:
    is_choice = question_type == "single_choice"
    return {
        "question_type": question_type,
        "question": question,
        "options": {"A": "分布式版本控制工具", "B": "数据库", "C": "浏览器", "D": "编译器"} if is_choice else {},
        "correct_option": correct_option if is_choice else "",
        "reference_answer": "Git 是一个分布式版本控制工具。",
        "evidence": ["Git是一个分布式版本控制工具。"],
        "core_point": f"{question}的核心结论。",
        "common_mistake": f"{question}的常见混淆。",
        "memory_tip": f"{question}的记忆方法。",
    }


def test_parse_quiz_questions_accepts_two_choices_and_one_short_answer():
    raw = json.dumps(
        {
            "questions": [
                make_question("single_choice", "Git 属于哪类工具？", correct_option="A"),
                make_question("single_choice", "下列哪项是 Git 的定义？", correct_option="A"),
                make_question("short_answer", "Git 是什么？"),
            ]
        },
        ensure_ascii=False,
    )

    questions = parse_quiz_questions(raw, CONTEXT, 3)

    assert [item["question_type"] for item in questions] == ["single_choice", "single_choice", "short_answer"]


def test_parse_quiz_questions_rejects_vague_or_ungrounded_questions():
    vague = make_question("single_choice", "Git 的主要内容是什么？", correct_option="A")
    ungrounded = make_question("single_choice", "GitHub 是什么？", correct_option="A")
    ungrounded["evidence"] = ["GitHub 是代码托管平台。"]
    raw = json.dumps(
        {"questions": [vague, ungrounded, make_question("short_answer", "Git 是什么？")]},
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="两道单选题和一道简答题"):
        parse_quiz_questions(raw, CONTEXT, 3)


def test_generate_quiz_questions_retries_and_overgenerates(monkeypatch):
    invalid = json.dumps({"questions": []}, ensure_ascii=False)
    valid = json.dumps(
        {
            "questions": [
                make_question("single_choice", "Git 属于哪类工具？", correct_option="A"),
                make_question("single_choice", "下列哪项是 Git 的定义？", correct_option="A"),
                make_question("short_answer", "Git 是什么？"),
            ]
        },
        ensure_ascii=False,
    )

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.prompts = []
            self.responses = iter([invalid, valid])

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return FakeResponse(next(self.responses))

    fake_model = FakeModel()
    model_options = []

    def fake_create_chat_model(api_key, **kwargs):
        model_options.append(kwargs)
        return fake_model

    monkeypatch.setattr(quiz_service, "load_quiz_context", lambda *_: CONTEXT)
    monkeypatch.setattr(quiz_service, "create_chat_model", fake_create_chat_model)

    diagnostics = {}
    questions = generate_quiz_questions(
        source="Git.md",
        chapter="整篇笔记",
        api_key="test-key",
        diagnostics=diagnostics,
    )

    assert len(questions) == 3
    assert len(fake_model.prompts) == 2
    assert "本次只生成：2 道 single_choice、1 道 short_answer" in fake_model.prompts[0]
    assert "本次只生成：1 道 single_choice、0 道 short_answer" in fake_model.prompts[1]
    assert model_options[0] == {
        "temperature": 0,
        "max_tokens": quiz_service.QUIZ_MAX_OUTPUT_TOKENS,
        "json_mode": True,
        "disable_thinking": True,
    }
    assert model_options[1]["json_mode"] is False
    assert model_options[1]["disable_thinking"] is True
    assert diagnostics["model_calls"] == 2
    assert len(diagnostics["call_elapsed_ms"]) == 2
    assert diagnostics["request_mode"] == "非思考兼容"


def test_parse_quiz_questions_accepts_common_model_field_variants():
    raw = json.dumps(
        {
            "questions": [
                {
                    "type": "单选题",
                    "question": "Git 属于哪类工具？",
                    "options": ["分布式版本控制工具", "数据库", "浏览器", "编译器"],
                    "correct_answer": "A",
                    "answer": "Git 是一个分布式版本控制工具。",
                    "evidence_ids": ["E1"],
                },
                {
                    "type": "multiple_choice",
                    "question": "下列哪项是 Git 的定义？",
                    "options": [
                        {"label": "A", "text": "分布式版本控制工具"},
                        {"label": "B", "text": "数据库"},
                        {"label": "C", "text": "浏览器"},
                        {"label": "D", "text": "编译器"},
                    ],
                    "answer_option": "A",
                    "answer": "Git 是一个分布式版本控制工具。",
                    "evidence_id": "E1",
                },
                {
                    "type": "简答题",
                    "question": "Git 是什么？",
                    "options": [],
                    "answer": "Git 是一个分布式版本控制工具。",
                    "evidence_ids": ["E1"],
                },
            ]
        },
        ensure_ascii=False,
    )

    questions = parse_quiz_questions(raw, CONTEXT, 3)

    assert [question["question_type"] for question in questions] == [
        "single_choice",
        "single_choice",
        "short_answer",
    ]
    assert questions[0]["options"]["A"] == "分布式版本控制工具"


def test_parse_quiz_questions_replaces_trailing_period_with_question_mark():
    items = [
        make_question("single_choice", "Git 属于哪类工具。", correct_option="A"),
        make_question("single_choice", "下列哪项是 Git 的定义。", correct_option="A"),
        make_question("short_answer", "Git 是什么。"),
    ]

    questions = parse_quiz_questions(
        json.dumps({"questions": items}, ensure_ascii=False),
        CONTEXT,
        3,
    )

    assert [question["question"] for question in questions] == [
        "Git 属于哪类工具？",
        "下列哪项是 Git 的定义？",
        "Git 是什么？",
    ]


def test_parse_quiz_questions_normalizes_full_width_punctuation_and_keeps_original_evidence():
    context = "【Maven > 私服】\n私服:是一种特殊的远程仓库。"
    question = make_question("short_answer", "什么是 Maven 私服？")
    question["reference_answer"] = "私服是一种特殊的远程仓库。"
    question["evidence"] = ["私服：是一种特殊的远程仓库。"]
    raw = json.dumps(
        {
            "questions": [
                make_question("single_choice", "Git 属于哪类工具？", correct_option="A"),
                make_question("single_choice", "下列哪项是 Git 的定义？", correct_option="A"),
                question,
            ]
        },
        ensure_ascii=False,
    )
    combined_context = f"{CONTEXT}\n\n{context}"

    questions = parse_quiz_questions(raw, combined_context, 3)

    assert questions[2]["evidence"] == ["私服:是一种特殊的远程仓库。"]


def test_generate_quiz_questions_repairs_only_missing_type(monkeypatch):
    first = json.dumps(
        {
            "questions": [
                make_question("single_choice", "Git 属于哪类工具？", correct_option="A"),
                make_question("short_answer", "Git 是什么？"),
            ]
        },
        ensure_ascii=False,
    )
    second = json.dumps(
        {"questions": [make_question("single_choice", "下列哪项是 Git 的定义？", correct_option="A")]},
        ensure_ascii=False,
    )

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.prompts = []
            self.responses = iter([first, second])

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return FakeResponse(next(self.responses))

    fake_model = FakeModel()
    monkeypatch.setattr(quiz_service, "load_quiz_context", lambda *_: CONTEXT)
    monkeypatch.setattr(quiz_service, "create_chat_model", lambda *_args, **_kwargs: fake_model)

    questions = generate_quiz_questions(source="Git.md", chapter="整篇笔记", api_key="test-key")

    assert [item["question_type"] for item in questions] == ["single_choice", "single_choice", "short_answer"]
    assert "本次只生成：1 道 single_choice、0 道 short_answer" in fake_model.prompts[1]


def test_generate_quiz_questions_falls_back_when_json_mode_is_rejected(monkeypatch):
    valid = json.dumps(
        {
            "questions": [
                make_question("single_choice", "Git 属于哪类工具？", correct_option="A"),
                make_question("single_choice", "下列哪项是 Git 的定义？", correct_option="A"),
                make_question("short_answer", "Git 是什么？"),
            ]
        },
        ensure_ascii=False,
    )
    created_options = []

    class FakeResponse:
        content = valid

    class RejectingModel:
        def invoke(self, _prompt):
            raise RuntimeError("unsupported response format")

    class WorkingModel:
        def invoke(self, _prompt):
            return FakeResponse()

    def fake_create_chat_model(_api_key, **kwargs):
        created_options.append(kwargs)
        return RejectingModel() if kwargs["json_mode"] else WorkingModel()

    monkeypatch.setattr(quiz_service, "load_quiz_context", lambda *_: CONTEXT)
    monkeypatch.setattr(quiz_service, "create_chat_model", fake_create_chat_model)

    diagnostics = {}
    questions = generate_quiz_questions(
        source="Git.md",
        chapter="整篇笔记",
        api_key="test-key",
        diagnostics=diagnostics,
    )

    assert len(questions) == 3
    assert [options["json_mode"] for options in created_options] == [True, False]
    assert diagnostics["model_calls"] == 2
    assert diagnostics["request_mode"] == "非思考兼容"


def test_parse_quiz_grades_accepts_only_question_evidence():
    questions = [make_question("short_answer", "Git 是什么？")]
    raw = json.dumps(
        {
            "grades": [
                {
                    "question_index": 0,
                    "score": 60,
                    "verdict": "部分支持",
                    "feedback": "答出了版本控制，但遗漏了分布式。",
                    "evidence": ["Git是一个分布式版本控制工具。"],
                }
            ]
        },
        ensure_ascii=False,
    )

    grades = parse_quiz_grades(raw, questions, [0])

    assert grades[0]["score"] == 60
    assert grades[0]["verdict"] == "部分支持"


def test_grade_quiz_answers_grades_choices_in_code_and_builds_review(monkeypatch):
    questions = [
        make_question("single_choice", "Git 属于哪类工具？", correct_option="A"),
        make_question("single_choice", "下列哪项是 Git 的定义？", correct_option="A"),
        make_question("short_answer", "Git 是什么？"),
    ]

    review = [
        {
            "question_index": index,
            "core_point": f"核心 {index}",
            "common_mistake": f"易错 {index}",
            "memory_tip": f"记忆 {index}",
        }
        for index in range(3)
    ]

    class FakeResponse:
        content = json.dumps({"grades": [], "review": review}, ensure_ascii=False)

    class FakeModel:
        def invoke(self, _prompt):
            return FakeResponse()

    monkeypatch.setattr(quiz_service, "create_chat_model", lambda *_args, **_kwargs: FakeModel())

    result = grade_quiz_answers(questions=questions, answers={0: "A", 1: "B", 2: ""}, api_key="test-key")
    grades = result["grades"]

    assert [grade["score"] for grade in grades] == [100, 0, 0]
    assert [grade["verdict"] for grade in grades] == ["笔记支持", "与笔记矛盾", "未作答"]
    assert result["review"] == review


def test_parse_quiz_review_requires_all_three_distinct_sections():
    review = [
        {
            "question_index": index,
            "core_point": f"核心 {index}",
            "common_mistake": f"易错 {index}",
            "memory_tip": f"记忆 {index}",
        }
        for index in range(3)
    ]

    parsed = parse_quiz_review(json.dumps({"review": review}, ensure_ascii=False), 3)

    assert parsed == review
