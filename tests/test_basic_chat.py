from backend.app.services.basic_chat import get_basic_chat_response


def test_greeting_is_answered_without_rag():
    answer = get_basic_chat_response("你好")

    assert answer is not None
    assert "个人笔记复习助手" in answer


def test_greeting_with_self_introduction_is_answered_without_rag():
    answer = get_basic_chat_response("你好 我叫小余")

    assert answer is not None
    assert answer.startswith("你好，小余。")


def test_greeting_introduction_and_capability_question_are_composed():
    answer = get_basic_chat_response("你好！我叫小余。请问你能干什么")

    assert answer is not None
    assert answer.startswith("你好，小余。")
    assert "章节小测" in answer
    assert "小余请问" not in answer


def test_composed_capability_question_without_punctuation_is_direct():
    answer = get_basic_chat_response("你好 我叫小余 你能干什么")

    assert answer is not None
    assert answer.startswith("你好，小余。")


def test_greeting_and_capability_question_are_composed():
    answer = get_basic_chat_response("你好，请问你能做什么？")

    assert answer is not None
    assert "根据原文进行评分和复盘" in answer


def test_greeting_with_technical_request_is_left_for_rag():
    assert get_basic_chat_response("你好，请解释 Git 是什么") is None


def test_introduction_with_technical_request_is_left_for_rag():
    assert get_basic_chat_response("你好，我叫小余，请解释 Git 是什么") is None


def test_technical_question_is_left_for_rag():
    assert get_basic_chat_response("Git 的常用命令有哪些？") is None


def test_project_usage_variants_are_answered_without_rag():
    for query in ("怎么使用这个项目", "如何使用这个项目", "这个项目怎么用", "这个助手怎么用"):
        assert get_basic_chat_response(query) is not None


def test_technical_usage_questions_are_left_for_rag():
    assert get_basic_chat_response("Git 怎么使用") is None
    assert get_basic_chat_response("这个项目使用了什么技术") is None
