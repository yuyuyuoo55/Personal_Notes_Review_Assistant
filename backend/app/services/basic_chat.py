"""不依赖笔记库的基础对话回复。"""


def get_basic_chat_response(query: str) -> str | None:
    """命中基础问候或产品说明时直接回复；其他问题返回 None 交给 RAG。"""
    normalized_query = "".join(query.strip().lower().split()).rstrip("，。！？?!")

    if normalized_query in {"你好", "嗨", "hello", "hi"}:
        return "你好，我是你的个人笔记复习助手。导入 Markdown 笔记后，我可以基于笔记帮你复习和答疑。"

    if "你是谁" in normalized_query or "你是什么" in normalized_query:
        return "我是个人笔记复习助手。我会优先从你导入的笔记中检索资料，再基于资料回答问题并展示来源。"

    if "你能做什么" in normalized_query or "怎么用" in normalized_query or "如何使用" in normalized_query:
        return "你可以先在左侧导入 Markdown 笔记，再在中间提问。我会检索笔记、给出回答，并附上可回看的来源片段。"

    if normalized_query in {"谢谢", "感谢", "thanks", "thankyou"}:
        return "不客气，继续拿笔记里的问题来问我就行。"

    return None
