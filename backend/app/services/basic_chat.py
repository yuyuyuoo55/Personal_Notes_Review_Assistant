"""不依赖笔记库的基础对话回复。"""

import re


_GREETING_PREFIXES = (
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "你好呀",
    "你好啊",
    "你好",
    "哈喽",
    "hello",
    "嗨",
    "hi",
)

_INTRODUCTION_PATTERN = re.compile(
    r"^(?:我叫|我是|你可以叫我|叫我)"
    r"(?P<name>[\u4e00-\u9fffa-z0-9_-]{1,16}?)"
    r"(?=\||请问|请解释|解释|介绍|讲解|你能|你会|你可以|你是|怎么|如何|"
    r"有什么|什么是|能做|能干|会做|$)"
)

_CAPABILITY_QUESTIONS = {
    "你能干什么",
    "你能做什么",
    "你会什么",
    "你可以做什么",
    "能干什么",
    "能做什么",
    "会做什么",
    "你有什么功能",
    "有什么功能",
    "怎么用",
    "怎么使用",
    "怎么使用这个项目",
    "如何使用这个项目",
    "这个项目怎么用",
    "这个助手怎么用",
    "如何使用",
    "如何用",
}

_IDENTITY_QUESTIONS = {"你是谁", "你是什么"}


def _normalize_query(query: str) -> str:
    """去掉空白，并把标点保留为分句边界，避免姓名吞掉后续问题。"""
    normalized = re.sub(r"\s+", "", query.strip().lower())
    return re.sub(r"[，。！？?!、；;：:]+", "|", normalized).strip("|")


def get_basic_chat_response(query: str) -> str | None:
    """命中基础问候或产品说明时直接回复；其他问题返回 None 交给 RAG。"""
    remaining = _normalize_query(query)
    had_greeting = False
    name: str | None = None

    for greeting in _GREETING_PREFIXES:
        if remaining.startswith(greeting):
            remaining = remaining[len(greeting) :].lstrip("|")
            had_greeting = True
            break

    introduction_match = _INTRODUCTION_PATTERN.match(remaining)
    if introduction_match:
        name = introduction_match.group("name")
        remaining = remaining[introduction_match.end() :].lstrip("|")

    if remaining.startswith("请问"):
        remaining = remaining[len("请问") :].lstrip("|")

    remaining = remaining.replace("|", "")

    greeting_prefix = f"你好，{name}。" if name else "你好。"

    if not remaining and (had_greeting or name):
        return (
            f"{greeting_prefix}我是你的个人笔记复习助手，可以陪你围绕已导入的笔记"
            "进行问答、章节小测和复盘。"
        )

    if remaining in _CAPABILITY_QUESTIONS:
        return (
            f"{greeting_prefix}我可以基于你导入的笔记回答问题、生成章节小测，"
            "并根据原文进行评分和复盘。"
        )

    if remaining in _IDENTITY_QUESTIONS:
        return (
            f"{greeting_prefix}我是个人笔记复习助手。我会优先从你导入的笔记中检索资料，"
            "再基于资料回答问题并展示来源。"
        )

    if remaining in {"谢谢", "感谢", "thanks", "thankyou"}:
        return "不客气，继续拿笔记里的问题来问我就行。"

    return None
