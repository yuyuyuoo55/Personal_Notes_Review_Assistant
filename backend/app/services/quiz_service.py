"""基于指定笔记章节生成可验证的小测题。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from time import perf_counter
import unicodedata

from backend.app.core.logger import get_logger
from backend.app.services.note_loader import load_notes
from backend.app.services.note_splitter import split_documents
from backend.app.services.rag_service import create_chat_model


MAX_QUIZ_CONTEXT_CHARS = 10_000
MAX_QUIZ_CONTEXT_CHUNKS = 8
# 整篇笔记先由代码挑出信息完整、互不重复的知识点，不再把全文一次性塞给模型。
QUIZ_FOCUS_SECTION_LIMIT = 3
QUIZ_FOCUS_SECTION_CHARS = 700
QUIZ_FOCUS_MIN_SECTION_CHARS = 40
# 代码给原文片段编号，模型只回 evidence_ids，避免逐字抄写出错导致整题作废。
QUIZ_EVIDENCE_BLOCK_CHARS = 220
# 首次生成 + 两次按题型补题；一道题校验失败时只补缺的那一道。
QUIZ_MAX_ATTEMPTS = 5
QUIZ_MAX_OUTPUT_TOKENS = 1_600
QUIZ_THINKING_MAX_OUTPUT_TOKENS = 4_096
QUIZ_MODEL_PROFILES = (
    {"label": "非思考 JSON", "json_mode": True, "disable_thinking": True},
    {"label": "非思考兼容", "json_mode": False, "disable_thinking": True},
    {"label": "思考兼容", "json_mode": False, "disable_thinking": False},
)
VAGUE_QUESTION_PATTERNS = (
    "主要内容是什么",
    "包含哪些关键要点",
    "请解释笔记中的",
    "你会怎样组织答案",
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class QuizQuestion:
    question_type: str
    question: str
    options: dict[str, str]
    correct_option: str
    reference_answer: str
    evidence: list[str]


class _QuizModelInvoker:
    """优先使用快速结构化配置，供应商拒绝参数时安全降级。"""

    def __init__(self, api_key: str, diagnostics: dict | None = None):
        self.api_key = api_key
        self.diagnostics = diagnostics
        self.profile_index = 0
        self.model = self._create_model()
        if diagnostics is not None:
            diagnostics["request_mode"] = QUIZ_MODEL_PROFILES[0]["label"]

    def _create_model(self):
        profile = QUIZ_MODEL_PROFILES[self.profile_index]
        return create_chat_model(
            self.api_key,
            temperature=0,
            max_tokens=(
                QUIZ_THINKING_MAX_OUTPUT_TOKENS
                if not profile["disable_thinking"]
                else QUIZ_MAX_OUTPUT_TOKENS
            ),
            json_mode=profile["json_mode"],
            disable_thinking=profile["disable_thinking"],
        )

    def invoke(self, prompt: str):
        while True:
            call_started_at = perf_counter()
            try:
                response = self.model.invoke(prompt)
            except Exception as error:
                self._record_call(call_started_at)
                status_code = getattr(error, "status_code", None)
                logger.warning(
                    "小测模型配置被拒绝，准备兼容降级：profile=%s error=%s status=%s",
                    QUIZ_MODEL_PROFILES[self.profile_index]["label"],
                    type(error).__name__,
                    status_code,
                )
                if self.profile_index + 1 >= len(QUIZ_MODEL_PROFILES):
                    raise
                self.profile_index += 1
                self.model = self._create_model()
                if self.diagnostics is not None:
                    self.diagnostics["request_mode"] = QUIZ_MODEL_PROFILES[self.profile_index]["label"]
                continue
            self._record_call(call_started_at)
            return response

    def use_repair_profile(self) -> None:
        """内容校验缺题时切到非思考兼容档，避免思考占满 JSON 预算。"""
        repair_index = 1
        if self.profile_index >= repair_index:
            return
        self.profile_index = repair_index
        self.model = self._create_model()
        if self.diagnostics is not None:
            self.diagnostics["request_mode"] = QUIZ_MODEL_PROFILES[self.profile_index]["label"]

    def _record_call(self, call_started_at: float) -> None:
        if self.diagnostics is None:
            return
        elapsed = round((perf_counter() - call_started_at) * 1000)
        self.diagnostics["call_elapsed_ms"].append(elapsed)
        self.diagnostics["model_calls"] += 1


def _header_path(metadata: dict) -> tuple[str, ...]:
    return tuple(
        str(metadata[name]).strip()
        for name in ("Header 1", "Header 2", "Header 3")
        if metadata.get(name)
    )


def load_quiz_context(source: str, chapter: str) -> str:
    """读取所选范围，并优先保留覆盖不同章节的有效片段。"""
    selected_path = tuple(chapter.split(" > ")) if chapter != "整篇笔记" else ()
    candidates: list[tuple[tuple[str, ...], str]] = []
    for chunk in split_documents(load_notes(source)):
        path = _header_path(chunk.metadata)
        if selected_path and path[: len(selected_path)] != selected_path:
            continue
        title = " > ".join(path) or "未标注章节"
        content = chunk.page_content.strip()
        if content:
            candidates.append((path, f"【{title}】\n{content}"))

    # 整篇笔记优先选取不同二级章节的片段，再按原顺序补足，避免只截到文件开头。
    selected: list[str] = []
    selected_indices: set[int] = set()
    seen_sections: set[tuple[str, ...]] = set()
    for index, (path, section) in enumerate(candidates):
        section_key = path[:2] if len(path) >= 2 else path
        if section_key in seen_sections:
            continue
        seen_sections.add(section_key)
        selected.append(section)
        selected_indices.add(index)
        if len(selected) == MAX_QUIZ_CONTEXT_CHUNKS:
            break
    for index, (_, section) in enumerate(candidates):
        if len(selected) == MAX_QUIZ_CONTEXT_CHUNKS:
            break
        if index not in selected_indices:
            selected.append(section)

    context = "\n\n".join(selected)
    if not context:
        raise ValueError("所选章节没有可用于出题的正文。")
    if len(context) > MAX_QUIZ_CONTEXT_CHARS:
        context = context[:MAX_QUIZ_CONTEXT_CHARS]
    return context


_SECTION_TITLE_PATTERN = re.compile(r"^【(?P<title>[^】]*)】[ \t]*$", re.MULTILINE)
_SENTENCE_END_CHARS = "。！？；!?;"


def split_context_sections(context: str) -> list[tuple[str, str]]:
    """把出题上下文拆成 (章节标题, 正文)，标题来自真实的 Markdown 标题路径。"""
    matches = list(_SECTION_TITLE_PATTERN.finditer(context))
    if not matches:
        body = context.strip()
        return [("未标注章节", body)] if body else []

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        sections.append((match.group("title").strip(), context[match.end() : body_end].strip()))
    return sections


def select_quiz_focus(context: str) -> str:
    """挑选信息完整且互不重复的知识点，控制单次出题的输入规模。

    整篇笔记往往横跨很多主题，原文越长模型越容易漏题或写错依据。
    这里先按真实章节切分，再优先保留正文充足的章节，最多 3 个。
    """
    sections = split_context_sections(context)
    if len(sections) <= QUIZ_FOCUS_SECTION_LIMIT:
        return context

    qualified = [
        index
        for index, (_, body) in enumerate(sections)
        if len(body) >= QUIZ_FOCUS_MIN_SECTION_CHARS
    ]
    if not qualified:
        return context

    ranked = sorted(
        qualified,
        key=lambda index: (-min(len(sections[index][1]), QUIZ_FOCUS_SECTION_CHARS), index),
    )
    focused: list[str] = []
    for index in sorted(ranked[:QUIZ_FOCUS_SECTION_LIMIT]):
        title, body = sections[index]
        if len(body) > QUIZ_FOCUS_SECTION_CHARS:
            body = body[:QUIZ_FOCUS_SECTION_CHARS]
        focused.append(f"【{title}】\n{body}")
    return "\n\n".join(focused)


def _line_spans(text: str) -> list[tuple[int, int, bool]]:
    """返回每一行的 (起点, 终点, 是否有正文)，空行用于切断片段。"""
    spans: list[tuple[int, int, bool]] = []
    cursor = 0
    for line in text.split("\n"):
        spans.append((cursor, cursor + len(line), bool(line.strip())))
        cursor += len(line) + 1
    return spans


def _split_long_span(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """把过长的一行按句末标点切成不超过 limit 的连续片段。"""
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > limit:
        window_end = cursor + limit
        cut = -1
        for index in range(window_end - 1, cursor, -1):
            if text[index] in _SENTENCE_END_CHARS:
                cut = index + 1
                break
        if cut < 0:
            cut = window_end
        pieces.append((cursor, cut))
        cursor = cut
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def build_evidence_blocks(
    context: str,
    max_chars: int = QUIZ_EVIDENCE_BLOCK_CHARS,
) -> list[tuple[str, str]]:
    """把原文切成带编号的片段，编号交给模型，原文由代码还原。

    返回的每段文本都是上下文的连续原文截取，因此仍然是"逐字依据"。
    """
    ranges: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for start, end, has_text in _line_spans(context):
        if not has_text:
            if current is not None:
                ranges.append(current)
                current = None
            continue
        for piece_start, piece_end in _split_long_span(context, start, end, max_chars):
            if current is not None and piece_end - current[0] <= max_chars:
                current = (current[0], piece_end)
            elif current is None:
                current = (piece_start, piece_end)
            else:
                ranges.append(current)
                current = (piece_start, piece_end)
    if current is not None:
        ranges.append(current)

    blocks: list[tuple[str, str]] = []
    for index, (start, end) in enumerate(ranges, start=1):
        text = context[start:end].strip()
        if text:
            blocks.append((f"E{index}", text))
    return blocks


_BLOCK_ID_PATTERN = re.compile(r"^[\[\(【]?\s*[Ee]?\s*(\d{1,3})\s*[\]\)】]?$")


def _normalize_block_id(value: object) -> str:
    """把 E1 / e1 / [1] / 1 这类写法统一成 E1，其余一律视为无效。"""
    match = _BLOCK_ID_PATTERN.match(str(value).strip())
    return f"E{match.group(1)}" if match else ""


def _extract_json(raw_content: str) -> dict:
    content = raw_content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("模型没有返回合法的小测 JSON。") from error
    if not isinstance(payload, dict):
        raise ValueError("模型返回的小测结构不正确。")
    return payload


_IGNORED_MARKDOWN_CHARS = frozenset("*_`#~")


def _normalized_with_positions(text: str) -> tuple[str, list[int]]:
    """规范化全半角、空白和 Markdown 展示符，并保留原文位置映射。"""
    normalized_chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        for normalized_char in unicodedata.normalize("NFKC", char).casefold():
            if normalized_char.isspace() or normalized_char in _IGNORED_MARKDOWN_CHARS:
                continue
            normalized_chars.append(normalized_char)
            positions.append(index)
    return "".join(normalized_chars), positions


def _normalized(text: str) -> str:
    return _normalized_with_positions(text)[0]


def _original_evidence(context: str, quote: str) -> str | None:
    """在保守规范化匹配后，返回真正来自上下文的原文片段。"""
    normalized_context, positions = _normalized_with_positions(context)
    normalized_quote = _normalized(quote)
    if not normalized_quote:
        return None
    start = normalized_context.find(normalized_quote)
    if start < 0:
        return None
    end = start + len(normalized_quote) - 1
    return context[positions[start] : positions[end] + 1].strip()


def _resolve_evidence(item: dict, context: str, blocks: dict[str, str]) -> list[str]:
    """优先用代码编号还原原文依据；旧格式的逐字引用仍然兼容。"""
    raw_ids = item.get("evidence_ids")
    id_values = list(raw_ids) if isinstance(raw_ids, list) else []
    single_id = item.get("evidence_id")
    if single_id not in (None, ""):
        id_values.append(single_id)

    resolved: list[str] = []
    for raw_id in id_values:
        block_id = _normalize_block_id(raw_id)
        if block_id and block_id in blocks and blocks[block_id] not in resolved:
            resolved.append(blocks[block_id])
    if resolved:
        return resolved

    # 兼容模型直接抄写原文的情况：仍要求能在上下文中逐字定位。
    raw_evidence = item.get("evidence", [])
    quotes = [str(value).strip() for value in raw_evidence] if isinstance(raw_evidence, list) else []
    originals = [_original_evidence(context, quote) for quote in quotes if quote]
    return [quote for quote in originals if quote]


def _count_rejection(report: dict | None, reason: str) -> None:
    if report is not None:
        report[reason] = report.get(reason, 0) + 1


def _normalized_question_type(item: dict) -> str:
    """兼容模型常见的题型字段和值，但内部只保留两种标准题型。"""
    raw = str(item.get("question_type", item.get("type", ""))).strip().casefold()
    aliases = {
        "single_choice": "single_choice",
        "single-choice": "single_choice",
        "multiple_choice": "single_choice",
        "choice": "single_choice",
        "单选": "single_choice",
        "单选题": "single_choice",
        "short_answer": "short_answer",
        "short-answer": "short_answer",
        "short answer": "short_answer",
        "qa": "short_answer",
        "简答": "short_answer",
        "简答题": "short_answer",
        "问答题": "short_answer",
    }
    return aliases.get(raw, raw)


def _normalized_options(raw_options: object) -> dict[str, str]:
    """接受 {A: ...}、[...四项] 以及 [{label, text}] 三种常见模型输出。"""
    if isinstance(raw_options, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in raw_options.items()}
    if not isinstance(raw_options, list):
        return {}
    if len(raw_options) == 4 and all(not isinstance(value, dict) for value in raw_options):
        return {label: str(value).strip() for label, value in zip("ABCD", raw_options)}
    options: dict[str, str] = {}
    for value in raw_options:
        if not isinstance(value, dict):
            return {}
        label = str(value.get("label", value.get("key", ""))).strip().upper()
        text = str(value.get("text", value.get("value", ""))).strip()
        if label:
            options[label] = text
    return options


def _parse_valid_question_candidates(
    raw_content: str,
    context: str,
    report: dict | None = None,
) -> dict[str, list[QuizQuestion]]:
    """解析并返回通过结构、质量和原文依据校验的候选题。"""
    payload = _extract_json(raw_content)
    items = payload.get("questions")
    if not isinstance(items, list):
        raise ValueError("模型返回结果缺少 questions 数组。")

    blocks = dict(build_evidence_blocks(context))
    validated: dict[str, list[QuizQuestion]] = {"single_choice": [], "short_answer": []}
    seen_questions: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            _count_rejection(report, "题目不是 JSON 对象")
            continue
        question = str(item.get("question", "")).strip()
        question_type = _normalized_question_type(item)
        options = _normalized_options(item.get("options", {}))
        correct_option = str(
            item.get("correct_option", item.get("correct_answer", item.get("answer_option", "")))
        ).strip().upper()
        reference_answer = str(item.get("reference_answer", item.get("answer", ""))).strip()
        evidence = _resolve_evidence(item, context, blocks)

        if not all((question, reference_answer, evidence)):
            _count_rejection(report, "缺少题干、参考答案，或依据不在所选原文中")
            continue
        if question_type not in validated:
            _count_rejection(report, "题型不是 single_choice 或 short_answer")
            continue
        if question_type == "single_choice":
            if set(options) != {"A", "B", "C", "D"} or correct_option not in options:
                _count_rejection(report, "单选题选项不完整或正确答案不在选项中")
                continue
            if len(set(options.values())) != 4 or any(not value for value in options.values()):
                _count_rejection(report, "单选题选项重复或为空")
                continue
        elif options or correct_option:
            _count_rejection(report, "简答题违规携带了选项")
            continue
        if question in seen_questions:
            _count_rejection(report, "题目重复")
            continue
        if any(pattern in question for pattern in VAGUE_QUESTION_PATTERNS):
            _count_rejection(report, "题目过于笼统")
            continue
        # 问号缺失属于格式细节，由代码补全，不因此丢弃整道题。
        if not question.endswith(("？", "?")):
            question = f"{re.sub(r'[。！!；;：:]+$', '', question)}？"

        seen_questions.add(question)
        validated[question_type].append(
            QuizQuestion(
                question_type=question_type,
                question=question,
                options=options,
                correct_option=correct_option,
                reference_answer=reference_answer,
                evidence=evidence,
            )
        )
    return validated


def parse_quiz_questions(raw_content: str, context: str, expected_count: int = 3) -> list[dict]:
    """解析完整小测，固定选出两道单选题和一道简答题。"""
    validated = _parse_valid_question_candidates(raw_content, context)
    selected = validated["single_choice"][:2] + validated["short_answer"][:1]
    if expected_count != 3 or len(selected) < expected_count:
        raise ValueError(
            "AI 没有生成完整的两道单选题和一道简答题，请重新生成；若整篇仍失败，请改选内容明确的具体章节。"
        )
    return [asdict(question) for question in selected]


def _describe_failure(attempts: list[dict], missing: dict[str, int]) -> str:
    """把每次尝试的失败原因汇总成用户看得懂的说明，而不是只说“生成失败”。"""
    shortfall: list[str] = []
    if missing["single_choice"]:
        shortfall.append(f"还缺 {missing['single_choice']} 道单选题")
    if missing["short_answer"]:
        shortfall.append(f"还缺 {missing['short_answer']} 道简答题")

    counts: dict[str, int] = {}
    for attempt in attempts:
        for reason, times in (attempt.get("rejections") or {}).items():
            counts[reason] = counts.get(reason, 0) + times
    detail = "；".join(f"{reason}（{times} 道）" for reason, times in counts.items())

    message = f"AI 连续 {len(attempts)} 次未能生成完整且有原文依据的 3 道题，{'、'.join(shortfall) or '题型不齐'}。"
    if detail:
        message += f" 被丢弃题目的原因：{detail}。"
    message += " 可以直接重试，或改选内容更集中的具体章节。"
    return message


def generate_quiz_questions(
    *,
    source: str,
    chapter: str,
    api_key: str,
    diagnostics: dict | None = None,
) -> list[dict]:
    """先由代码挑选知识点并给原文编号，再让模型只回编号，最后按题型补齐缺题。"""
    full_context = load_quiz_context(source, chapter)
    context = select_quiz_focus(full_context)
    blocks = build_evidence_blocks(context)
    if not blocks:
        raise ValueError("所选章节没有可用于出题的正文。")

    last_error: ValueError | None = None
    collected: dict[str, list[QuizQuestion]] = {"single_choice": [], "short_answer": []}
    attempts: list[dict] = []

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "model_calls": 0,
                "call_elapsed_ms": [],
                "context_chars": len(full_context),
                "focus_chars": len(context),
                "focus_sections": [title for title, _ in split_context_sections(context)],
                "evidence_blocks": len(blocks),
                "attempts": attempts,
            }
        )
    model_invoker = _QuizModelInvoker(api_key, diagnostics)
    block_lines = "\n\n".join(f"[{block_id}] {text}" for block_id, text in blocks)

    for attempt_index in range(QUIZ_MAX_ATTEMPTS):
        missing_choice = max(0, 2 - len(collected["single_choice"]))
        missing_short = max(0, 1 - len(collected["short_answer"]))
        if missing_choice == 0 and missing_short == 0:
            break
        if attempt_index >= 1:
            model_invoker.use_repair_profile()
            # 修复阶段每次只补一道，降低模型同时满足数量、题型和依据约束的难度。
            if missing_choice:
                missing_choice, missing_short = 1, 0
            else:
                missing_choice, missing_short = 0, 1
        existing_questions = [
            question.question
            for question_type in ("single_choice", "short_answer")
            for question in collected[question_type]
        ]
        prompt = f"""
你是个人笔记复习助手。请根据下面的原文片段补全一组小测。
本次只生成：{missing_choice} 道 single_choice、{missing_short} 道 short_answer。
不要重复这些已有问题：{json.dumps(existing_questions, ensure_ascii=False)}

要求：
1. 单选题必须有 A、B、C、D 四个不重复选项和唯一正确答案；简答题不提供选项。不要为了凑题增加原文没有的内容。
   short_answer 的 options 必须是空对象，correct_option 必须是空字符串。
2. 问题必须像真实考试或面试题，例如“Git 是什么？”“Git 的常用命令有哪些？”。
3. 禁止把章节标题机械改写成“主要内容是什么”“包含哪些关键要点”“请解释笔记中的某标题”。
4. 每题必须给出可直接作答的参考答案。错误选项可以来自常见混淆，但不能把无关内容伪装成笔记事实。
5. evidence_ids 只能填写下面原文片段的编号，例如 ["E1"]；每题给 1 到 2 个最能支持答案的编号，不要输出原文。
6. 不得使用原文之外的知识，不确定就不要出这道题。
7. 多道题请尽量依据不同的原文片段，保证知识点互不重复。
8. 只输出 JSON，不要 Markdown，不要解释。格式：
{{
  "questions": [
    {{
      "question_type": "single_choice 或 short_answer",
      "question": "问题？",
      "options": {{"A":"选项A","B":"选项B","C":"选项C","D":"选项D"}},
      "correct_option": "A",
      "reference_answer": "参考答案",
      "evidence_ids": ["E1"]
    }}
  ]
}}

原文片段：
{block_lines}
""".strip()

        report: dict[str, int] = {}
        attempt_record: dict = {
            "attempt": attempt_index + 1,
            "missing_single_choice": missing_choice,
            "missing_short_answer": missing_short,
        }
        attempts.append(attempt_record)

        response = model_invoker.invoke(prompt)
        content = response.content
        if not isinstance(content, str):
            last_error = ValueError("模型返回了不支持的小测内容格式。")
            attempt_record["error"] = str(last_error)
            continue
        try:
            candidates = _parse_valid_question_candidates(content, context, report)
        except ValueError as error:
            last_error = error
            attempt_record["error"] = str(error)
            continue

        attempt_record["rejections"] = report
        existing_set = set(existing_questions)
        accepted = {"single_choice": 0, "short_answer": 0}
        for question_type in ("single_choice", "short_answer"):
            limit = 2 if question_type == "single_choice" else 1
            for question in candidates[question_type]:
                if question.question in existing_set or len(collected[question_type]) >= limit:
                    continue
                collected[question_type].append(question)
                existing_set.add(question.question)
                accepted[question_type] += 1
        attempt_record["accepted"] = accepted

    selected = collected["single_choice"][:2] + collected["short_answer"][:1]
    if len(selected) == 3:
        return [asdict(question) for question in selected]

    missing = {
        "single_choice": max(0, 2 - len(collected["single_choice"])),
        "short_answer": max(0, 1 - len(collected["short_answer"])),
    }
    reason = _describe_failure(attempts, missing)
    if diagnostics is not None:
        diagnostics["failure_reason"] = reason
    raise ValueError(reason) from last_error


def _unanswered_grade(question_index: int) -> dict:
    return {
        "question_index": question_index,
        "score": 0,
        "verdict": "未作答",
        "feedback": "本题未作答，请先尝试用自己的话复述。",
        "evidence": [],
    }


def parse_quiz_grades(raw_content: str, questions: list[dict], answered_indices: list[int]) -> list[dict]:
    """解析模型判分，并验证判分引用只能来自对应题目的笔记依据。"""
    payload = _extract_json(raw_content)
    items = payload.get("grades")
    if not isinstance(items, list):
        raise ValueError("模型返回结果缺少 grades 数组。")

    expected = set(answered_indices)
    grades: dict[int, dict] = {}
    valid_verdicts = {"笔记支持", "部分支持", "与笔记矛盾"}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("question_index"))
            score = int(item.get("score"))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict", "")).strip()
        feedback = str(item.get("feedback", "")).strip()
        raw_evidence = item.get("evidence", [])
        evidence = [str(value).strip() for value in raw_evidence] if isinstance(raw_evidence, list) else []
        evidence = [value for value in evidence if value]
        if index not in expected or index in grades or verdict not in valid_verdicts or not feedback or not evidence:
            continue
        if not 0 <= score <= 100:
            continue
        allowed_evidence = _normalized("\n".join(questions[index]["evidence"]))
        if any(_normalized(quote) not in allowed_evidence for quote in evidence):
            continue
        grades[index] = {
            "question_index": index,
            "score": score,
            "verdict": verdict,
            "feedback": feedback,
            "evidence": evidence,
        }

    if set(grades) != expected:
        raise ValueError("模型判分结果不完整或引用依据无效。")
    return [grades[index] for index in answered_indices]


def parse_quiz_review(raw_content: str, question_count: int) -> list[dict]:
    """解析提交阶段生成的核心考点、易错点和记忆方法。"""
    payload = _extract_json(raw_content)
    items = payload.get("review")
    if not isinstance(items, list):
        raise ValueError("模型返回结果缺少 review 数组。")
    review_by_index: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("question_index"))
        except (TypeError, ValueError):
            continue
        core_point = str(item.get("core_point", "")).strip()
        common_mistake = str(item.get("common_mistake", "")).strip()
        memory_tip = str(item.get("memory_tip", "")).strip()
        if index not in range(question_count) or index in review_by_index:
            continue
        if not all((core_point, common_mistake, memory_tip)):
            continue
        if len({_normalized(core_point), _normalized(common_mistake), _normalized(memory_tip)}) < 3:
            continue
        review_by_index[index] = {
            "question_index": index,
            "core_point": core_point,
            "common_mistake": common_mistake,
            "memory_tip": memory_tip,
        }
    if len(review_by_index) != question_count:
        raise ValueError("模型返回的复习总结不完整。")
    return [review_by_index[index] for index in range(question_count)]


def grade_quiz_answers(*, questions: list[dict], answers: dict[int, str], api_key: str) -> dict:
    """代码判选择题；一次模型调用完成简答评分和三题复习总结。"""
    unanswered = [index for index in range(len(questions)) if not str(answers.get(index, "")).strip()]
    answered_indices = [
        index
        for index in range(len(questions))
        if index not in unanswered and questions[index].get("question_type") == "short_answer"
    ]
    grades_by_index = {index: _unanswered_grade(index) for index in unanswered}
    for index, question in enumerate(questions):
        if index in unanswered or question.get("question_type") != "single_choice":
            continue
        selected_option = str(answers[index]).strip().upper()
        correct_option = str(question.get("correct_option", "")).strip().upper()
        is_correct = selected_option == correct_option
        grades_by_index[index] = {
            "question_index": index,
            "score": 100 if is_correct else 0,
            "verdict": "笔记支持" if is_correct else "与笔记矛盾",
            "feedback": (
                "选择正确。"
                if is_correct
                else f"选择错误，正确答案是 {correct_option}：{question['options'][correct_option]}。"
            ),
            "evidence": question["evidence"],
        }

    grading_items = [
        {
            "question_index": index,
            "question": questions[index]["question"],
            "user_answer": str(answers[index]).strip(),
            "reference_answer": questions[index]["reference_answer"],
            "allowed_evidence": questions[index]["evidence"],
        }
        for index in answered_indices
    ]
    review_items = [
        {
            "question_index": index,
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "allowed_evidence": question["evidence"],
            "user_answer": str(answers.get(index, "")).strip(),
        }
        for index, question in enumerate(questions)
    ]
    prompt = f"""
你是严格但友好的笔记复习助手。本次调用同时完成简答题评分和三道题的复习总结。

判分规则：
- 笔记支持：核心结论完整且没有与笔记冲突的内容，80-100 分。
- 部分支持：答到部分核心结论但有明显遗漏，40-79 分。
- 与笔记矛盾：核心结论错误或与笔记冲突，0-39 分。
- 不评价笔记以外的事实是否正确。
- evidence 只能逐字复制对应题目的 allowed_evidence，不能改写。
- feedback 直接指出答对了什么、缺少什么或冲突在哪里。
- grades 只评价下面待评分的简答题；如果没有待评分简答题则返回空数组。
- review 必须为每道题生成 core_point、common_mistake、memory_tip，三项内容不能重复。
- review 只能根据 reference_answer、allowed_evidence 和用户作答总结，不补充外部事实。
- 只输出 JSON，不要 Markdown。格式：
{{
  "grades":[{{"question_index":2,"score":85,"verdict":"笔记支持","feedback":"具体反馈","evidence":["逐字依据"]}}],
  "review":[{{"question_index":0,"core_point":"核心结论","common_mistake":"易错点","memory_tip":"记忆方法"}}]
}}

待评分简答题：
{json.dumps(grading_items, ensure_ascii=False)}

全部题目复习材料：
{json.dumps(review_items, ensure_ascii=False)}
""".strip()

    model_invoker = _QuizModelInvoker(api_key)
    last_error: ValueError | None = None
    for _ in range(2):
        response = model_invoker.invoke(prompt)
        if not isinstance(response.content, str):
            last_error = ValueError("模型返回了不支持的评分格式。")
            continue
        try:
            for grade in parse_quiz_grades(response.content, questions, answered_indices):
                grades_by_index[grade["question_index"]] = grade
            review = parse_quiz_review(response.content, len(questions))
            return {
                "grades": [grades_by_index[index] for index in range(len(questions))],
                "review": review,
            }
        except ValueError as error:
            last_error = error
    raise ValueError("AI 连续两次未返回可靠的评分结果，请稍后再提交。") from last_error
