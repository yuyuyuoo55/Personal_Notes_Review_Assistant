"""小测的章节、模型输出、内部答案和对外题目结构。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class QuizData(BaseModel):
    """拒绝未声明字段，避免把内部答案误当作公共展示字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QuestionCounts(QuizData):
    """用户请求题量；布尔值、浮点数和字符串不能冒充整数。"""

    single_choice: StrictInt = Field(ge=0, le=5)
    short_answer: StrictInt = Field(ge=0, le=3)

    @model_validator(mode="after")
    def check_total(self):
        if self.single_choice + self.short_answer == 0:
            raise ValueError("至少选择一道题。")
        return self


class ChapterChunk(QuizData):
    """保留真实 Chunk ID、原文、来源、标题和原文顺序。"""

    chunk_id: str
    content: str
    source: str
    headings: dict[str, str]
    chunk_index: int


class ChapterResult(QuizData):
    scope: Literal["chapter", "note"] = "chapter"
    coverage: dict[str, int | bool] = Field(default_factory=dict)
    status: Literal["ok", "note_not_found", "chapter_not_found", "empty_chapter",
                    "ambiguous_chapter", "invalid_metadata", "storage_error"]
    message: str
    source: str
    headings: dict[str, str]
    chunks: list[ChapterChunk] = Field(default_factory=list)
    content: str = ""


class Evidence(QuizData):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, description="对应 Chunk 中逐字复制的答案依据")


class GeneratedQuestion(QuizData):
    """模型生成的完整题目，仅保存在后端。"""

    question_type: Literal["single_choice", "short_answer"]
    question: str = Field(min_length=1)
    knowledge_point: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    options: dict[str, str]
    correct_answer: str
    explanation: str
    reference_answer: str
    key_points: list[str]

    @model_validator(mode="after")
    def check_question_fields(self):
        if len(set(self.source_chunk_ids)) != len(self.source_chunk_ids):
            raise ValueError("来源 ID 不能重复。")
        if self.question_type == "single_choice":
            if set(self.options) != {"A", "B", "C", "D"}:
                raise ValueError("选择题必须有 A/B/C/D 四个选项。")
            values = [value.strip().casefold() for value in self.options.values()]
            if not all(values) or len(set(values)) != 4:
                raise ValueError("选项不能为空或重复。")
            if self.correct_answer not in self.options or not self.explanation.strip():
                raise ValueError("选择题答案或解析无效。")
            if self.reference_answer or self.key_points:
                raise ValueError("选择题不能混入问答题字段。")
        else:
            if self.options or self.correct_answer or self.explanation:
                raise ValueError("问答题不能混入选择题字段。")
            if not self.reference_answer.strip() or not self.key_points:
                raise ValueError("问答题需要参考答案和得分点。")
            if not all(point.strip() for point in self.key_points):
                raise ValueError("得分点不能为空。")
        return self


class GeneratedQuiz(QuizData):
    questions: list[GeneratedQuestion]
    insufficient_reason: str = Field(description="资料不足时说明原因；不要为凑数量编题")


class ReviewDecision(QuizData):
    candidate_id: str
    approved: StrictBool
    reason: str


class QualityReview(QuizData):
    decisions: list[ReviewDecision]


class PrivateQuestion(GeneratedQuestion):
    question_id: str


class PublicQuestion(QuizData):
    """显式白名单；不包含知识点、依据、解析或任何标准答案。"""

    question_id: str
    question_type: Literal["single_choice", "short_answer"]
    question: str
    options: dict[str, str]


class StoredQuiz(QuizData):
    quiz_id: str
    chapter: ChapterResult
    requested: QuestionCounts
    questions: list[PrivateQuestion]


class QuizResult(QuizData):
    scope: Literal["chapter", "note"] = "chapter"
    coverage: dict[str, int | bool] = Field(default_factory=dict)
    status: Literal["ready", "partial", "insufficient_content", "chapter_unavailable",
                    "invalid_request", "model_error", "invalid_model_output"]
    message: str
    quiz_id: str | None = None
    source: str = ""
    headings: dict[str, str] = Field(default_factory=dict)
    requested: QuestionCounts | None = None
    actual: dict[str, int] = Field(default_factory=lambda: {"single_choice": 0, "short_answer": 0})
    questions: list[PublicQuestion] = Field(default_factory=list)
    shortage_reason: str = ""
    model_calls: int = 0
    chapter_status: str | None = None


# Phase 16B：提交结构不接收前端所谓标准答案；题型以保存的小测为准。
class SubmittedAnswer(QuizData):
    question_id: str = Field(min_length=1)
    user_answer: str = Field(min_length=1, max_length=10000)
    question_type: Literal["single_choice", "short_answer"] | None = None


class AnswerSubmission(SubmittedAnswer):
    quiz_id: str = Field(min_length=1)


class QuizSubmission(QuizData):
    quiz_id: str = Field(min_length=1)
    answers: list[SubmittedAnswer] = Field(min_length=1, max_length=8)


GradeLevel = Literal["优秀", "良好", "中等", "及格", "不及格"]


class ShortAnswerAssessment(QuizData):
    """模型只输出评价；题目身份、用户答案和参考答案由后端组装。"""

    grade: GradeLevel
    feedback: str = Field(min_length=1)
    correct_points: list[str]
    missing_points: list[str]
    suggestion: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_points(self):
        if any(not point.strip() for point in self.correct_points + self.missing_points):
            raise ValueError("评价知识点不能为空。")
        if self.grade == "优秀" and (self.missing_points or not self.correct_points):
            raise ValueError("优秀评价必须有正确点且没有明显遗漏。")
        if self.grade != "优秀" and not self.missing_points:
            raise ValueError("非优秀评价必须说明具体遗漏或错误。")
        return self


class ChoiceGrade(QuizData):
    question_id: str
    question_type: Literal["single_choice"] = "single_choice"
    user_answer: str
    correct: StrictBool
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str


class ShortAnswerGrade(ShortAnswerAssessment):
    question_id: str
    question_type: Literal["short_answer"] = "short_answer"
    user_answer: str
    reference_answer: str


QuestionGrade = ChoiceGrade | ShortAnswerGrade


class AnswerResult(QuizData):
    status: Literal["graded", "already_graded", "in_progress", "invalid_request", "quiz_not_found",
                    "question_not_found", "submission_conflict", "model_error", "invalid_model_output"]
    message: str
    quiz_id: str = ""
    question_id: str = ""
    result: QuestionGrade | None = None


class GroundedPoint(QuizData):
    """报告知识内容保留可校验的原文依据；类比不能当作原笔记事实。"""

    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class ReviewSuggestion(QuizData):
    focus: Literal["巩固", "加强"]
    question_ids: list[str] = Field(min_length=1)
    suggestion: str = Field(min_length=1)


class ReportContent(QuizData):
    """模型只生成有依据的知识内容，不让它编造计数或用户实际错误。"""

    core_points: list[GroundedPoint] = Field(min_length=1)
    possible_confusions: list[GroundedPoint]
    memory_methods: list[GroundedPoint] = Field(min_length=1)
    review_suggestions: list[ReviewSuggestion] = Field(min_length=1)


class ActualMistake(QuizData):
    question_id: str
    knowledge_point: str
    description: str


class QuestionRating(QuizData):
    question_id: str
    grade: GradeLevel


class StudyOverview(QuizData):
    scope: Literal["chapter", "note"] = "chapter"
    coverage: dict[str, int | bool] = Field(default_factory=dict)
    source: str
    headings: dict[str, str]
    actual_question_count: int
    single_choice_count: int
    single_choice_correct: int
    short_answer_ratings: list[QuestionRating]


class ChapterReport(ReportContent):
    overview: StudyOverview
    actual_mistakes: list[ActualMistake]


class QuizProgress(QuizData):
    """仅后端可见；先保存提交，再保存批改，失败时允许原答案重试。"""

    answers: dict[str, str] = Field(default_factory=dict)
    grades: dict[str, QuestionGrade] = Field(default_factory=dict)
    active_questions: list[str] = Field(default_factory=list)
    report_active: bool = False
    report: ChapterReport | None = None


class QuizCompletionResult(QuizData):
    status: Literal["completed", "incomplete", "in_progress", "partial_failure", "invalid_request",
                    "quiz_not_found", "question_not_found", "submission_conflict", "model_error", "invalid_model_output"]
    message: str
    quiz_id: str = ""
    results: list[QuestionGrade] = Field(default_factory=list)
    pending_question_ids: list[str] = Field(default_factory=list)
    failed_question_ids: list[str] = Field(default_factory=list)
    report: ChapterReport | None = None
