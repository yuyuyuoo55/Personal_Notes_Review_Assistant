"""最简单的进程内存储；重启后丢失，不提供用户管理或持久化。"""

from copy import deepcopy
from secrets import token_urlsafe
from threading import Lock

from .schemas import (ChapterReport, ChapterResult, PrivateQuestion, QuestionCounts,
                      QuestionGrade, QuizProgress, StoredQuiz)


class QuizStore:
    """独立小测存储对象；只有后端业务代码可以访问完整答案。"""

    def __init__(self):
        self._quizzes: dict[str, StoredQuiz] = {}
        self._progress: dict[str, QuizProgress] = {}
        self._lock = Lock()

    def save(self, chapter: ChapterResult, requested: QuestionCounts,
             questions: list[PrivateQuestion]) -> str:
        """【业务需求】把完整小测与答案留在后端。
        【输入】真实章节、请求题量、已验证题目。
        【执行步骤】1. 生成随机 ID。2. 复制保存数据。
        【输出】不可随意猜测的小测 ID。
        """
        quiz_id = token_urlsafe(32)
        quiz = StoredQuiz(
            quiz_id=quiz_id, chapter=deepcopy(chapter), requested=deepcopy(requested),
            questions=deepcopy(questions),
        )
        with self._lock:
            self._quizzes[quiz_id] = quiz
            self._progress[quiz_id] = QuizProgress()
        return quiz_id

    def get(self, quiz_id: str) -> StoredQuiz | None:
        """【业务需求】供后续后端批改取回标准答案，不相信前端答案。
        【输入】小测 ID。
        【执行步骤】1. 查找小测。2. 返回副本，防止外部修改保存状态。
        【输出】完整内部小测或 None；禁止将该对象直接返回前端。
        """
        with self._lock:
            quiz = self._quizzes.get(quiz_id)
            return deepcopy(quiz) if quiz is not None else None

    def get_progress(self, quiz_id: str) -> QuizProgress | None:
        """返回内部进度副本；不得直接作为公共响应。"""
        with self._lock:
            progress = self._progress.get(quiz_id)
            return deepcopy(progress) if progress is not None else None

    def claim_answer(self, quiz_id: str, question_id: str, user_answer: str):
        """【业务需求】防止相同题目并发批改和重复提交混乱。
        【输入】已校验的小测 ID、题目 ID、规范化用户答案。
        【执行步骤】1. 锁内检查旧答案与结果。2. 标记本题正在批改。
        【输出】claimed/cached/conflict/busy 和已保存结果；模型调用在锁外完成。
        """
        with self._lock:
            progress = self._progress[quiz_id]
            if question_id in progress.answers and progress.answers[question_id] != user_answer:
                return "conflict", None
            if question_id in progress.grades:
                return "cached", deepcopy(progress.grades[question_id])
            if question_id in progress.active_questions:
                return "busy", None
            progress.answers[question_id] = user_answer
            progress.active_questions.append(question_id)
            return "claimed", None

    def finish_answer(self, quiz_id: str, question_id: str, result: QuestionGrade | None):
        """【业务需求】保存真实批改结果，失败时解除正在批改标记。
        【输入】已占用的题目和批改结果；None 表示失败。
        【执行步骤】1. 保存成功结果副本。2. 解除占用但保留原提交答案。
        【输出】无；相同答案可重试失败题，成功题不重复调用模型。
        """
        with self._lock:
            progress = self._progress[quiz_id]
            if result is not None:
                progress.grades[question_id] = deepcopy(result)
            progress.active_questions.remove(question_id)

    def claim_report(self, quiz_id: str):
        """【业务需求】只为全部题目批改成功的小测生成一份报告。
        【输入】有效小测 ID。
        【执行步骤】1. 检查全部结果。2. 返回缓存或锁内占用报告任务。
        【输出】incomplete/cached/busy/claimed 和进度副本。
        """
        with self._lock:
            progress = self._progress[quiz_id]
            ids = {question.question_id for question in self._quizzes[quiz_id].questions}
            if not ids or set(progress.grades) != ids or progress.active_questions:
                return "incomplete", deepcopy(progress)
            if progress.report is not None:
                return "cached", deepcopy(progress)
            if progress.report_active:
                return "busy", deepcopy(progress)
            progress.report_active = True
            return "claimed", deepcopy(progress)

    def finish_report(self, quiz_id: str, report: ChapterReport | None):
        """【业务需求】报告成功才保存，失败保持未完成并允许重试。
        【输入】有效小测 ID、已校验报告或 None。
        【执行步骤】1. 保存成功报告副本。2. 解除生成标记。
        【输出】无；不会把模型失败伪装成完成。
        """
        with self._lock:
            progress = self._progress[quiz_id]
            if report is not None:
                progress.report = deepcopy(report)
            progress.report_active = False


DEFAULT_QUIZ_STORE = QuizStore()
