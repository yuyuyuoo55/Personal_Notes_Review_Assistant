"""章节小测路由：只投影公共问题与已批改结果。"""
from fastapi import APIRouter, Request

from ..quiz.service import generate_quiz
from ..quiz.grading import submit_answer, submit_quiz
from ..quiz.schemas import AnswerSubmission, QuizSubmission, QuizCompletionResult, QuizResult, AnswerResult
from .runtime import ApiError, response, run_job, runtime_from, LazyQuizModel
from .schemas import QuizRequest, Envelope

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/generate", response_model=Envelope[QuizResult])
async def generate(request: Request, payload: QuizRequest):
    runtime = runtime_from(request)
    def execute():
        return generate_quiz(runtime.store(), payload.source, payload.headings,
                             payload.single_choice, payload.short_answer,
                             model=LazyQuizModel(runtime.model_factory), quiz_store=runtime.quiz_store, scope=payload.scope)
    return response(await run_job(runtime, execute))


@router.post("/answer", response_model=Envelope[AnswerResult])
async def answer(request: Request, payload: AnswerSubmission):
    runtime = runtime_from(request)
    def execute():
        model = LazyQuizModel(runtime.model_factory)
        return submit_answer(payload, model=model, quiz_store=runtime.quiz_store)
    return response(await run_job(runtime, execute))


@router.post("/submit", response_model=Envelope[QuizCompletionResult])
async def submit(request: Request, payload: QuizSubmission):
    runtime = runtime_from(request)
    def execute():
        return submit_quiz(payload, model=LazyQuizModel(runtime.model_factory), quiz_store=runtime.quiz_store)
    return response(await run_job(runtime, execute))


@router.get("/result/{quiz_id}", response_model=Envelope[QuizCompletionResult])
async def result(request: Request, quiz_id: str):
    """【业务需求】无模型副作用查询已有结果，不提前公开未批改答案。
    【输入】小测ID。
    【执行步骤】读取后端副本，仅投影已保存grades、pending IDs和已保存report。
    【输出】公共QuizCompletionResult；不返回StoredQuiz、原始答案或内部进度对象。
    """
    runtime = runtime_from(request)
    def read():
        quiz = runtime.quiz_store.get(quiz_id)
        if quiz is None:
            raise ApiError(404, "quiz_not_found", "小测不存在或服务重启后已丢失。")
        progress = runtime.quiz_store.get_progress(quiz_id)
        status = "completed" if progress.report is not None else "in_progress" if progress.report_active or progress.active_questions else "incomplete"
        return QuizCompletionResult(status=status, message="只返回已保存结果，不自动生成报告。", quiz_id=quiz_id,
            results=[progress.grades[q.question_id] for q in quiz.questions if q.question_id in progress.grades],
            pending_question_ids=[q.question_id for q in quiz.questions if q.question_id not in progress.grades],
            report=progress.report)
    return response(await run_job(runtime, read))
