"""FastAPI 入口：当前只提供本地健康检查与占位笔记列表。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.notes import router as notes_router
from backend.app.api.chat import router as chat_router
from backend.app.api.key_validate import router as key_router

app = FastAPI(
    title="个人笔记复习助手 API",
    version="0.1.0",
    description="本地开发阶段的后端接口。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 将 notes.py 中的 APIRouter 注册到应用；相当于让 Spring Boot 挂载 Controller。
app.include_router(notes_router)
app.include_router(chat_router)
app.include_router(key_router)


@app.get("/api/health", tags=["system"])
def health_check() -> dict[str, str]:
    """最小健康检查，类似 Spring Boot 中用于确认服务存活的 GET 接口。"""
    return {"status": "ok", "services": "personal-notes-review-assistant"}
