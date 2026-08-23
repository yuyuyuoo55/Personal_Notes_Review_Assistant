from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """前端发送给问答接口的请求 DTO。"""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="用户输入的原始问题",
    )
    mode: Literal["fast", "accurate"] = Field(
        default="fast",
        description="检索模式：fast 只走向量检索；accurate 走完整混合检索链路。",
    )
    conversation_id: str = Field(
        min_length=1,
        max_length=64,
        description="浏览器会话标识，仅快速模式用于读取和更新内存记忆。",
    )
