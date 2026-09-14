from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """前端发送给问答接口的请求 DTO。"""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="用户输入的原始问题",
    )
    mode: Literal["unified", "fast", "accurate"] = Field(
        default="unified",
        description="统一检索模式；fast/accurate 仅用于兼容旧客户端，处理链路相同。",
    )
    conversation_id: str = Field(
        min_length=1,
        max_length=64,
        description="浏览器会话标识，当前版本仅为兼容旧客户端保留。",
    )
