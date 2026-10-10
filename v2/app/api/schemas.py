"""API 请求结构；公共小测响应沿用现有Schema。"""
from typing import Literal, Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequestData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ChatMessage(RequestData):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class ChatRequest(RequestData):
    question: str = Field(min_length=1, max_length=10000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    web_enabled: bool = False


class QuizRequest(RequestData):
    scope: Literal["note", "chapter"] = "chapter"
    source: str = Field(min_length=1, max_length=2048)
    headings: dict[str, str] = Field(default_factory=dict, max_length=3)
    single_choice: int = Field(default=1, ge=0, le=5)
    short_answer: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def check_request(self):
        if self.scope == "note" and self.headings:
            raise ValueError("整篇模式不能指定章节。")
        if self.single_choice + self.short_answer < 1:
            raise ValueError("至少一道题。")
        if any(key not in {"Header 1", "Header 2", "Header 3"} or not value.strip()
               for key, value in self.headings.items()):
            raise ValueError("章节标题格式无效。")
        return self


Data = TypeVar("Data")


class Envelope(BaseModel, Generic[Data]):
    status: str
    data: Data
