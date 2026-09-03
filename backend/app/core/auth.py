"""BYOK 请求头校验；用户密钥只在当前请求内存中使用。"""

from typing import Annotated

from fastapi import Header, HTTPException, status


DEEPSEEK_API_KEY_HEADER = "X-DeepSeek-API-Key"


def require_user_deepseek_api_key(
    api_key: Annotated[str | None, Header(alias=DEEPSEEK_API_KEY_HEADER)] = None,
) -> str:
    """读取用户自带 Key；不记录、不持久化，也不放入业务 DTO。"""
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先输入API Key",
        )
    return cleaned_key
