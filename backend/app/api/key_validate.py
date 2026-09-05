"""Validate request-scoped DeepSeek credentials."""

from fastapi import APIRouter, Depends

from backend.app.core.auth import require_user_deepseek_api_key
from backend.app.services.multimodal_service import (
    ImageProcessingError,
    InvalidApiKeyError,
    validate_deepseek_api_key,
)

router = APIRouter(prefix="/api/key", tags=["key"])


@router.post("/validate")
async def validate_key(
    api_key: str = Depends(require_user_deepseek_api_key),
) -> dict[str, bool | str]:
    try:
        await validate_deepseek_api_key(api_key)
        return {"valid": True, "message": "您的 DeepSeek API Key 有效，可以使用"}
    except InvalidApiKeyError:
        return {"valid": False, "message": "API Key 无效，请检查后重试"}
    except ImageProcessingError:
        return {"valid": False, "message": "无法连接 DeepSeek，请稍后重试"}
