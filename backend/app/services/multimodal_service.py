"""DeepSeek Vision、OSS 与可选 Qwen-VL 后备调用。"""

import asyncio
import base64
import json
from mimetypes import guess_extension
from urllib.parse import quote
from uuid import uuid4

import httpx

from backend.app.core.config import (
    DASHSCOPE_API_KEY,
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_VISION_MODEL,
    MAX_IMAGE_BYTES,
    OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET,
    OSS_BUCKET_NAME,
    OSS_ENDPOINT,
    OSS_OBJECT_PREFIX,
    OSS_PUBLIC_BASE_URL,
    OSS_URL_EXPIRES_SECONDS,
    QWEN_VL_MODEL,
    VLM_FALLBACK_TO_QWEN,
    VLM_TIMEOUT_SECONDS,
)


ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class ImageProcessingError(RuntimeError):
    """可安全展示给用户的图片处理错误，不包含底层凭据信息。"""


class InvalidApiKeyError(ImageProcessingError):
    """用户 Key 无效；该错误不能回退为部署者付费模型。"""


def validate_image(image_bytes: bytes, content_type: str | None) -> str:
    """限制图片类型和大小；返回可用于 data URL/OSS 的 MIME。"""
    if not image_bytes:
        raise ImageProcessingError("图片内容为空")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageProcessingError("图片过大，已跳过")

    mime = (content_type or "").split(";", 1)[0].lower()
    if mime not in ALLOWED_IMAGE_MIME_TYPES:
        raise ImageProcessingError("仅支持 JPEG、PNG、GIF 或 WebP 图片")

    signatures_match = {
        "image/jpeg": image_bytes.startswith(b"\xff\xd8\xff"),
        "image/png": image_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/gif": image_bytes.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": (
            len(image_bytes) >= 12
            and image_bytes.startswith(b"RIFF")
            and image_bytes[8:12] == b"WEBP"
        ),
    }
    if not signatures_match[mime]:
        raise ImageProcessingError("图片内容与文件类型不匹配")
    return mime


def image_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _deepseek_payload(question: str, image_url: str, *, stream: bool) -> dict:
    return {
        "model": DEEPSEEK_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "stream": stream,
    }


def stream_deepseek_image_answer(
    *, question: str, image_bytes: bytes, content_type: str, api_key: str
):
    """使用用户 BYOK Key 流式理解聊天图片；不经过 RAG。"""
    mime = validate_image(image_bytes, content_type)
    url = f"{DEEPSEEK_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=VLM_TIMEOUT_SECONDS, trust_env=False) as client:
            with client.stream(
                "POST",
                url,
                headers=headers,
                json=_deepseek_payload(question, image_data_url(image_bytes, mime), stream=True),
            ) as response:
                if response.status_code in {401, 403}:
                    raise InvalidApiKeyError("API Key无效，请检查后重试")
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw_data = line.removeprefix("data:").strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        payload = json.loads(raw_data)
                        content = payload["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                        continue
                    if content:
                        yield content
    except ImageProcessingError:
        raise
    except httpx.TimeoutException as error:
        raise ImageProcessingError("图片识别超时，请稍后重试") from error
    except httpx.HTTPError as error:
        raise ImageProcessingError("图片识别失败，请稍后重试") from error


async def describe_image_url(image_url: str, api_key: str) -> str:
    """优先用用户 DeepSeek Key 描述 OSS 图片，按配置选择 Qwen-VL 后备。"""
    prompt = "请准确描述这张笔记图片中的文字、图表和关键含义，输出简洁中文描述。"
    url = f"{DEEPSEEK_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT_SECONDS, trust_env=False) as client:
            response = await client.post(
                url,
                headers=headers,
                json=_deepseek_payload(prompt, image_url, stream=False),
            )
            if response.status_code in {401, 403}:
                raise InvalidApiKeyError("API Key无效，图片已跳过")
            response.raise_for_status()
            description = response.json()["choices"][0]["message"]["content"]
            if not description:
                raise ImageProcessingError("图片识别未返回描述")
            return str(description).strip()
    except InvalidApiKeyError:
        raise
    except ImageProcessingError:
        if not VLM_FALLBACK_TO_QWEN:
            raise
    except (httpx.TimeoutException, httpx.HTTPError, KeyError, IndexError, ValueError) as error:
        if not VLM_FALLBACK_TO_QWEN:
            raise ImageProcessingError("图片识别失败或超时，已跳过") from error

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_describe_with_qwen, image_url, prompt),
            timeout=VLM_TIMEOUT_SECONDS,
        )
    except TimeoutError as error:
        raise ImageProcessingError("Qwen-VL识别超时，图片已跳过") from error


def _describe_with_qwen(image_url: str, prompt: str) -> str:
    """部署者可显式开启的 Qwen-VL 后备；消耗部署者 DashScope 额度。"""
    if not DASHSCOPE_API_KEY:
        raise ImageProcessingError("Qwen-VL后备未配置，图片已跳过")
    try:
        from dashscope import MultiModalConversation

        response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"image": image_url}, {"text": prompt}],
                }
            ],
        )
        if getattr(response, "status_code", None) != 200:
            raise ImageProcessingError("Qwen-VL识别失败，图片已跳过")
        content = response.output.choices[0].message.content
        if isinstance(content, list):
            text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "".join(text_parts).strip()
        return str(content).strip()
    except ImageProcessingError:
        raise
    except Exception as error:
        raise ImageProcessingError("Qwen-VL识别失败，图片已跳过") from error


def upload_image_to_oss(image_bytes: bytes, content_type: str) -> str:
    """使用部署者 OSS 凭据上传图片并返回公网 URL。"""
    if not all([OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET_NAME]):
        raise ImageProcessingError("OSS未配置，图片已跳过")
    try:
        import oss2

        extension = guess_extension(content_type) or ".bin"
        object_name = f"{OSS_OBJECT_PREFIX}/{uuid4().hex}{extension}"
        bucket = oss2.Bucket(
            oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET),
            OSS_ENDPOINT,
            OSS_BUCKET_NAME,
            connect_timeout=VLM_TIMEOUT_SECONDS,
        )
        result = bucket.put_object(
            object_name,
            image_bytes,
            headers={"Content-Type": content_type},
        )
        if result.status not in {200, 201}:
            raise ImageProcessingError("OSS上传失败，图片已跳过")

        base_url = (OSS_PUBLIC_BASE_URL or "").rstrip("/")
        if base_url:
            return f"{base_url}/{quote(object_name, safe='/')}"
        # 私有 Bucket 默认返回短时签名 URL，VLM 可公网拉取但不会永久公开图片。
        return bucket.sign_url("GET", object_name, OSS_URL_EXPIRES_SECONDS)
    except ImageProcessingError:
        raise
    except Exception as error:
        raise ImageProcessingError("OSS上传失败，图片已跳过") from error
