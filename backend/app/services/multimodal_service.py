"""DeepSeek Vision、OSS 与可选 Qwen-VL 后备调用。"""

import asyncio
import base64
import io
import json
from hashlib import sha256
from mimetypes import guess_extension
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
from langchain_core.documents import Document

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
COMPRESS_MAX_EDGE = 1600
COMPRESS_TARGET_BYTES = 2 * 1024 * 1024


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


def _maybe_compress_image(image_bytes: bytes, content_type: str) -> bytes:
    """离线缩放大图并尽量压到 2MB；失败时安全回退到原始字节。"""
    try:
        from PIL import Image, ImageOps

        mime = (content_type or "").split(";", 1)[0].lower()
        output_format = {
            "image/jpeg": "JPEG",
            "image/png": "PNG",
            "image/gif": "GIF",
            "image/webp": "WEBP",
        }.get(mime)
        if output_format is None:
            return image_bytes

        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            if max(image.size) <= COMPRESS_MAX_EDGE and len(image_bytes) <= COMPRESS_TARGET_BYTES:
                return image_bytes

            image.thumbnail((COMPRESS_MAX_EDGE, COMPRESS_MAX_EDGE), Image.Resampling.LANCZOS)
            working = image.copy()
            best = image_bytes
            quality = 88
            for _attempt in range(4):
                buffer = io.BytesIO()
                save_options: dict = {"format": output_format, "optimize": True}
                if output_format in {"JPEG", "WEBP"}:
                    if output_format == "JPEG" and working.mode not in {"RGB", "L"}:
                        working = working.convert("RGB")
                    save_options["quality"] = quality
                working.save(buffer, **save_options)
                candidate = buffer.getvalue()
                if len(candidate) < len(best):
                    best = candidate
                if len(candidate) <= COMPRESS_TARGET_BYTES:
                    return candidate
                quality = max(60, quality - 10)
                working = working.resize(
                    (
                        max(1, int(working.width * 0.82)),
                        max(1, int(working.height * 0.82)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            return best
    except Exception:
        return image_bytes


def save_image_to_local(image_bytes: bytes, content_type: str, doc_dir: str | Path) -> str:
    """Validate and save an imported image below the note's local image directory."""
    mime = validate_image(image_bytes, content_type)
    target_dir = Path(doc_dir) / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    extension = guess_extension(mime) or ".bin"
    target_path = target_dir / f"{uuid4().hex}{extension}"
    target_path.write_bytes(image_bytes)
    return str(target_path.resolve())


def image_path_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    content_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(path.suffix.lower())
    if content_type is None:
        raise ImageProcessingError("图片格式不支持，已跳过")
    image_bytes = path.read_bytes()
    mime = validate_image(image_bytes, content_type)
    return image_data_url(image_bytes, mime)


async def build_image_chunk(
    image_bytes: bytes,
    content_type: str,
    doc_dir: str | Path,
    source_path: str | Path,
    api_key: str,
    doc_id: str,
) -> Document:
    """Save one standalone image, describe it with Vision, and build a RAG chunk."""
    local_path: str | None = None
    try:
        validate_image(image_bytes, content_type)
        image_bytes = await asyncio.to_thread(
            _maybe_compress_image, image_bytes, content_type
        )
        local_path = await asyncio.to_thread(
            save_image_to_local, image_bytes, content_type, doc_dir
        )
        description = await describe_image_url(image_path_to_data_url(local_path), api_key)
        source = str(source_path)
        chunk_id = sha256(
            f"{source}:{local_path}:{description}".encode("utf-8")
        ).hexdigest()[:16]
        return Document(
            page_content=description,
            metadata={
                "source": source,
                "image_path": local_path,
                "is_image_chunk": True,
                "standalone_image": True,
                "Header 1": Path(source).name,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            },
        )
    except Exception:
        if local_path:
            Path(local_path).unlink(missing_ok=True)
        raise


async def validate_deepseek_api_key(api_key: str) -> None:
    """Validate a BYOK key without persisting or logging it."""
    url = f"{DEEPSEEK_API_BASE_URL.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT_SECONDS, trust_env=False) as client:
            response = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code in {401, 403}:
            raise InvalidApiKeyError("API Key 无效，请检查后重试")
        response.raise_for_status()
    except InvalidApiKeyError:
        raise
    except (httpx.TimeoutException, httpx.HTTPError) as error:
        raise ImageProcessingError("无法连接 DeepSeek，请稍后重试") from error


async def get_deepseek_balance(api_key: str) -> dict[str, str | bool]:
    """Query the current total balance without persisting or logging the BYOK key."""
    url = f"{DEEPSEEK_API_BASE_URL.rstrip('/')}/user/balance"
    try:
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT_SECONDS, trust_env=False) as client:
            response = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code in {401, 403}:
            raise InvalidApiKeyError("API Key 无效，请检查后重试")
        response.raise_for_status()
        payload = response.json()
        balance_infos = payload.get("balance_infos") or []
        if not isinstance(balance_infos, list) or not balance_infos:
            raise ValueError("missing balance_infos")
        balance = next(
            (item for item in balance_infos if item.get("currency") == "CNY"),
            balance_infos[0],
        )
        return {
            "is_available": bool(payload.get("is_available")),
            "currency": str(balance["currency"]),
            "total_balance": str(balance["total_balance"]),
        }
    except InvalidApiKeyError:
        raise
    except (httpx.TimeoutException, httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise ImageProcessingError("余额查询失败，请稍后重试") from error


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


def _raise_for_vision_response(response: httpx.Response) -> None:
    """把 Vision 常见失败转成安全、可操作的中文提示，不回显服务端原文。"""
    if response.status_code in {401, 403}:
        raise InvalidApiKeyError("API Key无效，请检查后重试")
    if response.status_code == 402:
        raise ImageProcessingError("DeepSeek账户余额不足，充值后再导入图片")
    if response.status_code == 429:
        raise ImageProcessingError("DeepSeek请求过于频繁，请稍后再导入图片")
    if response.status_code == 404:
        raise ImageProcessingError("DeepSeek图片模型暂不可用，请稍后重试")
    if response.status_code == 400:
        raise ImageProcessingError("图片请求被DeepSeek拒绝，请检查图片尺寸或格式")
    if response.status_code >= 500:
        raise ImageProcessingError("DeepSeek图片服务暂时异常，请稍后重试")
    response.raise_for_status()


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
                _raise_for_vision_response(response)
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
        raise ImageProcessingError(
            "图片较大或暂时无法识别，请更换更清晰、较小尺寸的图片后重试"
        ) from error
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
            _raise_for_vision_response(response)
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
            raise ImageProcessingError(
                "图片较大或暂时无法识别，请更换更清晰、较小尺寸的图片后重试"
            ) from error

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
