"""Markdown 图片安全获取、OSS 上传与描述回填。"""

import asyncio
import base64
import binascii
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from backend.app.core.config import MAX_IMAGE_BYTES, VLM_TIMEOUT_SECONDS
from backend.app.services.multimodal_service import (
    ALLOWED_IMAGE_MIME_TYPES,
    ImageProcessingError,
    InvalidApiKeyError,
    describe_image_url,
    upload_image_to_oss,
    validate_image,
)


MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<source>[^\s)]+)(?:\s+[^)]*)?\)")
DATA_URI_PATTERN = re.compile(r"^data:(?P<mime>image/[^;]+);base64,(?P<data>.+)$", re.I | re.S)


@dataclass
class MarkdownImageResult:
    markdown: str
    image_total: int = 0
    image_processed: int = 0
    image_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def _assert_public_https_url(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ImageProcessingError("仅支持公网 HTTPS 图片")
    return parsed.hostname


def _validate_public_host(hostname: str) -> None:
    try:
        addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except OSError as error:
        raise ImageProcessingError("图片地址无法访问") from error
    if not addresses:
        raise ImageProcessingError("图片地址无法访问")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ImageProcessingError("不允许访问内网或本机图片地址")


async def _download_remote_image(source: str) -> tuple[bytes, str]:
    hostname = _assert_public_https_url(source)
    await asyncio.to_thread(_validate_public_host, hostname)
    try:
        async with httpx.AsyncClient(
            timeout=VLM_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream("GET", source) as response:
                response.raise_for_status()
                declared_size = int(response.headers.get("content-length", "0") or 0)
                if declared_size > MAX_IMAGE_BYTES:
                    raise ImageProcessingError("图片过大，已跳过")
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_IMAGE_BYTES:
                        raise ImageProcessingError("图片过大，已跳过")
                mime = validate_image(bytes(chunks), response.headers.get("content-type"))
                return bytes(chunks), mime
    except ImageProcessingError:
        raise
    except httpx.TimeoutException as error:
        raise ImageProcessingError("图片下载超时，已跳过") from error
    except httpx.HTTPError as error:
        raise ImageProcessingError("图片下载失败，已跳过") from error


def _decode_data_uri(source: str) -> tuple[bytes, str]:
    match = DATA_URI_PATTERN.match(source)
    if match is None:
        raise ImageProcessingError("图片 data URI 格式无效")
    mime = match.group("mime").lower()
    if mime not in ALLOWED_IMAGE_MIME_TYPES:
        raise ImageProcessingError("图片格式不支持，已跳过")
    try:
        image_bytes = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, binascii.Error) as error:
        raise ImageProcessingError("图片 data URI 无法解析") from error
    validate_image(image_bytes, mime)
    return image_bytes, mime


async def enrich_markdown_images(markdown: str, api_key: str) -> MarkdownImageResult:
    """逐图增强 Markdown；单图失败只记录安全 warning，不中断文档。"""
    matches = list(MARKDOWN_IMAGE_PATTERN.finditer(markdown))
    result = MarkdownImageResult(markdown=markdown, image_total=len(matches))
    if not matches:
        return result

    replacements: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches, start=1):
        source = match.group("source").strip("<>")
        try:
            if source.lower().startswith("data:image/"):
                image_bytes, mime = _decode_data_uri(source)
            elif source.lower().startswith("https://"):
                image_bytes, mime = await _download_remote_image(source)
            else:
                raise ImageProcessingError("本地图片未随 Markdown 上传，已跳过")

            public_url = await asyncio.to_thread(upload_image_to_oss, image_bytes, mime)
            description = await describe_image_url(public_url, api_key)
            replacement = f"{match.group(0)}\n\n> 图片内容：{description}"
            replacements.append((match.start(), match.end(), replacement))
            result.image_processed += 1
        except InvalidApiKeyError as error:
            # 用户 Key 无效：后续 RAG 问答同样依赖该 Key，继续导入没有意义，
            # 且该错误不允许回退到部署者付费模型。直接中止整个导入，明确提示用户。
            raise InvalidApiKeyError(
                "DeepSeek API Key 无效，笔记导入已中止。请检查 API Key 后重试。"
            ) from error
        except ImageProcessingError as error:
            result.image_skipped += 1
            result.warnings.append(f"第 {index} 张图片：{error}")

    enriched = markdown
    for start, end, replacement in reversed(replacements):
        enriched = enriched[:start] + replacement + enriched[end:]
    result.markdown = enriched
    return result
