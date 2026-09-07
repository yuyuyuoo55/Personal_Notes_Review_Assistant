"""Read a Markdown + images ZIP bundle without extracting it to disk."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from zipfile import BadZipFile, ZipFile

from backend.app.services.multimodal_service import ALLOWED_IMAGE_MIME_TYPES


MAX_BUNDLE_FILES = 500
MAX_BUNDLE_UNCOMPRESSED_BYTES = 80 * 1024 * 1024
IMAGE_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class MarkdownBundle:
    markdown_name: str
    markdown: str
    local_images: dict[str, tuple[bytes, str]]


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and re.fullmatch(r"[A-Za-z]:", path.parts[0]))
    ):
        raise ValueError("ZIP 中包含不安全的文件路径")
    return PurePosixPath(*[part for part in path.parts if part != "."]).as_posix()


def read_markdown_bundle(content: bytes) -> MarkdownBundle:
    """Return the single Markdown document and its bundled image bytes."""
    try:
        with ZipFile(BytesIO(content)) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            if len(files) > MAX_BUNDLE_FILES:
                raise ValueError(f"ZIP 文件过多，最多支持 {MAX_BUNDLE_FILES} 个文件")
            if sum(item.file_size for item in files) > MAX_BUNDLE_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP 解压后过大，请控制在 80MB 以内")
            if any(item.flag_bits & 0x1 for item in files):
                raise ValueError("暂不支持加密 ZIP")

            members = {_safe_member_name(item.filename): item for item in files}
            markdown_members = [name for name in members if Path(name).suffix.lower() == ".md"]
            if len(markdown_members) != 1:
                raise ValueError("ZIP 中必须且只能包含一个 Markdown 文件")

            markdown_member = markdown_members[0]
            try:
                markdown = archive.read(members[markdown_member]).decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("ZIP 中的 Markdown 必须使用 UTF-8 编码") from error

            markdown_dir = PurePosixPath(markdown_member).parent
            local_images: dict[str, tuple[bytes, str]] = {}
            basename_counts: dict[str, int] = {}
            image_entries: list[tuple[str, bytes, str]] = []
            for name, item in members.items():
                mime = IMAGE_MIME_BY_SUFFIX.get(Path(name).suffix.lower())
                if mime not in ALLOWED_IMAGE_MIME_TYPES:
                    continue
                image_bytes = archive.read(item)
                relative_name = PurePosixPath(name).relative_to(markdown_dir).as_posix() if PurePosixPath(name).is_relative_to(markdown_dir) else name
                image_entries.append((relative_name, image_bytes, mime))
                basename = PurePosixPath(name).name
                basename_counts[basename] = basename_counts.get(basename, 0) + 1

            for relative_name, image_bytes, mime in image_entries:
                local_images[relative_name] = (image_bytes, mime)
                basename = PurePosixPath(relative_name).name
                if basename_counts.get(basename) == 1:
                    local_images[basename] = (image_bytes, mime)

            return MarkdownBundle(
                markdown_name=PurePosixPath(markdown_member).name,
                markdown=markdown,
                local_images=local_images,
            )
    except BadZipFile as error:
        raise ValueError("ZIP 文件损坏或格式不正确") from error
