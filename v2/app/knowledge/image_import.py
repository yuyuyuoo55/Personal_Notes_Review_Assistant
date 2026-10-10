"""Phase 17A：本地 Markdown 图片与单张图片，复用现有 Chroma 更新机制。"""

import io
import json
import os
import re
import warnings
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from ..integrations.vision import (VISION_PROMPT_VERSION, VisionContent, describe_image,
                                   vision_model_name)
from ..rag.vector_retrieval import create_vector_store
from .note_import import import_markdown, save_note_chunks


MAX_IMAGES = 20
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_TOTAL_IMAGE_BYTES = 50 * 1024 * 1024
MAX_MARKDOWN_BYTES = 2 * 1024 * 1024
IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_PATTERN = re.compile(r'!\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^)\n]+)\s*\)')


def resolve_image_path(reference, note_path, allowed_image_roots=None):
    """【业务需求】本地图片必须位于明确允许的目录，禁止越界和网络路径。
    【输入】Markdown 图片引用、笔记 Path；额外允许的本地目录，默认仅笔记目录。
    【执行步骤】1. 规范路径。2. 拒绝 URL/UNC。3. resolve 后核对根目录，含符号链接目标。
    【输出】允许范围内的绝对 Path；非法引用抛出 ValueError。
    """
    note_path = Path(note_path).resolve()
    reference = unquote(reference.strip())
    if reference.startswith("<") and reference.endswith(">"):
        reference = reference[1:-1]
    else:
        reference = re.sub(r'\s+["\'][^"\']*["\']\s*$', "", reference)
    if not reference or "\x00" in reference or reference.startswith(("//", "\\\\")):
        raise ValueError("不允许空图片路径或网络共享路径。")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", reference) and not re.match(r"^[a-zA-Z]:[\\/]", reference):
        raise ValueError("Phase 17A 仅支持本地图片路径。")
    # Windows 笔记常使用反斜杠；在不同测试平台也按本地分隔符解析。
    reference = reference.replace("\\", os.sep)
    image_path = Path(reference)
    if not image_path.is_absolute():
        image_path = note_path.parent / image_path
    image_path = image_path.resolve()
    roots = [note_path.parent]
    for root in allowed_image_roots or []:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError("允许的图片目录不存在。")
        roots.append(root)
    if not any(image_path.is_relative_to(root) for root in roots):
        raise ValueError("图片路径超出允许目录。")
    if str(image_path).startswith(("\\\\", "//")):
        raise ValueError("不允许网络共享图片路径。")
    return image_path


def read_image(path):
    """【业务需求】限制图片体积与像素，读取真实可解码的 PNG/JPEG/WebP。
    【输入】本地图片 Path。
    【执行步骤】1. 有界读取。2. 解码并核对格式/像素。3. 验证文件完整性。
    【输出】(bytes, mime, sha256)；失败抛出安全错误。
    """
    path = Path(path)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("仅支持 PNG、JPG、JPEG、WebP 图片。")
    try:
        with path.open("rb") as file:
            image_bytes = file.read(MAX_IMAGE_BYTES + 1)
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("图片为空或超过 10MB。")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as image:
                expected = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else path.suffix[1:].upper()
                if image.format not in IMAGE_FORMATS or image.format != expected:
                    raise ValueError("图片格式与文件扩展名不符。")
                if image.width * image.height > MAX_IMAGE_PIXELS or max(image.size) > 8192:
                    raise ValueError("图片像素过大。")
                mime = IMAGE_FORMATS[image.format]
                image.verify()
    except ValueError:
        raise
    except Exception:
        raise ValueError("图片不存在、不可读取或文件损坏。") from None
    return image_bytes, mime, sha256(image_bytes).hexdigest()


def _markdown_parts(text):
    """顺序扫描标题与内联图片；代码围栏和行内代码中的图片语法不处理。"""
    parts = []
    headings = {}
    lines = []
    fence = None

    def flush():
        content = "\n".join(lines).strip()
        if content:
            parts.append(("text", content, dict(headings)))
        lines.clear()

    for line in text.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence is not None:
            lines.append(line)
            continue
        heading = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if heading:
            flush()
            level = len(heading[1])
            for deeper in range(level, 4):
                headings.pop(f"Header {deeper}", None)
            headings[f"Header {level}"] = heading[2]
        # 遮蔽行内代码，保留位置，防止误识别示例中的图片语法。
        masked = re.sub(r"(`+).*?\1", lambda match: " " * len(match[0]), line)
        position = 0
        for match in IMAGE_PATTERN.finditer(masked):
            reference = line[match.start(1):match.end(1)].strip()
            prefix = line[position:match.start()]
            if prefix:
                lines.append(prefix)
            flush()
            parts.append(("image", reference, dict(headings)))
            position = match.end()
        lines.append(line[position:])
    flush()
    return parts


def _cache(records, model_name):
    """仅复用同来源、同图片字节、同视觉配置生成的明确描述。"""
    cached = {}
    for metadata in records.get("metadatas", []):
        if metadata.get("vision_model") == model_name and metadata.get("vision_prompt_version") == VISION_PROMPT_VERSION:
            key = (metadata.get("image_path"), metadata.get("image_hash"))
            description = metadata.get("image_description")
            if isinstance(description, str) and description.strip():
                cached[key] = VisionContent(confirmed_content=description,
                    uncertain_details=json.loads(metadata.get("uncertain_details", "[]")))
    return cached


def _image_documents(path, reference, headings, image_hash, description, model_name):
    metadata = {**headings, "image_path": str(path), "image_reference": reference,
                "image_hash": image_hash, "is_image_chunk": True,
                "vision_model": model_name, "vision_prompt_version": VISION_PROMPT_VERSION,
                "image_description": description.confirmed_content,
                "uncertain_details": json.dumps(description.uncertain_details, ensure_ascii=False)}
    # 不确定信息不进入检索正文；引用格式也防止识别文字被误当 Markdown 章节标题。
    content = "图片识别内容（模型提取，请核对原图）：\n" + "\n".join(
        "> " + line for line in description.confirmed_content.splitlines())
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(
        [Document(page_content=content, metadata=metadata)])


def _import_parts(note_path, parts, vector_store, vision, allowed_image_roots):
    """【业务需求】先识别整份图文笔记，避免失败时用不完整描述覆盖旧知识。
    【输入】原文件、按章节排序的文字/图片、Chroma、视觉函数与允许目录。
    【执行步骤】1. 校验所有图片。2. 复用缓存或识别。3. 成功后写新片段并清旧。
    【输出】导入统计及 warning；识别失败不写库，存储异常明确标记未完成。
    """
    result = {"filename": note_path.name, "source": os.path.normcase(str(note_path)),
              "status": "failed", "chunk_count": 0, "image_total": sum(p[0] == "image" for p in parts),
              "image_processed": 0, "image_cached": 0, "vision_calls": 0, "warnings": [], "images": []}
    if result["image_total"] > MAX_IMAGES:
        result["warnings"].append("图片引用超过 20 个，未写入知识库。")
        return result
    # 1. 全部本地路径和图片先检查，避免已知非法输入也消耗模型额度。
    images = {}
    image_number = 0
    total_bytes = 0
    for index, (kind, reference, _) in enumerate(parts):
        if kind != "image":
            continue
        image_number += 1
        try:
            path = resolve_image_path(reference, note_path, allowed_image_roots)
            image_bytes, mime, image_hash = read_image(path)
            total_bytes += len(image_bytes)
            if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("图片总大小超过 50MB。")
            images[index] = (path, image_bytes, mime, image_hash)
        except ValueError as error:
            result["warnings"].append(f"第 {image_number} 张图片：{error}")
    if result["warnings"]:
        result["warnings"].append("本次未写入，原笔记与旧索引保留；请修正后重试。")
        return result
    try:
        if vector_store is None:
            vector_store = create_vector_store()
            # 多模态默认库禁用 Embedding 自动补试；注入的存储由调用方负责其重试预算。
            vector_store._embedding_function.max_retries = 1
        records = vector_store.get(where={"source": result["source"]}, include=["metadatas"])
        model_name = vision_model_name()
        cached = _cache(records, model_name)
        chunks = []
        for index, (kind, content, headings) in enumerate(parts):
            if kind == "text":
                chunks.extend(RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(
                    [Document(page_content=content, metadata=headings)]))
                continue
            path, image_bytes, mime, image_hash = images[index]
            key = (str(path), image_hash)
            if key in cached:
                description = cached[key]
                result["image_cached"] += 1
            else:
                result["vision_calls"] += 1
                description = VisionContent.model_validate(vision(image_bytes, mime))
                if not description.confirmed_content:
                    raise ValueError("图片没有可明确提取的知识，未写入索引。")
                cached[key] = description
            chunks.extend(_image_documents(path, content, headings, image_hash, description, model_name))
            result["image_processed"] += 1
            result["images"].append({"image_path": str(path), "headings": headings,
                                     "uncertain_details": description.uncertain_details})
        # 2. 所有描述成功后统一写入，图片删除会随同来源的旧 ID 清理。
        result.update(save_note_chunks(note_path, chunks, vector_store))
    except Exception:
        result["warnings"].append("图片识别或存储失败，本次未完成；原文件保留，若存储失败请核对索引状态。")
    return result


def import_markdown_images(file_path, vector_store=None, *, vision=None, allowed_image_roots=None):
    """【业务需求】将 Markdown 和本地图片知识按原文顺序导入同一笔记。
    【输入】Markdown 路径、可选 Chroma/Fake Vision、额外允许的图片目录。
    【执行步骤】1. 读 UTF-8 并扫描引用。2. 路径/图片校验。3. 缓存或识别。4. 统一写入清旧。
    【输出】导入状态、Chunk/图片数量、识别次数和明确 warning；失败不静默覆盖旧索引。
    """
    path = Path(file_path).expanduser().resolve()
    try:
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("只支持 Markdown 文件。")
        with path.open("rb") as file:
            raw = file.read(MAX_MARKDOWN_BYTES + 1)
        if len(raw) > MAX_MARKDOWN_BYTES:
            raise ValueError("Markdown 超过 2MB。")
        text = raw.decode("utf-8-sig")
        if not text.strip():
            raise ValueError("Markdown 为空。")
        parts = _markdown_parts(text)
    except (ValueError, OSError):
        return {"status": "failed", "warnings": ["Markdown 不存在、编码无效、为空或超出大小限制。"],
                "filename": path.name, "source": os.path.normcase(str(path)), "chunk_count": 0,
                "image_total": 0, "vision_calls": 0}
    if not any(part[0] == "image" for part in parts):
        try:
            # 无图片时沿用原行为，也会清理上一版已删除的图片描述。
            result = import_markdown(path, vector_store)
            return {**result, "image_total": 0, "image_processed": 0, "image_cached": 0,
                    "vision_calls": 0, "warnings": [], "images": []}
        except Exception:
            return {"status": "failed", "warnings": ["Markdown 写入失败，未完成更新。"],
                    "filename": path.name, "source": os.path.normcase(str(path)), "chunk_count": 0,
                    "image_total": 0, "vision_calls": 0}
    return _import_parts(path, parts, vector_store, vision or describe_image, allowed_image_roots)


def import_image(file_path, vector_store=None, *, vision=None):
    """【业务需求】单张图片作为独立可检索、可出题的笔记。
    【输入】明确选择的图片路径、可选 Chroma/Fake Vision。
    【执行步骤】1. 标题使用文件名。2. 校验/识别或缓存。3. 共用来源与连续 ID 写入。
    【输出】与图文 Markdown 相同的状态；图片没有知识或识别失败时不伪造内容。
    """
    path = Path(file_path).expanduser().resolve()
    parts = [("image", str(path), {"Header 1": path.name})]
    return _import_parts(path, parts, vector_store, vision or describe_image, None)
