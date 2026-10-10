"""Phase17B：安全读取图文 ZIP，复用现有图片描述与笔记更新流程。"""

import os
import posixpath
import re
import stat
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..integrations.vision import VisionContent, describe_image, vision_model_name
from ..rag.vector_retrieval import create_vector_store
from .image_import import (_cache, _image_documents, _markdown_parts, read_image,
                           IMAGE_SUFFIXES, MAX_IMAGE_BYTES, MAX_MARKDOWN_BYTES,
                           MAX_TOTAL_IMAGE_BYTES)
from .note_import import save_note_chunks

class ZipImportError(ValueError):
    """只携带本地校验产生的安全错误说明，不包含模型响应或密钥。"""


MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 200 * 1024 * 1024
MAX_ZIP_FILES = 512
MAX_ZIP_IMAGES = 100
HTML_IMAGE = re.compile(r'<img\b[^>]*\bsrc\s*=\s*(["\'])(.*?)\1[^>]*>', re.I)


def _member_name(name):
    """校验 ZIP 成员名，统一分隔符，不接受 Windows 特殊路径。"""
    name = name.replace("\\", "/")
    parts = name.rstrip("/").split("/")
    if not name or name.startswith("/") or any(
        part in {"", ".", ".."} or ":" in part or "\x00" in part
        or any(c in part for c in '<>"|?*') or part.rstrip(". ") != part
        or re.match(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)", part, re.I)
        for part in parts
    ):
        raise ZipImportError("ZIP 成员路径不安全。")
    return name.rstrip("/")


def inspect_zip(file_path):
    """【业务需求】在读取内容前检查 ZIP 安全性与完整资源清单。
    【输入】本地 ZIP 路径。
    【执行步骤】限制包体/条目/展开大小；拒绝危险路径、重复、特殊文件和异常压缩比。
    【输出】成员清单与统计；非法包抛出 ValueError，不解压或调用模型。
    """
    path = Path(file_path).expanduser().resolve()
    if path.suffix.lower() != ".zip" or not path.is_file() or path.stat().st_size > MAX_ZIP_BYTES:
        raise ZipImportError("ZIP 不存在、格式不支持或超过100MB。")
    members = {}
    seen = set()
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_FILES:
            raise ZipImportError("ZIP 条目超过512。")
        for info in infos:
            name = _member_name(info.filename)
            if name.casefold() in seen:
                raise ZipImportError("ZIP 有重复或大小写冲突路径。")
            seen.add(name.casefold())
            kind = stat.S_IFMT(info.external_attr >> 16)
            if info.flag_bits & 1 or kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ZipImportError("ZIP 不允许加密、链接或特殊文件。")
            total += info.file_size
            if total > MAX_ZIP_TOTAL_BYTES or info.file_size / max(info.compress_size, 1) > 200:
                raise ZipImportError("ZIP 展开大小或压缩比超限。")
            if info.is_dir():
                continue
            suffix = Path(name).suffix.lower()
            limit = MAX_MARKDOWN_BYTES if suffix in {".md", ".markdown"} else MAX_IMAGE_BYTES
            if suffix not in IMAGE_SUFFIXES | {".md", ".markdown"} or info.file_size > limit:
                raise ZipImportError("ZIP 包含不支持文件或成员过大。")
            members[name] = info.filename
    # 文件不能同时作为其他成员的父目录。
    for name in members:
        parent = posixpath.dirname(name)
        while parent:
            if parent.casefold() in {n.casefold() for n in members}:
                raise ZipImportError("ZIP 文件和目录冲突。")
            parent = posixpath.dirname(parent)
    markdown = sorted(n for n in members if Path(n).suffix.lower() in {".md", ".markdown"})
    if not markdown:
        raise ZipImportError("ZIP 内没有 Markdown。")
    return {"zip_path": str(path), "members": members, "markdown_members": markdown,
            "file_count": len(members), "total_bytes": total}


def _zip_parts(text):
    """仅在非代码区域把 Typora HTML img 转为现有扫描器支持的内联图片。"""
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence is None:
            masked = re.sub(r"(`+).*?\1", lambda m: " " * len(m[0]), line)
            for match in reversed(list(HTML_IMAGE.finditer(masked))):
                line = line[:match.start()] + "![ZIP图片](<" + match[2] + ">)" + line[match.end():]
        lines.append(line)
    return _markdown_parts("\n".join(lines))


def _image_member(reference, markdown_member):
    """只解析包内相对图片路径；拒绝外部盘符、URL、UNC和越界引用。"""
    reference = unquote(reference.strip())
    if reference.startswith("<") and reference.endswith(">"):
        reference = reference[1:-1]
    else:
        reference = re.sub(r'\s+["\'][^"\']*["\']\s*$', "", reference)
    reference = reference.replace("\\", "/")
    if not reference or reference.startswith("/") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", reference):
        raise ZipImportError("外部图片路径已拦截，不读取 ZIP 外文件。")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(markdown_member), reference))
    return _member_name(target)


def import_zip(file_path, vector_store=None, *, vision=None):
    """【业务需求】图文 ZIP 按章节导入，保持来源稳定、更新清旧和明确失败结果。
    【输入】ZIP 路径、可选存储与视觉函数；默认真实调用需要事先授权。
    【执行步骤】1. 安全检查与有界读取。2. 校验引用图片。3. 缓存/识别。
    4. 复用图像描述切分和笔记写入；外部/缺失引用明确警告，失败不假装完整。
    【输出】status/write_status、Chunk/图片统计、来源、warnings；partial表示确有遗漏。
    """
    path = Path(file_path).expanduser().resolve()
    result = {"source": os.path.normcase(str(path)), "filename": path.name, "status": "failed",
              "chunk_count": 0, "vision_calls": 0, "image_cached": 0, "image_processed": 0,
              "image_references": 0, "image_eligible": 0, "warnings": [], "images": []}
    try:
        manifest = inspect_zip(path)
        parts = []
        images = {}
        total_image_bytes = 0
        # 只复制明确引用的图片字节；不执行 extractall，不读 ZIP 外文件。
        with zipfile.ZipFile(path) as archive, TemporaryDirectory(prefix="v2-zip-") as temporary:
            for member in manifest["markdown_members"]:
                info = archive.getinfo(manifest["members"][member])
                with archive.open(info) as stream:
                    raw = stream.read(MAX_MARKDOWN_BYTES + 1)
                if len(raw) > MAX_MARKDOWN_BYTES:
                    raise ZipImportError("Markdown 超限。")
                text = raw.decode("utf-8-sig")
                if not text.strip():
                    raise ZipImportError("Markdown 为空。")
                for kind, content, headings in _zip_parts(text):
                    if kind == "text":
                        parts.append((kind, content, headings, member))
                        continue
                    result["image_references"] += 1
                    try:
                        target = _image_member(content, member)
                    except ValueError:
                        result["warnings"].append(f"{member}：外部或越界图片引用已拦截：{content}")
                        continue
                    if target not in manifest["members"]:
                        result["warnings"].append(f"{member}：包内图片缺失：{target}")
                        continue
                    if target not in images:
                        if len(images) >= MAX_ZIP_IMAGES:
                            raise ZipImportError("引用图片超过100张。")
                        info = archive.getinfo(manifest["members"][target])
                        with archive.open(info) as stream:
                            raw = stream.read(MAX_IMAGE_BYTES + 1)
                        total_image_bytes += len(raw)
                        if total_image_bytes > MAX_TOTAL_IMAGE_BYTES or len(raw) > MAX_IMAGE_BYTES:
                            raise ZipImportError("引用图片大小超限。")
                        local = Path(temporary) / (str(len(images)) + Path(target).suffix)
                        local.write_bytes(raw)
                        images[target] = read_image(local)
                    parts.append((kind, target, headings, member))
            result["image_eligible"] = len(images)
            store = vector_store if vector_store is not None else create_vector_store()
            if vector_store is None:
                store._embedding_function.max_retries = 1
            model = vision_model_name()
            cached = _cache(store.get(where={"source": result["source"]}, include=["metadatas"]), model)
            descriptions = {}
            chunks = []
            for kind, content, headings, member in parts:
                metadata = {**headings, "zip_path": str(path), "markdown_member": member}
                if kind == "text":
                    chunks.extend(RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(
                        [Document(page_content=content, metadata=metadata)]))
                    continue
                image_bytes, mime, image_hash = images[content]
                # metadata 和缓存只使用稳定 ZIP 成员标识，不泄露临时解压路径。
                image_source = "zip:" + result["source"] + "!/" + content
                if content not in descriptions:
                    key = (image_source, image_hash)
                    if key in cached:
                        description = cached[key]
                        result["image_cached"] += 1
                    else:
                        result["vision_calls"] += 1
                        description = VisionContent.model_validate((vision or describe_image)(image_bytes, mime))
                        if not description.confirmed_content:
                            raise ZipImportError("图片未提取出明确知识。")
                    descriptions[content] = description
                    result["image_processed"] += 1
                description = descriptions[content]
                metadata["image_member"] = content
                chunks.extend(_image_documents(image_source, content, metadata, image_hash, description, model))
                result["images"].append({"image_member": content, "markdown_member": member, "headings": headings})
            saved = save_note_chunks(path, chunks, store)
            result.update(saved)
            result["write_status"] = saved["status"]
            if result["warnings"]:
                result["status"] = "partial"
    except Exception as error:
        result["status"] = "failed"
        reason = str(error) if isinstance(error, ZipImportError) else type(error).__name__
        result["warnings"].append("ZIP 导入未完成（" + reason + "）；若存储失败需核对索引，未自动重试。")
    return result
