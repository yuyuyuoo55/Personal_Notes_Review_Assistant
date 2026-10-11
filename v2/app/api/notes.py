"""笔记列表、章节与持久化安全上传。"""
import os
import asyncio
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response

from ..knowledge.image_import import import_image, import_markdown_images
from ..knowledge.note_import import import_markdown
from ..knowledge.zip_import import import_zip, _zip_parts
from ..quiz.chapter import list_notes, list_chapters
from .runtime import ApiError, response, run_job, runtime_from
from .schemas import Envelope
from .source_images import source_image

router = APIRouter(prefix="/notes", tags=["notes"])
LIMITS = {".md": 2*1024*1024, ".markdown": 2*1024*1024,
          ".png": 10*1024*1024, ".jpg": 10*1024*1024, ".jpeg": 10*1024*1024,
          ".webp": 10*1024*1024, ".zip": 100*1024*1024}


@router.get("/images/{chunk_id}")
async def image(request: Request, chunk_id: str):
    runtime = runtime_from(request)
    raw, mime = await run_job(runtime, source_image, runtime, chunk_id)
    return Response(raw, media_type=mime, headers={"Cache-Control": "private, no-store",
                    "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'"})


def upload_name(name):
    """只允许单个原始文件名，拒绝遍历、盘符、ADS及Windows特殊名称。"""
    if not name or len(name) > 128 or name in {".", ".."} or name.rstrip(". ") != name or any(
        ord(c) < 32 or c in '/\\:<>"|?*' for c in name
    ) or re.match(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)", name, re.I):
        raise ApiError(422, "invalid_filename", "文件名不安全，请只上传普通文件名。")
    if Path(name).suffix.lower() not in LIMITS:
        raise ApiError(415, "unsupported_file", "仅支持Markdown、PNG/JPEG/WebP和ZIP。")
    return name


@router.get("", response_model=Envelope[list[dict]])
async def notes(request: Request):
    runtime = runtime_from(request)
    return response(await run_job(runtime, lambda: list_notes(runtime.store())))


@router.get("/chapters", response_model=Envelope[list[dict]])
async def chapters(request: Request, source: str):
    runtime = runtime_from(request)
    def read():
        store = runtime.store()
        if source not in {note["source"] for note in list_notes(store)}:
            raise ApiError(404, "note_not_found", "笔记不存在。")
        return list_chapters(store, source)
    return response(await run_job(runtime, read))


@router.post("/upload", response_model=Envelope[dict])
async def upload(request: Request, file: UploadFile = File(...), markdown_images: bool = Form(False)):
    """【业务需求】持久化上传原文件，再复用现有导入，不允许任意服务器路径。
    【输入】multipart file、Markdown图片开关。
    【执行步骤】校验名称和大小；有界暂存；线程内原子保存并调用业务入口。
    【输出】实际导入状态、来源、Chunk/图片统计与warnings；原文件保留用于溯源。
    """
    runtime = runtime_from(request)
    temporary = None
    try:
        name = upload_name(file.filename)
        suffix = Path(name).suffix.lower()
        runtime.upload_root.mkdir(parents=True, exist_ok=True)
        target = runtime.upload_root / name
        if target.is_symlink() or not target.resolve().is_relative_to(runtime.upload_root):
            raise ApiError(422, "invalid_filename", "上传目标不安全。")
        temporary = runtime.upload_root / (".upload-" + uuid4().hex)
        total = 0
        with temporary.open("xb") as output:
            while chunk := await file.read(64*1024):
                total += len(chunk)
                if total > LIMITS[suffix]:
                    raise ApiError(413, "file_too_large", "上传文件超过格式大小限制。")
                await asyncio.to_thread(output.write, chunk)
        if not total:
            raise ApiError(422, "empty_file", "上传文件不能为空。")
        staged_path = temporary
        def perform():
            try:
                os.replace(staged_path, target)
                store = runtime.store()
                if suffix in {".md", ".markdown"}:
                    if markdown_images:
                        result = import_markdown_images(target, store, vision=runtime.vision)
                    else:
                        try:
                            text = target.read_text(encoding="utf-8-sig")
                        except UnicodeDecodeError:
                            text = ""
                        if not text.strip():
                            result = {"status": "failed", "chunk_count": 0, "warnings": ["Markdown内容为空或编码无效。"]}
                        else:
                            result = import_markdown(target, store)
                            # 普通 Markdown 请求只向量化文字；检测引用但不定位/读取图片。
                            references=sum(kind=="image" for kind,_,_ in _zip_parts(text))
                            warnings=list(result.get("warnings",[]))
                            if references:
                                warnings.append(f"本次仅导入文字；{references} 处图片引用未随文件上传、未识别。需要图文内容时，请将 Markdown 和图片打包为图文 ZIP。")
                            result={**result,"image_references":references,"image_processed":0,
                                    "vision_calls":0,"warnings":warnings}
                elif suffix == ".zip":
                    result = import_zip(target, store, vision=runtime.vision)
                else:
                    result = import_image(target, store, vision=runtime.vision)
                return {**result, "original_filename": name, "stored_path": str(target), "file_bytes": total}
            finally:
                staged_path.unlink(missing_ok=True)
        # worker接管暂存路径，超时后由它完成；请求不能提前删除后台还在使用的文件。
        try:
            result = await run_job(runtime, perform)
        except asyncio.CancelledError:
            temporary = None
            raise
        except ApiError as error:
            if error.code == "timeout":
                # 超时不能删除后台线程尚未接管的文件；可能留有待清理暂存文件。
                temporary = None
            raise
        return response(result)
    finally:
        await file.close()
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)
