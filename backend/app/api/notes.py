"""笔记接口：相当于 Spring Boot 的 NotesController。"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.core.auth import require_user_deepseek_api_key
from backend.app.core.config import UPLOAD_DIRECTORY
from backend.app.schemas.note import ImportResult, NoteSummary
from backend.app.services.note_loader import load_notes
from backend.app.services.markdown_image_service import enrich_markdown_images
from backend.app.services.image_chunk_store import manifest_path, save_image_chunks
from backend.app.services.note_splitter import split_documents
from backend.app.services.rag_service import invalidate_rag_cache
from backend.app.storage.vector_store import knowledge_to_vector, vector_store

# 1. 创建本模块的路由器；prefix 相当于类上的 @RequestMapping("/api/notes")。
router = APIRouter(
    prefix="/api/notes",
    tags=["notes"],
)


@router.post(
    "/import",
    response_model=ImportResult,
    status_code=status.HTTP_201_CREATED,
)
async def import_note(
    file: UploadFile = File(...),
    api_key: str = Depends(require_user_deepseek_api_key),
) -> ImportResult:
    """接收一个 Markdown 文件，调用既有 RAG 服务完成加载、切分和向量化。"""
    # 2. 读取安全的文件名，并限制当前 MVP 只接收 Markdown。
    file_name = Path(file.filename or "").name

    if not file_name or Path(file_name).suffix.lower() != ".md":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前只支持上传 .md 格式笔记",
        )

    # 3. 保存上传文件。UploadFile.read() 是异步 I/O，所以接口使用 async def。
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIRECTORY / file_name

    if file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="同名笔记已存在；当前版本不重复导入",
        )

    file_content = await file.read()

    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="上传文件不能为空",
        )

    try:
        try:
            markdown = file_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Markdown 文件必须使用 UTF-8 编码",
            ) from error

        image_result = await enrich_markdown_images(
            markdown,
            api_key,
            source_path=str(file_path),
            doc_dir=UPLOAD_DIRECTORY / file_path.stem,
        )
        await run_in_threadpool(file_path.write_text, image_result.markdown, encoding="utf-8")

        # 4. 调用已有 service：加载文档 → 标题切分 → 写入 Chroma。
        docs = await run_in_threadpool(load_notes, str(file_path))
        chunks = await run_in_threadpool(split_documents, docs)
        chunks.extend(image_result.image_chunks)
        await run_in_threadpool(save_image_chunks, file_path, image_result.image_chunks)
        success = await run_in_threadpool(knowledge_to_vector, chunks)

        if not success:
            raise RuntimeError("笔记切分后没有可写入向量库的内容")

        # 新笔记会改变检索范围，清除缓存并让下一次问答重建 BM25 索引。
        invalidate_rag_cache()

        # 5. 按 response_model 返回 DTO，FastAPI 会自动转为 JSON。
        return ImportResult(
            file_name=file_name,
            chunk_count=len(chunks),
            status="success",
            error_msg=None,
            image_total=image_result.image_total,
            image_processed=image_result.image_processed,
            image_skipped=image_result.image_skipped,
            warnings=image_result.warnings,
        )

    except HTTPException:
        file_path.unlink(missing_ok=True)
        manifest_path(file_path).unlink(missing_ok=True)
        raise
    except Exception as error:
        # 本次导入失败时删除刚保存的文件，避免留下无法使用的上传文件。
        file_path.unlink(missing_ok=True)
        manifest_path(file_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="笔记导入失败，请检查配置后重试",
        ) from error


@router.get("", response_model=list[NoteSummary])
def list_notes() -> list[NoteSummary]:
    """返回已上传笔记及其已写入 Chroma 的 Chunk 数量。"""
    # 1. 首次启动还没有上传目录时，返回空列表。
    if not UPLOAD_DIRECTORY.exists():
        return []

    notes: list[NoteSummary] = []

    # 2. 遍历本地已上传的 Markdown 笔记。
    for file_path in UPLOAD_DIRECTORY.glob("*.md"):
        # 3. 根据 TextLoader 保存的 source 元数据，统计该文件对应的 Chunk 数量。
        stored_chunks = vector_store.get(where={"source": str(file_path)})

        notes.append(
            NoteSummary(
                note_id=file_path.stem,
                file_name=file_path.name,
                chunk_count=len(stored_chunks["ids"]),
            )
        )

    return notes
