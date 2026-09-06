"""笔记删除服务：同步清理 Chroma、本地文件和 RAG 缓存。"""

from pathlib import Path
import shutil
from typing import Literal

from backend.app.core.config import UPLOAD_DIRECTORY
from backend.app.services.image_chunk_store import manifest_path, remove_image_doc_dir
from backend.app.services.rag_service import invalidate_rag_cache
from backend.app.storage.vector_store import vector_store

NoteKind = Literal["md", "image"]


def _safe_child(upload_directory: Path, name: str) -> Path:
    """只允许定位上传根目录的直接子项，阻止路径穿越。"""
    if not name or Path(name).name != name:
        raise ValueError("笔记标识无效")
    target = (upload_directory / name).resolve()
    if target.parent != upload_directory.resolve():
        raise ValueError("笔记标识无效")
    return target


def delete_note(
    identifier: str,
    kind: NoteKind,
    upload_directory: str | Path = UPLOAD_DIRECTORY,
) -> None:
    """精准删除一个 Markdown 或独立图片笔记。"""
    upload_root = Path(upload_directory)
    if kind == "md":
        file_path = _safe_child(upload_root, identifier)
        if file_path.suffix.lower() != ".md" or not file_path.is_file():
            raise FileNotFoundError("笔记不存在")
        vector_store._collection.delete(where={"source": str(file_path)})
        file_path.unlink()
        manifest_path(file_path).unlink(missing_ok=True)
        image_dir = upload_root / file_path.stem
        if image_dir.exists():
            remove_image_doc_dir(image_dir, upload_root)
    elif kind == "image":
        doc_dir = _safe_child(upload_root, identifier)
        if not doc_dir.is_dir():
            raise FileNotFoundError("笔记不存在")
        vector_store._collection.delete(where={"doc_id": identifier})
        remove_image_doc_dir(doc_dir, upload_root)
    else:
        raise ValueError("不支持的笔记类型")
    invalidate_rag_cache()


def delete_all_notes(upload_directory: str | Path = UPLOAD_DIRECTORY) -> None:
    """清空全部笔记，但保留 uploads 根目录本身。"""
    upload_root = Path(upload_directory)
    upload_root.mkdir(parents=True, exist_ok=True)

    stored = vector_store.get(include=[])
    if stored.get("ids"):
        vector_store.delete(ids=stored["ids"])

    for child in upload_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()
    invalidate_rag_cache()
