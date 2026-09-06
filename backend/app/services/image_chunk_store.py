"""Persist image-description chunks so BM25 can be rebuilt after restart."""

import json
from pathlib import Path
import shutil

from langchain_core.documents import Document


def manifest_path(note_path: str | Path) -> Path:
    path = Path(note_path)
    if path.is_dir() or not path.suffix:
        return path / f"{path.name}.images.json"
    return path.with_suffix(path.suffix + ".images.json")


def save_image_chunks(note_path: str | Path, chunks: list[Document]) -> None:
    payload = [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    manifest_path(note_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_image_chunks(note_path: str | Path) -> list[Document]:
    path = manifest_path(note_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in payload]


def load_standalone_image_chunks(upload_directory: str | Path) -> list[Document]:
    """Load standalone-image manifests stored below per-document directories."""
    upload_path = Path(upload_directory)
    chunks: list[Document] = []
    if not upload_path.exists():
        return chunks
    for path in upload_path.glob("*/*.images.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload:
            metadata = item.get("metadata", {})
            if metadata.get("standalone_image") is True:
                chunks.append(
                    Document(page_content=item["page_content"], metadata=metadata)
                )
    return chunks


def standalone_image_names(upload_directory: str | Path) -> set[str]:
    return {
        Path(str(chunk.metadata.get("source", ""))).name
        for chunk in load_standalone_image_chunks(upload_directory)
        if chunk.metadata.get("source")
    }


def remove_image_doc_dir(doc_dir: str | Path, upload_directory: str | Path) -> None:
    """Remove one generated document directory after verifying its exact parent."""
    target = Path(doc_dir).resolve()
    upload_root = Path(upload_directory).resolve()
    if target.parent != upload_root:
        raise ValueError("拒绝清理上传目录之外的图片文档")
    if target.exists():
        shutil.rmtree(target)
