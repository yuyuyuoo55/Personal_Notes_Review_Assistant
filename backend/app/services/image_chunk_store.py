"""Persist image-description chunks so BM25 can be rebuilt after restart."""

import json
from pathlib import Path

from langchain_core.documents import Document


def manifest_path(note_path: str | Path) -> Path:
    path = Path(note_path)
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
