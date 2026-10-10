"""Phase 15：独立导入 Markdown，复用 V2 的 Embedding 与 Chroma。"""

import json
import os
from hashlib import sha256
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..rag.vector_retrieval import create_vector_store


def import_markdown(file_path, vector_store=None):
    """【业务需求】导入或更新一份 Markdown，不重复保存、不删除其他文件。
    【输入】本地 Markdown 路径；可选已有 V2 Chroma 对象。
    【执行步骤】
    1. 检查文件并读取 UTF-8 文本。
    2. 按标题与长度切分，保留标题层级。
    3. 设置来源与稳定的片段 ID。
    4. 获取同一文件的旧片段，未变化时跳过写入。
    5. 向量化并写入新片段，再删除该文件的过期片段。
    6. 返回文件名、片段数量与状态。
    【输出】{filename, source, chunk_count, status}。
    """
    # 1. 解析完整路径，检查文件类型、编码与空内容。
    path = Path(file_path).expanduser().resolve()
    """
    Path(file_path)	把路径字符串变成 Path 路径对象
    .expanduser()	如果路径包含 ~，就展开为用户主目录
    .resolve()	转换成规范的绝对路径，并处理 .. 等路径部分
    """
    if not path.is_file():
        raise ValueError("Markdown 文件不存在或路径不是文件。")
    if path.suffix.lower() not in [".md", ".markdown"]:
        raise ValueError("只支持 .md 或 .markdown 文件。")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Markdown 文件必须使用 UTF-8 编码。") from error
    if not text.strip():
        raise ValueError("Markdown 文件为空，请添加内容后再导入。")

    # 2. 先按三级标题分组，再将长段落切到最多 800 字符，重叠最多 100 字符。
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)
    length_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = length_splitter.split_documents(sections)
    if not chunks:
        raise ValueError("Markdown 中没有可以导入的文本片段。")

    return save_note_chunks(path, chunks, vector_store)


def save_note_chunks(path, chunks, vector_store=None):
    """【业务需求】纯文本和图片描述共用来源、连续顺序与更新规则。
    【输入】原文件 Path、按原文排序的 Document 列表、可选 V2 Chroma。
    【执行步骤】1. 设置稳定 ID。2. 比较旧片段。3. 写新片段成功后清理过期 ID。
    【输出】与 import_markdown 相同的 filename/source/chunk_count/status。
    """
    path = Path(path).expanduser().resolve()
    if not chunks:
        raise ValueError("没有可以导入的文本片段。")

    # 3. 统一来源路径；文件、顺序、标题和正文共同确定 ID。
    source = os.path.normcase(str(path)) # os.path.normcase()在 Windows 上，主要将路径统一为小写，并规范路径分隔符。
    chunk_ids = []
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["source"] = source
        chunk.metadata["filename"] = path.name
        chunk.metadata["chunk_index"] = index

        # 它会将 fingerprint 转换成一个固定长度的哈希值，作为 Chunk ID。
        # 这样设计的效果：
        # - 同一个文件、相同位置、相同内容和 metadata，通常生成相同 ID。
        # - 正文修改后，ID 就会改变。
        # - 不同文件即使正文相同，由于 source 不同，也会生成不同 ID。
        # 这就方便后面判断 Chunk 是否需要更新。
        # fingerprint可以理解成：
        # {"chunk_index": 1, "source": "d:\\notes\\rag.md"}
        # RRF是一种排名融合算法。

        fingerprint = json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True)
        fingerprint += "\n" + chunk.page_content
        chunk_id = sha256(fingerprint.encode("utf-8")).hexdigest()

        chunk.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)

    # 4. 只读取该来源的旧片段；相同内容重复导入不再调用 Embedding。
    if vector_store is None:
        vector_store = create_vector_store()
    records = vector_store.get(where={"source": source}, include=["documents", "metadatas"])
    old_chunks = {}
    for index, chunk_id in enumerate(records["ids"]):
        old_chunks[chunk_id] = {
            "content": records["documents"][index],
            "metadata": records["metadatas"][index],
        }
    unchanged = len(old_chunks) == len(chunks)
    for chunk in chunks:
        old = old_chunks.get(chunk.metadata["chunk_id"])
        if old is None or old["content"] != chunk.page_content or old["metadata"] != chunk.metadata:
            unchanged = False

    # 5. Chroma 自动使用已有 Embedding 向量化并 upsert；写入成功后才清理旧 ID。
    status = "unchanged"
    if not unchanged:
        vector_store.add_documents(chunks, ids=chunk_ids)
        stale_ids = []
        for chunk_id in records["ids"]:
            if chunk_id not in chunk_ids:
                stale_ids.append(chunk_id)
        if stale_ids:
            vector_store.delete(ids=stale_ids)
        status = "updated" if records["ids"] else "imported"

    # 6. 返回实际导入状态，不声称执行了未发生的写入。
    return {"filename": path.name, "source": source, "chunk_count": len(chunks), "status": status}
