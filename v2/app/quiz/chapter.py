"""只按真实来源与标题 metadata 读取章节，不做相似度检索。"""

import os
import json
import re
from pathlib import Path

from .schemas import ChapterChunk, ChapterResult


HEADER_KEYS = ("Header 1", "Header 2", "Header 3")
NOTE_MATERIAL_BUDGET = 12000


def read_note(vector_store, source: str) -> ChapterResult:
    """【业务需求】整篇出题读取同一来源，长笔记均匀选取不同章节原文。
    【输入】已有 Chroma、笔记来源。
    【执行步骤】校验真实片段；按连续标题路径分组；跨章节轮流选完整片段，限制序列化长度。
    【输出】有来源的代表性原文快照及覆盖数量；不承诺覆盖全部知识。
    """
    source = normalize_source(source)
    base = dict(source=source, headings={}, scope="note")
    try:
        chunks = _read_note(vector_store, source)
    except ValueError:
        return ChapterResult(status="invalid_metadata", message="片段来源或顺序不可靠。", **base)
    except Exception:
        return ChapterResult(status="storage_error", message="读取知识库失败。", **base)
    if not chunks:
        return ChapterResult(status="note_not_found", message="笔记尚未导入。", **base)
    groups = []
    for chunk in chunks:
        if not groups or groups[-1][0].headings != chunk.headings:
            groups.append([])
        groups[-1].append(chunk)
    # 优先选章节首、中、尾；超长材料跳过，绝不截取开头冒充整篇。
    queues = []
    for group in groups:
        priority = list(dict.fromkeys([0, len(group)//2, len(group)-1] + list(range(len(group)))))
        queues.append([group[i] for i in priority if _non_heading_lines(group[i].content)])
    # 章节过多时先均匀覆盖首、中、尾，再填充其他章节。
    order = list(dict.fromkeys([round(i*(len(groups)-1)/min(7,len(groups)-1))
                               for i in range(min(8,len(groups)))])) if len(groups)>1 else [0]
    order += [i for i in range(len(groups)) if i not in order]
    selected, selected_groups = [], set()
    while any(queues):
        for index in order:
            if not queues[index]:
                continue
            candidate = queues[index].pop(0)
            material = [c.model_dump() for c in selected + [candidate]]
            if len(json.dumps(material, ensure_ascii=False)) <= NOTE_MATERIAL_BUDGET:
                selected.append(candidate)
                selected_groups.add(index)
    selected.sort(key=lambda c: c.chunk_index)
    if not selected:
        return ChapterResult(status="empty_chapter", message="没有预算内可用的完整正文片段，请选择具体章节。", **base)
    coverage = dict(total_chunks=len(chunks), selected_chunks=len(selected),
                    total_sections=len(groups), selected_sections=len(selected_groups), sampled=len(selected)<len(chunks))
    return ChapterResult(status="ok", message="已读取整篇笔记的代表性材料。", chunks=selected,
                         content="\n\n".join(c.content for c in selected), coverage=coverage, **base)


def normalize_source(source: str) -> str:
    """与现有 Markdown 导入使用相同的来源规范化规则。"""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("请选择笔记来源。")
    return os.path.normcase(str(Path(source).expanduser().resolve()))


def _read_note(vector_store, source):
    """读取该来源全部片段，确保 ID、标题和原文顺序可用。"""
    records = vector_store.get(where={"source": source}, include=["documents", "metadatas"])
    chunks = []
    if not (len(records["ids"]) == len(records["documents"]) == len(records["metadatas"])):
        raise ValueError("片段字段数量不一致。")

    # 你可以把它和前面学过的
    # enumerate()
    # 对比：
    # - enumerate()：拿到索引和元素。
    # - zip()：同时拿到多个列表里对应位置的元素。

    for chunk_id, content, metadata in zip(records["ids"], records["documents"], records["metadatas"]):
        if not isinstance(metadata, dict) or metadata.get("source") != source:
            raise ValueError("来源 metadata 不完整。")
        index = metadata.get("chunk_index")
        if type(index) is not int or index < 1 or metadata.get("chunk_id") != chunk_id:
            raise ValueError("缺少可靠的 chunk_index 或 chunk_id。")
        headings = {}
        for key in HEADER_KEYS:
            if key in metadata:
                if not isinstance(metadata[key], str) or not metadata[key].strip():
                    raise ValueError("标题 metadata 无效。")
                headings[key] = metadata[key]
        if not isinstance(content, str):
            raise ValueError("片段正文无效。")

        chunks.append(ChapterChunk(chunk_id=chunk_id, content=content, source=source,
                                   headings=headings, chunk_index=index))

    chunks.sort(key=lambda chunk: chunk.chunk_index)
    if [chunk.chunk_index for chunk in chunks] != list(range(1, len(chunks) + 1)):
        raise ValueError("片段顺序缺失或重复，不能可靠整理原文。")

    return chunks


def list_notes(vector_store) -> list[dict[str, str]]:
    """【业务需求】让用户选择实际已导入的笔记。
    【输入】已有 V2 Chroma 对象；不会自动创建默认数据库。
    【执行步骤】1. 读取 metadata。2. 按完整 source 去重。
    【输出】source/filename 列表；同名文件按 source 区分。
    """
    try:
        records = vector_store.get(include=["metadatas"])
        notes = {}
        for metadata in records["metadatas"]:
            if metadata and isinstance(metadata.get("source"), str):
                source = metadata["source"]
                notes[source] = {"source": source, "filename": metadata.get("filename", Path(source).name)}
        return [notes[source] for source in sorted(notes)]
    except Exception:
        raise ValueError("读取笔记列表失败，请检查已有 V2 知识库。") from None


def list_chapters(vector_store, source: str) -> list[dict[str, str]]:
    """【业务需求】列出笔记中的真实标题路径，供用户选择。
    【输入】已有 Chroma、笔记 source。
    【执行步骤】1. 来源过滤并按原文排序。2. 列出各层完整标题路径。
    【输出】标题字典列表；{} 表示无标题正文；缺失笔记返回 []。
    """
    source = normalize_source(source)
    try:
        chunks = _read_note(vector_store, source)
    except Exception:
        raise ValueError("读取章节列表失败，可能缺少 source、chunk_index 或 chunk_id。") from None
    chapters = []
    for chunk in chunks:
        headings = {}
        if not chunk.headings and {} not in chapters:
            chapters.append({})
        for key in HEADER_KEYS:
            if key in chunk.headings:
                headings[key] = chunk.headings[key]
                if headings not in chapters:
                    chapters.append(dict(headings))
    return chapters


def _non_heading_lines(text):
    """标题本身不当作可出题正文；围栏中的内容仍属于原文。"""
    return [line for line in text.splitlines() if line.strip() and not re.match(r"^\s*#{1,6}\s", line)]


def _heading_occurrences(chunks, headings):
    """识别重复目标标题；先去掉长度切分造成的相邻正文重叠。"""
    text = ""
    for chunk in chunks:
        content = chunk.content
        overlap = 0
        for size in range(min(100, len(text), len(content) - 1), 0, -1):
            if text.endswith(content[:size]):
                overlap = size
                break
        text += "\n" + content[overlap:]
    key = list(headings)[-1]
    level = HEADER_KEYS.index(key) + 1
    target = headings[key]
    count = 0
    fence = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            match = re.match(r"^(#{1,3})\s+(.*?)\s*$", stripped)
            if match and len(match[1]) == level and match[2] == target:
                count += 1
    return count


def read_chapter(vector_store, source: str, headings: dict[str, str]) -> ChapterResult:
    """【业务需求】定位指定笔记章节，防止其他文件或同名章节混入。
    【输入】已有 Chroma、source、list_chapters 返回的完整标题字典。
    【执行步骤】
    1. 规范化来源并校验标题路径。
    2. 只读取该 source，按 chunk_index 排序。
    3. 匹配完整标题前缀，父章节包含子章节；{} 只选无标题正文。
    4. 检查重复章节歧义与空正文。
    【输出】ChapterResult；成功保留真实 Chunk ID、来源、标题及顺序。
    """
    # 1. 字典保留 Header 层级，兼容原笔记跳级使用 ### 的情况。
    source = normalize_source(source)
    if not isinstance(headings, dict) or any(key not in HEADER_KEYS for key in headings):
        raise ValueError("章节必须使用 Header 1/2/3 标题字典。")
    if any(not isinstance(value, str) or not value.strip() for value in headings.values()):
        raise ValueError("章节标题不能为空。")
    headings = {key: headings[key] for key in HEADER_KEYS if key in headings}
    base = {"source": source, "headings": headings}
    # 2. 不调用 Embedding、不访问默认库、不发送模型请求。
    try:
        chunks = _read_note(vector_store, source)
    except ValueError:
        return ChapterResult(status="invalid_metadata", message="缺少可靠的片段顺序、来源或标题 metadata。", **base)
    except Exception:
        return ChapterResult(status="storage_error", message="读取知识库失败，请检查已有 V2 Chroma。", **base)
    if not chunks:
        return ChapterResult(status="note_not_found", message="笔记尚未导入。", **base)
    # 3. 既比较标题，也比较选定层级内缺失的父标题，避免同名章节串库。
    depth = max((HEADER_KEYS.index(key) + 1 for key in headings), default=0)
    selected = []
    positions = []
    for position, chunk in enumerate(chunks):
        # prefix 的作用是拿出需要比较的那几级标题。
        prefix = {key: chunk.headings[key] for key in HEADER_KEYS[:depth] if key in chunk.headings}
        matches = prefix == headings if headings else not chunk.headings
        if matches:
            selected.append(chunk)
            positions.append(position)
    if not selected:
        return ChapterResult(status="chapter_not_found", message="所选章节不存在。", **base)
    # 4. 可检测的重复标题或分散的同路径章节，明确拒绝合并出题。
    separated = positions != list(range(positions[0], positions[-1] + 1))
    if separated or (headings and _heading_occurrences(selected, headings) > 1):
        return ChapterResult(status="ambiguous_chapter", message="相同标题路径重复出现，现有 metadata 无法唯一定位，请选更明确的章节。", **base)
    content = "\n\n".join(chunk.content for chunk in selected)
    if not _non_heading_lines(content):
        return ChapterResult(status="empty_chapter", message="所选章节只有标题或空白，没有可出题正文。", **base)
    return ChapterResult(status="ok", message="章节读取成功。", chunks=selected, content=content, **base)
