# 新增：笔记删除功能（给 codex）+ 修复"重复导入删不掉"卡死

> 本文件是给 codex 的改造说明。目标：给笔记库增加「删除」能力，解决用户导入同名文件后无法删除的卡死问题。
> 现状：项目只有「导入(import)」「列表(list)」，没有「删除」。上传同名 .md / 图片会 409/提示已存在，但用户删不掉。

---

## 0. 已确认的决策（不要再改）

1. **支持两种删除**：单条删除（删某一个笔记）+ 清空全部（删光所有笔记）。
2. **删除前要确认**（避免误删）：单条删除弹确认；清空全部弹确认。
3. 两种笔记类型都要能删：
   - `.md` 笔记：删除 `data/uploads/<file>.md` + Chroma 里 `source` 匹配的块 + 失效 BM25 缓存。
   - **独立图片**笔记：删除 `data/uploads/<doc_id>/` 整个文件夹（含 `images/`、`.images.json`）+ Chroma 里 `doc_id` 匹配的块 + 失效 BM25。
4. 删除后前端笔记库要刷新（rerun 或重新 list）。

---

## 1. 需要先解决的问题：`list_notes` 返回字段不够用于删除

现状 `streamlit_demo.py` 的 `list_notes()` 返回 `{"file_name", "chunk_count"}`，**没有区分 .md 和图片，也没有定位信息（doc_id/source）**。

删除时需按「笔记类型 + 定位信息」精准删。请给每个笔记条目**增加可标识字段**，例如：
- `.md`：`{"file_name", "kind": "md", "source": <abs path>}`
- 图片：`{"file_name", "kind": "image", "doc_id": "<doc_id>", "source": <abs path>}`

`NoteSummary`（`backend/app/schemas/note.py`）若被复用，需同步加字段（`kind`、`doc_id`、`source`），或前端单独用 dict。

---

## 2. 要改的文件

### 文件 1：`backend/app/services/image_chunk_store.py`

- 已有 `remove_image_doc_dir(doc_dir, upload_dir)`（删整个 doc 目录）。确认它安全（前面已验证清理逻辑）。
- 可加一个辅助 `delete_standalone_image_chunks(upload_dir, doc_id)`：删除属于某 doc_id 的 `.images.json` 及对应图片（若某 doc_id 对应多个独立图片块）。

### 文件 2：`backend/app/services/rag_service.py`（或独立删除服务）

新增删除函数，例如：

```python
def delete_note(identifier: str, kind: str) -> None:
    """删除单个笔记：kind = 'md' | 'image'"""
    if kind == "md":
        # 1) 删 Chroma：vector_store.collection.delete(where={"source": <md abs path>})
        # 2) 删本地 .md 文件
        # 3) 清 manifest（<md>.images.json，若存在）
    else:  # image
        # 1) 删 Chroma：vector_store.collection.delete(where={"doc_id": <doc_id>})
        # 2) 删整个 doc 目录：remove_image_doc_dir
    invalidate_rag_cache()

def delete_all_notes() -> None:
    """清空全部：删 Chroma 全部记录 + 清空 data/uploads 下所有笔记/图片目录 + 失效缓存。"""
```

> ⚠️ Chroma 删除要确认 API：`vector_store._collection.delete(where={...})` 或 `vector_store.delete(ids=[...])`。请用当前 `chromadb` 版本支持的方式，**按 metadata 的 where 条件删除**（不要按 id 硬编码）。
> ⚠️ 清空全部时，要清 `data/uploads/` 下的 `.md` 和所有 `<doc_id>/` 文件夹，并清 vector_store 全部（可用 `vector_store.delete(where={})` 或重新建 collection）。

### 文件 3：`backend/app/api/notes.py`（双进程后端，保持一致）

- 新增 `DELETE /api/notes/{note_id}`（删除单条）和 `DELETE /api/notes`（清空全部）。
- 逻辑与文件 2 的删除函数一致。`note_id` 可以编码 kind + 定位（如 `md:<file>` 或 `image:<doc_id>`），由后端解析。

### 文件 4：`streamlit_demo.py`（单进程 Demo，Cloud 用这个，**最关键**）

- **笔记库渲染**（约第 399-409 行）：每个笔记项右侧加一个 **🗑 删除** 按钮（用 `st.button` 或 `st.columns([...])` 布局）。
  - `.md` 项 → 删除传 `kind="md", source=<path>`
  - 图片项 → 删除传 `kind="image", doc_id=<doc_id>`
- **点删除时**：弹确认（`st.warning` + 再确认，或用 Streamlit 的确认方式），确认后调用删除函数（单进程版直接调后端同名函数），`invalidate_rag_cache()`，重新 `list_notes()`，`st.rerun()`。
- **清空全部**：在笔记库上方加一个「🗑 清空全部」按钮，确认后 `delete_all_notes()`，rerun。
- `has_api_key` / 导入逻辑保持不变（只加删除入口）。

---

## 3. 注意事项 / 不要踩的坑

1. **不要删除 `data/uploads` 根目录本身**，只删里面的 `.md` 和 `<doc_id>/` 子目录；清空后保留空目录（`UPLOAD_DIRECTORY.mkdir(exist_ok=True)`）。
2. **删除要精准**：图片按 `doc_id` 删（不要误删其它图片的目录）；`remove_image_doc_dir` 已有"父目录校验"，继续用。
3. **删除后必须 `invalidate_rag_cache()`**，否则 BM25/Chunk 缓存仍含旧数据，检索还会命中已删内容。
4. **Chroma 删除条件**用 metadata 的 where（`source` 或 `doc_id`），确认 `chromadb` 当前版本 API（`collection.delete(where={"source": ...})`）。删除后 `vector_store` 计数会更新。
5. **两套入口同步**：`streamlit_demo.py`（单进程）与 `backend/app/api/notes.py`（双进程后端）都要加删除，行为一致。
6. **删除接口鉴权**：`DELETE` 接口应像 import 一样要求 `X-DeepSeek-API-Key`（`require_user_deepseek_api_key`），保持 BYOK 一致性。
7. **前端删除按钮**：注意 Streamlit `st.session_state` 里如果存了下载/历史，删除后要同步清理对应历史（可选，MVP 至少刷新列表）。
8. **单条删除后**：若该笔记在 `st.session_state.messages` 历史里有引用，历史里可能仍显示旧的 source 卡片——MVP 可保留历史，不影响主功能。

---

## 4. 完成标准（验收）

- [ ] 笔记库每个条目有「🗑 删除」按钮（.md 和 图片都有）
- [ ] 点删除弹确认，确认后该笔记从列表消失
- [ ] 图片笔记删除后，`data/uploads/<doc_id>/` 整个目录被清理
- [ ] .md 笔记删除后，`.md` 文件及 Chroma 对应块被删
- [ ] 有「清空全部」按钮，确认后所有笔记被清空
- [ ] 删除后重新提问不会命中已删内容（缓存已失效）
- [ ] 后端 `DELETE /api/notes/{id}` 与 `DELETE /api/notes` 可用，且要求 Key 头
- [ ] `streamlit_demo.py` 与 `notes.py` 行为一致
- [ ] `pytest tests/` 全绿（新增删除测试用 mock，无外部费用）
- [ ] 不破坏现有导入/图片入库/检索功能
