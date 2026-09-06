# 新增：单独上传图片当笔记入库（给 codex）

> 本文件是给 codex 的改造说明。目标：让「导入笔记」除了 `.md`，还能直接上传**一张独立图片**，
> 大模型（Vision）识别图片内容生成文字描述，作为可检索的知识块存入向量库，并妥善分层存放。
>
> 现有能力：Markdown内图片、聊天区传图问答已可用。**本任务不删除聊天区传图问答**，只新增「图片当笔记入库」。

---

## 0. 已确认的决策（不要再改）

1. **图片存本地，不存 OSS。** 图片保存到本地目录，metadata 记 `image_path`。
2. **保留** 聊天区传图问答（不要删）。
3. **图片识别用 Vision 看图**（`describe_image_url`），不做专门 OCR 库。
4. **分层目录**：每个导入文档一个文件夹，图片放 `<doc_id>/images/`；并维护一份清单描述每块的来源/类型。
5. 图片入库后要**能被检索**（和其它文字块一起进向量库），检索命中带图块时前端可展示原图。

---

## 1. 目标行为

用户在侧边栏「导入笔记」上传一张图片（jpg/jpeg/png/webp）→
① 图片保存到 `data/uploads/<doc_id>/images/<uuid>.<ext>`；
② Vision 看图生成一段文字描述；
③ 该描述作为一个「图片块」存入向量库，metadata 带 `source`、`image_path`、`is_image_chunk=True`、标题、`chunk_id`、`doc_id`；
④ `import_note` 返回 `image_processed` 等信息；
⑤ 之后提问可检索到该图片块，前端来源卡片显示原图。

---

## 2. 要改的文件

### 文件 1：`streamlit_demo.py`（单进程 Demo，Streamlit Cloud 用这个，**最关键**）

- **导入入口**（约第 307 行 `st.file_uploader`）：`type=["md"]` → 增加图片类型：
  ```python
  type=["md", "jpg", "jpeg", "png", "webp"]
  ```
- **`import_note`（约第 105-140 行）**：
  - 现在强制 `.md`（`if not file_name.lower().endswith(".md")`）。改为允许图片。
  - 新增分支：
    - 若 `.md` → 走现有逻辑（不变）。
    - 若是**图片** → 走新逻辑：
      ```python
      # 判断是图片
      if file_name.lower().endswith((".jpg",".jpeg",".png",".webp")):
          # 1) 用 save_image_to_local 存图片到 doc_dir/images/
          local_path = save_image_to_local(file_content, mime, doc_dir)
          # 2) 用 describe_image_url(image_path_to_data_url(local_path), api_key) 生成描述
          description = await describe_image_url(image_path_to_data_url(local_path), api_key)
          # 3) 构造一个 Document（图片块），metadata 带上 source/image_path/is_image_chunk/chunk_id/doc_id
          image_chunk = Document(page_content=description, metadata={...})
          chunks = [image_chunk]
          # 4) save_image_chunks(file_path, [image_chunk]) 持久化（重启后 BM25 可重建）
          # 5) knowledge_to_vector(chunks)
          # 6) invalidate_rag_cache()
      else:
          # 现有 .md 逻辑
      ```
  - **注意**：doc_dir 用于分层，建议 `doc_dir = UPLOAD_DIRECTORY / file_path.stem`（与 .md 分支一致）。
  - 返回 dict 里 `chunk_count` 等沿用；`image_processed`/`image_skipped` 对应图片处理结果。
  - **metadata 必须包含** `image_path`（绝对路径）、`source`、`is_image_chunk=True`，供检索命中后前端展示原图。

### 文件 2：`backend/app/api/notes.py`（双进程后端 import_note，保持一致）

- 与 `streamlit_demo.py` 的 `import_note` 同步：允许图片，图片当笔记入库。
- 后端接口 `POST /api/notes/import` 目前只接受 `.md`（第 37-41 行校验）。改为也接受图片，并走新分支。
- 上层错误处理、`ImportResult` 返回字段（image_total/image_processed/image_skipped/warnings）沿用现有逻辑。

### 文件 3：`backend/app/services/multimodal_service.py`（若缺则补）

- 已有 `save_image_to_local`、`image_path_to_data_url`、`describe_image_url`、`validate_image`、`validate_deepseek_api_key`。
- **如需要**，可加一个辅助函数：给一张图片字节，存本地 + 生成描述 + 返回 Document（把「存图 + 看图 + 构造块」封装成一个可复用函数，供两套入口调用）。推荐加：
  ```python
  async def build_image_chunk(image_bytes, content_type, doc_dir, source_path, api_key) -> dict:
      """存图片到本地 -> Vision 生成描述 -> 返回 {Document(chunk), image_path, description}。"""
  ```
  这样 `streamlit_demo.py` 和 `notes.py` 都能复用，避免重复代码。

---

## 3. 分层目录与 metadata 统一定义

- 目录：
  ```
  data/uploads/<doc_id>/
    images/<uuid>.<ext>         # 图片本体
    <doc_id>.images.json        # 图片块清单（save_image_chunks 已生成）
  ```
- `doc_id`：建议用文件名 stem（与现有 `.md` 分支一致），或 `uuid4().hex` 避免重名冲突。
- 每个图片块 metadata：
  | 字段 | 说明 |
  |---|---|
  | `source` | 文档/图片来源路径 |
  | `image_path` | 图片本地绝对路径（前端展示用）|
  | `is_image_chunk` | `True`，标记图片块 |
  | `Header 1/2/3` | 标题路径（图片笔记可省略或给个默认）|
  | `chunk_id` | 稳定 ID（`sha256(source:image_path:description)[:16]`）|
  | `doc_id` | 所属文档 ID |

---

## 4. 注意事项 / 不要踩的坑

1. **不要删聊天区传图问答**（`uploaded_chat_image` 那段）。只新增「导入图片笔记」入口。
2. **图片存本地、不存 OSS**。`multimodal_service.upload_image_to_oss` 不要调用。
3. **Key 无效时中止**：`describe_image_url` 若抛 `InvalidApiKeyError`，要中止并提示「API Key 无效，导入已中止」（与 .md 分支一致），可删除已存的图片文件。
4. **单图失败不中断**：单个图片描述失败，只跳过该图并记 warning（`image_skipped++`），不中断。
5. **同名不重复**：`import_note` 已有同名文件冲突判断（409/ValueError），图片笔记沿用。
6. **保持 SSE 协议 / 前端展示**：改完不影响现有的 `meta → token → done`；前端来源卡片已在 `streamlit_demo.py` 用 `st.image(source["image_path"])` 展示原图，metadata 带 `image_path` 即可。
7. **两套入口同步**：`streamlit_demo.py`（Cloud 用它）和 `backend/app/api/notes.py`（双进程）都要改，否则行为不一致。
8. **`requirements.txt`**：无需新增依赖（图片 Vision 走现有 httpx；不引入 OCR 库）。若用了 `Pillow`（图片尺寸/预处理），确认已加入；不加也可。

---

## 5. 完成标准（验收）

- [ ] 侧边栏「导入笔记」能选图片（jpg/jpeg/png/webp）
- [ ] 上传单张图片 → 图片存到 `data/uploads/<doc_id>/images/`
- [ ] Vision 生成图片描述 → 图片块入库，metadata 含 `image_path`/`is_image_chunk`/`source`/`chunk_id`/`doc_id`
- [ ] 导入结果返回 `image_processed`；图片块能被 `knowledge_to_vector` 写入 Chroma
- [ ] 提问时能检索到该图片块；命中后前端来源卡片显示原图
- [ ] `streamlit_demo.py` 与 `backend/app/api/notes.py` 行为一致
- [ ] `pytest tests/` 全绿（若新增测试，用 mock，无外部费用）
- [ ] 不删除聊天区传图问答
- [ ] 不引入 OSS / OCR 依赖
