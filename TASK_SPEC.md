# 多模态 + Key 验证改造任务清单（给 codex 执行）

> 本文件是给 codex 的改造说明。请按文件逐一实现，保持现有代码风格（中文注释、SSE 协议、BYOK 设计）不变。
> 目标：让笔记里的图片真正参与检索并可在回答中引用；用户输入 DeepSeek Key 后先验证再解锁控件。

---

## 0. 两个已确认的决策（很重要，别再改为别的方案）

1. **图片存本地，不存 OSS。** 设备图片转 Base64 直接发给 DeepSeek Vision。**不依赖 OSS、不配公网 URL、不做签名。**
   - `.env` / Streamlit Secrets 里的 `OSS_*` 配置可保留但**不再参与图片链路**。
   - `markdown_image_service.py` 里的 `upload_image_to_oss` 调用**不再使用**；改为把图片保存到本地目录，返回本地路径。
2. **前端要能展示被引用的原图。** 回答的来源卡片里，若该块带图，用 `st.image()` 显示原图。

---

## 1. 总设计原则（不要违背）

> **"引用哪张图"由"最终被选中的块"决定，不让模型凭空猜测。**

图片在导入时被转成一段文字描述，与文字块一起存进向量库。检索层（无论精确还是快速模式）本来就用"相关性排序 + Top-K 截断"决定用哪些块：
- **精确模式（混合检索）**：`vector + BM25 → RRF → 精排 → Top-3`。不相关的图片块会在排序中被淘汰，**不需要额外写"过滤不相关图"的逻辑**。
- **快速模式（Agent）**：Agent 在提示词指导下自己判断是否引用带图块。**不需要额外代码做过滤**，只改提示词。
- 最终**只把"被选中的块"里的图片交给生成环节/前端展示**。这样即使召回很多图，只要没进最终结果集，就不会被错误引用。

需要新写的只有 3 类能力：
1. 图片导入时变成"可检索的文本描述"，并把原图本地路径存进块的 metadata。
2. 检索结果里带图时，能把图像信息带到生成环节（模型看图/引用）和前端展示。
3. Key 验证接口 + 前端解锁。

---

## 1.5 流转全景图（务必按这个理解，避免实现错方向）

### 入库时：图片 → 文字描述 → 向量（**不是给图片单独提取"特征向量"**）

> **关键概念**：本项目没有多模态 embedding 模型，所以图片**不能**直接转成向量。正确做法是：
> **图片 → VLM 看图 → 生成一段文字描述 → 这段描述走现有 `text-embedding-v4` → 转成向量 → 存入向量库。**
> 即图片和笔记文字走**同一套向量化**，图片只是"先变成文字"。

```
文档
 ├─ 文字部分 ──> 标题切分成文字块 ──> 向量化 ──┐
 └─ 图片部分 ──> VLM看图 ──> 文字描述(图片块) ─> 向量化 ─┴─> 一起存入向量库
                                                       (图片块 metadata 带 image_path)
```

### 提问时：用户传图 + 文字（图片作为"查询"参与检索）

```
用户上传图片 + 文字问题
   ──> VLM看图 ──> 生成图片文字描述
   ──> 该描述 + 用户文字问题 拼接成"查询"
   ──> 检索(向量 + BM25 + RRF + 精排，两种模式都适用)
   ──> 命中高相关块
         ├─ 文字块 ──> 前端显示文字来源
         └─ 命中块 metadata 带图 ──> 前端显示"文字 + 图片"一起
```

### 图片与笔记内容不符时（重要：不要设计"让模型判断符不符合"）

**让现有相关性/拒答机制自动兜底，不让模型凭空判断：**

- **精确模式**：`rag_service.py` 已有 `has_relevant_material` 判断（向量距离 ≤ `MAX_VECTOR_DISTANCE` 或 BM25 命中）。
  - 图片描述转文字去检索，若与笔记相符 → 召回并回答；
  - 若不符 → 向量距离远 / BM25 不命中 → `has_relevant_material = False` → 走 `no_material_preparation`（拒答："找到足够可靠的资料"）。
- **快速模式**：Agent 在提示词指导下，检索不到时回复"没有找到对应片段，不要编造"。
- **结论**：图片与笔记不符时，系统自动"不知道就说不知道"，**不需要额外写"图片相关性判断"代码**。

---

## 2. 任务 A：图片变文本描述 + 存本地 + metadata 带图

### 文件 1：`backend/app/services/multimodal_service.py`

- 新增函数 `save_image_to_local(image_bytes, content_type, doc_dir) -> str`（替代 `upload_image_to_oss`）：
  - 在指定笔记文件夹下建 `images/` 子目录。
  - 用 `uuid4().hex + 扩展名` 命名图片并写入磁盘。
  - 返回保存后的本地路径。
  - 校验：`validate_image` 继续用于限制类型/大小；此处保持与现有缺陷处理一致。
- `describe_image_url(image_url, api_key)` 保留（仍用于"对图片 URL/路径生成描述"）。注意：DeepSeek Vision 支持 Base64 与图片 URL 两种输入。**为了让模型看图，可改为传入 Base64 data URL（读本地图片转 base64），或传本地 URL。** 建议：`image_path_to_data_url(image_path)` 辅助函数。
- `stream_deepseek_image_answer` 保持不变（聊天图片直答已可用）。

### 文件 2：`backend/app/services/markdown_image_service.py`

- 现状：`enrich_markdown_images` 已做"下载/解析图片 → 上传 OSS → 生成描述 → 回填 Markdown"。
- **要改：**
  1. 删除/绕过 `upload_image_to_oss`，改用 `save_image_to_local`（存本地）。
  2. 图片生成描述后，除了回填 Markdown，还要**额外产出一条"图片块"**，内容为图片描述文本。
  3. 返回结构扩展：除 `MarkdownImageResult` 外，新增 `image_chunks: list[Document]`（或对等结构），每条的 `page_content` = 图片描述，`metadata` 至少含：
     - `source`：所属文档路径
     - `Header 1/2/3`：所在标题路径（与本文件文字块一致）
     - `image_path`：本地图片路径
     - `is_image_chunk`: True（用于识别图片块）
  4. 单张图失败只跳过该图并记 warning，不中断整篇（现有 bug 处理逻辑保留）。

### 文件 3：`backend/app/api/notes.py`（import_note）

- 现状：`enrich_markdown_images` 只回填 Markdown，然后 `load_notes + split_documents + knowledge_to_vector`。
- **要改：**
  1. 调用 `enrich_markdown_images` 后，把返回的 `image_chunks` 与 `split_documents` 的文字块**合并**，一起 `knowledge_to_vector`。
  2. 保证图片块与文字块的 `source`/标题元数据一致，避免检索归属错乱。
  3. `ImportResult` 里 `image_processed / image_skipped / warnings` 逻辑保持不变。

### 文件 4：`streamlit_demo.py`（import_note 单进程版）—— 与本文件 3 一致

- `streamlit_demo.py` 里也有一个 `import_note`（第 96-125 行区间），它直调 `enrich_markdown_images`。
- 同步改动：把 `image_chunks` 合并进 `knowledge_to_vector`，确保单进程 Demo 与后端行为一致。

---

## 3. 任务 B：两种检索模式都能"用带图的块"

> **前置约定（重要）**：查询时若用户**同时上传了图片**，先调用 VLM（`stream_deepseek_image_answer` 或图片描述函数）把图片转成文字描述，再**把该描述 + 用户文字问题**拼接成最终查询，参与检索。检索本身不重复调 VLM。入库/查询的图已在 1.5 全景图定义。

### B1. 精确模式（混合检索）

**文件：`backend/app/services/rag_service.py`（`prepare_rag_answer` 精确分支）**

- 现状：`reranked_results`（第 240-246 行）已经是最终 Top-3 块，天然过滤了不相关图。
- **要改：**
  1. `reranked_results` 里每个块的 metadata 如果带 `image_path`，将其记录到最终 `SourceChunk`（供前端展示）。
  2. 在把块交给 `generate_responses_based_on_the_data`（第 254 行）之前，若块带图，把**图片内容（Base64/描述）一并并入传给模型的上下文**，让模型能引用图中信息。
     - 实现建议：新增一个辅助函数把"带图块的 image_path → 图片描述/base64"并入 `context`；若不想让模型看原始图，至少把该块已有的图片描述文本并入即可。

**文件：`backend/app/services/chat_service.py`（`generate_responses_based_on_the_data`）**

- 现状：第 4-20 行，只把 `reranked_results` 的 `content` 拼成上下文。
- **要改：**
  1. 当某个 result 带 `image_path` / 图片描述时，也把它放进 `data_chunk`（或单独拼接进 `context`），这样模型生成时能看到图片信息。
  2. 提示词里加一句："若参考资料含图片，可在回答中说明图片内容。"

### B2. 快速模式（Agent）

**文件：`backend/app/services/agent_service.py`**

- 现状：`search_personal_notes` 工具已返回 `chunks`（含 metadata），用 `vector_retriever top_k=3` 检索。
- **要改：**
  1. 工具返回的 chunk 若 metadata 带 `image_path`，在 chunk dict 里保留该字段。
  2. `FAST_AGENT_SYSTEM_PROMPT`（第 47-57 行）加一条规则："若检索到的片段含图片且与问题相关，回答时引用该图片信息；无关图片不要引用。"
  3. `_build_source_chunks`（第 175-200 行）把 `image_path` 填入 `SourceChunk`（见任务 C 的字段扩展），供前端展示。

---

## 4. 任务 C：前端展示引用的原图

### 文件：`backend/app/schemas/note.py`

- `SourceChunk`（第 21-26 行）增加可选字段：
  - `image_path: str | None = None`
  - （若用 Base64 展示可另加 `image_data_url: str | None = None`，但本地路径更稳，推荐 `image_path`。）

### 文件：`streamlit_demo.py`

- 渲染来源处（约第 547-556 行，及快速/精确共用的来源展开块）：若 `source.get("image_path")` 存在，则 `st.image(路径)` 显示。
- 视觉上保持与现有 `source-label` 样式一致。

> 注：`streamlit_demo.py` 单进程版里，`SourceChunk` 直接来自 `rag_service` 返回；确保该字段被正确填充即可。

---

## 5. 任务 D：Key 验证接口 + 前端解锁

### D1. 新增后端接口

**文件：新增 `backend/app/api/key_validate.py`（或并入 `chat.py`）**

```python
POST /api/key/validate
Header: X-DeepSeek-API-Key: <user key>

后端拿该 key 去 DeepSeek 发一个最小请求（models 列表或一个极小的 chat）。
→ 200        → {"valid": true,  "message": "您的 DeepSeek API Key 有效，可以使用"}
→ 401/403    → {"valid": false, "message": "API Key 无效，请检查后重试"}
→ 超时/网络  → {"valid": false, "message": "无法连接 DeepSeek，请稍后重试"}
```

- 判断 Key 是否有效的标准：复用 `InvalidApiKeyError` 逻辑（Multimodal `stream_deepseek_image_answer` 已用 401/403 判定）。
- 建议：用 `httpx` 向 `DEEPSEEK_API_BASE_URL` 发 `GET /models` 或一个最小 chat 请求，`timeout=VLM_TIMEOUT_SECONDS`，`trust_env=False`。

**文件：`backend/app/main.py`** — 注册该路由（与现有 `/api/chat`、`/api/notes` 一致）。

### D2. 前端调用 + 解锁控制

**文件：`streamlit_demo.py`**

- 现状：`has_api_key`（约第 257 行）只判断 `bool(st.session_state.deepseek_api_key.strip())`，**不判断对不对**。
- **要改：**
  1. 用户输入 Key 后（或点一个"测试 Key"按钮），调 `/api/key/validate`。
  2. 用 `st.session_state` 记录"验证通过"标志（如 `key_validated`），并缓存 key 用于后续请求。
  3. `has_api_key` 改为"**填入且验证通过**"才为 True；否则上传/提问控件保持禁用。
  4. 验证通过 → 显示 `st.success("您的 DeepSeek API Key 有效，可以开始使用")`；失败 → `st.error(...)`。

> ⚠️ 给 codex：`streamlit_demo.py` 单进程版**没有 FastAPI 后端**，它直调 `rag_service`/`multimodal_service`。因此 Key 验证在单进程版里要**直接调 `stream_deepseek_image_answer` 或建一个本地函数**验证（避免依赖 `/api/key/validate` 接口）。请区分：双进程版走 `POST /api/key/validate`；单进程版走函数直调。

---

## 6. 注意事项 / 潜在 bug（务必防止）

1. **Base64 体积**：图片转 Base64 约膨胀 33%，DeepSeek Vision 有输入上限。`MAX_IMAGE_BYTES`（默认 8MB）要兜底；超限时明确抛"图片过大，已跳过"。
2. **图片描述失败不中断整篇**：单张图描述失败只跳过该图，`image_skipped++`，不能中断整篇导入（保留现有 `ImageProcessingError` 处理）。仅当 **Key 无效**（`InvalidApiKeyError`）时中止整篇并提示（现有逻辑保留）。
3. **图片块去重**：同一张图不要重复生成描述/重复入库（可用 `image_path` 去重）。
4. **本地目录结构**：建议在 `data/uploads/`（即 `UPLOAD_DIRECTORY`）下，每个文档一个文件夹？或统一 `data/uploads/images/`。请遵循现有 `UPLOAD_DIRECTORY` 约定，避免破坏 `list_notes()` 对 `*.md` 的遍历逻辑（`list_notes` 只读 `*.md`，图片放 `images/` 子目录不影响）。
5. **不要破坏现有 SSE 协议**：`meta → token → done` 结构不变；`SourceChunk` 加字段是向后兼容的（新增可选字段，旧前端忽略）。
6. **Key 不落日志/DTO**：延续 BYOK 设计，Key 只在请求内存中用，不写入日志、数据库、DTO。
7. **`requirements.txt`**：如需 `Pillow`（图片处理/Base64），确认已加入；若不需要则不加。

---

## 7. 完成标准（验收）

- [ ] 导入带图的 Markdown → 图片描述生成 → 图片块与文字块一起写入 Chroma；图片存到本地 `data/uploads/...`
- [ ] 提问时（精确/快速模式）：检索命中的带图块，回答能引用图中信息；不相关图片不被引用
- [ ] 前端来源卡片能显示被引用的原图（`st.image`）
- [ ] `POST /api/key/validate` 能正确判断 Key 有效/无效/网络错误
- [ ] 双进程版：前端调接口验证通过后解锁控件；单进程版：函数直调验证后解锁
- [ ] `pytest tests/` 全绿
- [ ] Key 不落日志、不落 DTO
