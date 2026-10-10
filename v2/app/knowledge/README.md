# Phase 15 / Phase 17A 笔记导入

Phase17A 已实现 Markdown 内联本地图片与单张图片；Phase17B 已实现图文 ZIP 后端与离线回归。
视觉识别离线测试使用 Fake Vision，不能代表真实模型识别质量通过。

## 调用

从项目根目录运行 Python 时，把 `v2` 放入 `sys.path`；也可从 `v2` 目录启动：

```python
from app.knowledge.note_import import import_markdown
from app.knowledge.image_import import import_markdown_images, import_image
from app.rag.vector_retrieval import create_vector_store

# 纯文本入口保持 Phase15 原行为，图片引用不会自动触发 Vision。
result = import_markdown("notes.md")

# 以下会调用真实 Vision/Embedding，执行前须明确确认费用和请求预算。
result = import_markdown_images("notes.md")
result = import_image("diagram.png")

# 外部图片目录必须显式允许；默认仅允许 Markdown 所在目录及其子目录。
result = import_markdown_images("notes.md", allowed_image_roots=[r"C:\notes\images"])
```

图文入口返回 `status`（imported / updated / unchanged / failed）、`source`、
`filename`、`chunk_count`、图片总数/处理数/缓存数、`vision_calls`、`warnings`。
`vision_calls` 是视觉函数尝试次数，不能替代供应商 HTTP 计数或账单。

## 数据与失败规则

- 原始文件不修改，图片仍保留在原路径；未复制图片或创建 OSS 资源。
- 图像描述是模型提取结果，正文明确标记；`uncertain_details` 只存 metadata/结果，不进入检索正文。
- 每个图像描述继承原文 Header1/2/3；整篇连续 chunk_index、source 与 ID 校验兼容 Phase16。
- 单图来源是图片原路径，Header1 为文件名，可以 list_notes/list_chapters/read_chapter。
- Chroma 始终使用原 V2 `v2/data/chroma`、`v2_notes`，不改 Embedding 模型和 RAG 算法。
- 图片字节、Vision 模型名和提示词版本不变时复用同来源的成功描述。
- 重复导入无变化时不再次向量化。图片变化或删除时写新 ID，再删除同来源过期 ID。
- 所有图片先校验；任一图片校验/识别失败，明确返回 failed，整份本次更新不写库，旧索引保留。
  首次导入失败也不写入“看起来完整”的部分笔记，修正后可显式重试。
- Chroma add/delete 不是事务；存储中途失败不保证自动回滚，返回 failed 后应核对记录，不能假装完成。
- 图片实际文件被删除但 Markdown 仍引用它属于错误；移除 Markdown 引用后再导入才清理对应旧记录。
- 使用纯文本入口重新导入图文笔记会恢复文字索引并移除旧图片描述；需要图片知识时保持使用图文入口。

## 范围与限制

支持内联 `![说明](relative.png)`、角括号路径、标题、PNG/JPG/JPEG/WebP；代码围栏/行内代码不识别为图片。
17A 暂不支持 GIF、多帧、HTML 图片、引用式 Markdown 图片、复杂嵌套括号路径、远程 URL、UNC；17B 的 ZIP 入口额外支持 Typora HTML img。
路径经 resolve 后校验允许目录，包含符号链接目标；不能因为原文写了绝对路径就允许读取任意文件。
图片最多20个引用、单图10MB、合计50MB、20百万像素/最长边8192；Markdown最大2MB。
来源图像若迁移路径，metadata不会自动跟随，需要重新导入。

视觉配置单独读取 `DEEPSEEK_VISION_MODEL`，缺省 `deepseek-flash`；不使用 `DEEPSEEK_TEXT_MODEL` 代替。
当前本地配置为旧名 `deepseek-v4-flash-vision-exp`；官方文档说明该别名仍接受并路由至当前 Flash。
实际请求使用 user 的 Base64 `image_url` 与 JSON 输出，httpx transport 无重试，无 Qwen 自动后备。
参见 [DeepSeek 官方视觉文档](https://api-docs.deepseek.com/guides/vision/)。
实际 Key 权限、响应兼容性、识别准确性仍待单独授权真实测试。
默认多模态 Embedding 设置仅尝试一次；注入 store 的 SDK/HTTP 重试与总预算由验收调用方限制。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_image_import.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_quiz.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_quiz_grading.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_graph.py --offline --note-import
```

本阶段不开发FastAPI/Streamlit，不安排个人学习验收。


## Phase17B 图文 ZIP

```python
from app.knowledge.zip_import import inspect_zip, import_zip
manifest = inspect_zip("notes.zip")  # 只读安全检查，不调用模型。
# 默认真实 Vision/Embedding，调用前须授权总预算；测试注入离线存储和视觉函数。
result = import_zip("notes.zip", vector_store=store, vision=vision_function)
```

- 先检查全部成员，再有界读取 UTF-8 Markdown 与实际引用的 PNG/JPEG/WebP。ZIP 体积最多100MB、512条目、展开总量200MB、压缩比200；Markdown单份2MB、单图10MB、引用图片最多100张/合计50MB。
- 拒绝路径穿越、绝对路径、盘符/ADS、Windows保留名、重复大小写路径、文件/目录冲突、链接、特殊/加密文件与不支持扩展名。只将引用图片复制到随机临时文件，不使用 extractall，也不读取ZIP外资源。
- 支持内联 Markdown 图片和带引号 src 的 Typora HTML img；代码围栏和行内代码不识别为图片。沿用现有三级标题与800/100切分。
- 外部/越界引用或缺失包内图片产生 warnings；其余文字及有效图片可写入，status=partial、write_status记录实际 imported/updated/unchanged，不能视为完整导入。格式/模型失败在写入前停止；存储失败需核对索引。
- ZIP原路径为source，保存zip_path/markdown_member/image_member；image_path用稳定zip:来源!/成员标识，不是可直接打开的本地图片路径。原ZIP应保留，临时图片不作为长期来源。
- 同成员/字节/模型/提示词版本复用描述；同内容重复导入不向量化。更新成功后只清理该ZIP来源的旧ID。ZIP改名/移动会成为另一来源。
- 多份Markdown按成员名排序；同三级标题的章节读取会聚合同ZIP的匹配片段，暂不提供按Markdown成员单独选择小测章节。引用式图片、复杂嵌套括号、GIF及HTML实体路径尚不支持。
- 默认真实Embedding仅约束外层尝试；正式付费批量验收仍须由调用方限制底层HTTP/SDK重试并统计总请求，不能把vision_calls作为账单请求数量。

离线测试：`.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_zip_import.py`。
17B没有执行真实批量Vision/Embedding，不代表64张图识别质量已经验收。

## 统一文本向量服务
Markdown、图片描述、ZIP与查询均使用 DashScope `text-embedding-v4` / 1024维 / `v2_notes`。
图片仍先经 DeepSeek Vision 提取文字，再向量化；配置 Key 不触发重新导入或修改已存向量。
