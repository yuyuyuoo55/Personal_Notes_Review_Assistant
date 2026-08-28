# 个人笔记复习助手

一个面向个人技术笔记的本地 RAG 学习工具。上传 Markdown 笔记后，可以用自然语言提问，系统会返回带文件名、章节和原文片段的答案；没有可靠依据时明确拒答。

![项目主界面](docs/images/project-overview.png)

## 项目亮点

- **回答可追溯**：答案附带来源文件、标题路径和原文片段，不只返回模型文本。
- **两条检索链路**：快速模式由 Agent 按需检索；精确模式固定执行混合检索与精排。
- **面向中文笔记**：BM25 使用 `pkuseg` 分词，并与向量检索进行 RRF 排名融合。
- **资料不足拒答**：检索结果不可靠时提示补充笔记，不使用联网知识强行回答。
- **本地数据持久化**：原始 Markdown、Chroma 向量索引和 BM25 索引均保存在本机。
- **流式交互**：FastAPI 通过 SSE 返回检索阶段、来源、回答 token 和耗时。

> 当前版本是单机 MVP：仅支持 `.md` 文件；章节小测、批量导入、笔记更新/删除和多用户能力尚未实现。

## 双模式设计

| 模式 | 真实链路 | 适用场景 | 主要取舍 |
| --- | --- | --- | --- |
| 快速模式 `fast` | Agent 判断是否检索 → Chroma 向量 Top-3 → 基于片段回答 | 日常复习、普通追问、短对话 | 延迟较低，但弱关键词问题可能漏召回 |
| 精确查找 `accurate` | 查询改写 → 向量 Top-6 + BM25 Top-6 → RRF → Cross-Encoder → Top-3 → 生成 | 术语查找、命令定位、强调来源的问题 | 召回更稳，但首次需下载精排模型，耗时更高 |

快速模式具有进程内会话记忆；精确模式每次固定执行完整链路，不使用会话记忆。

## 技术架构

```mermaid
flowchart LR
    U[用户] --> UI[Streamlit 前端]
    UI --> API[FastAPI + SSE]

    API --> INGEST[Markdown 导入与标题切分]
    INGEST --> FILES[(本地原文)]
    INGEST --> EMB[DashScope Embedding]
    EMB --> CHROMA[(Chroma)]

    API --> MODE{检索模式}
    MODE -->|快速模式| AGENT[LangChain Agent]
    AGENT -->|按需调用工具| VECTOR[向量检索 Top-3]

    MODE -->|精确查找| REWRITE[查询改写]
    REWRITE --> DENSE[向量检索 Top-6]
    REWRITE --> BM25[BM25 Top-6]
    DENSE --> RRF[RRF 融合]
    BM25 --> RRF
    RRF --> RERANK[Cross-Encoder 精排 Top-3]

    VECTOR --> LLM[DeepSeek 生成]
    RERANK --> LLM
    LLM --> API
```

### 笔记导入流程

```mermaid
flowchart LR
    A[上传 Markdown] --> B[格式/空文件/重名校验]
    B --> C[按 H1/H2/H3 标题切分]
    C --> D[生成稳定 chunk_id]
    D --> E[DashScope Embedding]
    E --> F[写入 Chroma]
    F --> G[下次精确查询时重建 BM25]
```

## 核心代码地图

| 模块 | 文件 | 负责内容 |
| --- | --- | --- |
| API 入口 | `backend/app/main.py` | 注册健康检查、笔记和问答路由 |
| 笔记导入 | `backend/app/api/notes.py` | 上传校验、保存、切分、向量化与失败回滚 |
| 双模式编排 | `backend/app/services/rag_service.py` | 快速/精确分流、阈值判断、混合检索与生成 |
| 快速 Agent | `backend/app/services/agent_service.py` | 工具调用、向量 Top-3、会话记忆和流式事件 |
| Markdown 切分 | `backend/app/services/note_splitter.py` | 标题感知切分和稳定 `chunk_id` |
| 混合检索 | `bm25_retriever.py` / `rrf_fusion.py` | 中文关键词召回与排名融合 |
| 精排 | `backend/app/services/reranker.py` | `BAAI/bge-reranker-base` Cross-Encoder 精排 |
| 向量存储 | `backend/app/storage/vector_store.py` | DashScope Embedding 和 Chroma 持久化 |
| 前端 | `frontend/app.py` | 导入、模式切换、SSE 解析、来源卡片和耗时展示 |
| 回归评测 | `eval_10questions.py` | 双模式逐题请求、规则判定和 Markdown 报告生成 |

## 回归评测

仓库提供一份 10 题轻量回归脚本。它不是通用 RAG 准确率评测：其中 9 题可按“来源文件是否命中”或“是否出现拒答措辞”自动判定，第 10 题需要人工检查回答质量。

一次本机回归快照（2026-08-24）：

| 模式 | 可自动判定题 | 规则命中 | 未命中案例 |
| --- | ---: | ---: | --- |
| 快速模式 | 9 | 7/9 | 2 道资料不足问题未按预期拒答 |
| 精确查找 | 9 | 9/9 | 无 |

完整逐题结果见 [`eval_result_20260824_010958.md`](eval_result_20260824_010958.md)。这里的 `9/9` 表示当前规则命中，不等同于答案准确率、召回率或生产环境指标；单次耗时也会受到网络、模型缓存和 API 状态影响。

### 这份结果如何跑出来

1. 准备并导入与 `eval_10questions.py` 中 `QUESTIONS` 对应的脱敏 Markdown 笔记。
2. 启动后端，确保 `http://127.0.0.1:8000/api/health` 可访问。
3. 执行：

```powershell
uv run python eval_10questions.py
```

4. 脚本对每道题分别请求 `fast` 和 `accurate` 模式，解析 SSE 中的来源与回答。
5. `file:xxx` 检查来源文件名，`refuse` 检查拒答关键词，`any` 留给人工判断。
6. 运行后生成 `eval_result_YYYYMMDD_HHMMSS.md`，还需人工复核回答是否真的被来源支持。

> 出于隐私考虑，个人笔记和本地索引未提交到仓库，因此新克隆项目不能直接复现上述固定分数。请先替换为自己的脱敏测试笔记，并相应修改 `QUESTIONS`。

## 从零运行

### 1. 环境要求

- Windows 10/11（仓库提供 Windows 一键启动脚本）
- Python `3.12.x`
- [uv](https://docs.astral.sh/uv/)
- 可访问 DeepSeek、DashScope 和 Hugging Face

精确模式首次使用会下载 `BAAI/bge-reranker-base`，需要额外等待；后续优先使用本机缓存。

### 2. 克隆并安装依赖

```powershell
git clone https://gitee.com/yuyuyuoo55/personal-note-review-assistant.git
cd personal-note-review-assistant
uv sync
```

### 3. 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
DEEPSEEK_API_KEY=YOUR_API_KEY_HERE
DASHSCOPE_API_KEY=YOUR_API_KEY_HERE
```

不要把真实 Key 写入 README、截图、日志或 Git 提交。

### 4. 启动项目

推荐双击根目录的 `启动项目.cmd`。启动器会：

- 检查 8000、8501 端口；
- 后台启动 FastAPI 与 Streamlit；
- 写入 `logs/backend.log`、`logs/frontend.log`；
- 打开 `http://127.0.0.1:8501`。

也可以手动启动。第一个 PowerShell：

```powershell
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

第二个 PowerShell：

```powershell
uv run streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

访问地址：

- 前端：`http://127.0.0.1:8501`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/health`

### 5. 使用步骤

1. 在左侧上传一份非空 `.md` 笔记。
2. 点击“导入到笔记库”，等待切分与向量化完成。
3. 选择“快速模式”或“精确查找”。
4. 输入问题，查看回答、来源片段和本次耗时。

同名文件会返回 409；当前版本不支持覆盖导入，请先在本机数据目录中处理旧数据后再导入。

## 测试与验证

```powershell
uv run pytest -q
```

当前自动化测试包含：

- `GET /api/health` 健康检查；
- logger 命名行为。

本仓库当前验证结果为 `2 passed`。它们属于工程烟雾测试，RAG 效果需结合上面的 10 题脚本与人工验收判断。

## 工程结构

```text
Personal_Notes_Review_Assistant/
├─ backend/
│  └─ app/
│     ├─ api/                 # FastAPI 路由
│     ├─ core/                # 配置与日志
│     ├─ schemas/             # 请求/响应 DTO
│     ├─ services/            # 切分、检索、融合、精排、Agent、生成
│     └─ storage/             # Chroma 与 Embedding
├─ frontend/app.py            # Streamlit 前端
├─ tests/                     # 工程烟雾测试
├─ docs/images/               # README 展示图片
├─ eval_10questions.py        # 双模式回归脚本
├─ eval_result_*.md           # 单次评测快照
├─ start_all.ps1              # Windows 启动器
├─ 启动项目.cmd               # 双击入口
├─ pyproject.toml             # uv 项目与依赖配置
└─ .env.example               # 脱敏配置模板
```

运行时会创建 `data/uploads`、`data/chroma`、`data/bm25` 和 `logs`。这些目录以及 `.env` 已加入 `.gitignore`，不会上传个人笔记、索引、日志或密钥。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/notes` | 查询已导入笔记及片段数 |
| `POST` | `/api/notes/import` | 上传单个 `.md` 文件，表单字段名为 `file` |
| `POST` | `/api/chat` | SSE 问答；支持 `fast` / `accurate` 模式 |

`POST /api/chat` 请求示例：

```json
{
  "query": "Git 的分支有什么用？",
  "mode": "accurate",
  "conversation_id": "your-session-id"
}
```

## 已知边界

- 仅支持单个 Markdown 文件导入，不支持 PDF、批量导入、更新和删除。
- 快速模式会话记忆保存在进程内，后端重启后清空。
- 精确模式的 Cross-Encoder 首次加载较慢，且当前没有最低精排分阈值。
- 自动评测依赖特定测试笔记，固定分数不能直接迁移到其他知识库。
- 当前没有用户系统、权限隔离、云端同步或生产部署配置。
- 章节小测仍是界面中的下一阶段规划，不属于已实现能力。

## 安全说明

- `.env`、个人笔记、向量索引、BM25 索引和日志默认不提交。
- 上传前请先移除笔记中的姓名、账号、公司内部信息等隐私内容。
- 模型问答与向量化会调用外部 API；敏感资料不应直接导入。
- 项目不联网搜索补充答案，但模型服务本身仍是外部依赖。
