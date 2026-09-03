# Findings

- 目标目录在初始化前只有 `.idea/`，没有代码或依赖配置。
- 本机可用 Python 3.12.13 与 uv 0.12.0。
- `D:\PythonProject\.venv` 已安装 FastAPI、Uvicorn、Streamlit、LangChain 与 Chroma；新项目仍使用自己的 uv 虚拟环境，避免耦合旧学习环境。
- 当前只安装 Web 基础依赖；RAG 依赖应在阶段 2 按实际模块加入。
- 复制目录时不可把带 `*` 的路径传给 PowerShell `-LiteralPath`；需用 `Get-ChildItem` 枚举后复制。
- Windows + Python 3.12 安装 `pkuseg==0.0.25` 时，它未声明构建期 numpy 依赖；在 `pyproject.toml` 的 `tool.uv.extra-build-dependencies` 中为 pkuseg 显式添加 numpy 后再同步。
# 双模式设计结论

- 纯向量检索不是废弃的临时代码，而是快速模式的正式链路。
- 混合检索和 Cross-Encoder 只在精确查找中使用，避免将更高延迟强加给普通问题。
- 两种模式共享笔记导入、切分、Chroma 和“仅依据资料回答”的边界；区别只在在线检索与排序步骤。

# BYOK 与多模态升级发现（2026-09-03）

- 当前分支已有 2026-08-28 之后的部署修复，`uv.lock` 已移除且依赖可能已迁移到 `requirements.txt`；实现必须以当前工作树为准，不能沿用旧快照。
- 历史经验：浏览器可访问的图片 URL 不代表模型服务一定能拉取；OSS URL 必须公网可达，失败时需降级并跳过图片。
- 历史安全经验：OSS 配置对象和密钥不得整体写入日志。
- BYOK 必须走独立请求头，不能进入 Pydantic DTO、Prompt、SSE、日志或持久化。
- 当前 `get_chat_model()` 无参数缓存和快速 Agent 全局单例会绑定首个 Key，必须取消跨请求模型/Agent复用。
- 聊天图片建议使用独立 `POST /api/chat/image` multipart 路由，响应继续采用 SSE。
- Markdown 图片增强应发生在切分前；任意单图失败只产生 warning，不中断整篇导入。
- Markdown 单文件上传无法携带用户电脑上的本地图片；可靠支持范围应限定为 HTTPS 图片和内嵌 data URI，本地绝对路径跳过。
- 当前实际依赖权威是 `requirements.txt`；`pyproject.toml` 不含运行依赖，README 现有 `uv sync` 指令需要修正。
- 用户 DeepSeek Key 无法支付 Qwen-VL 调用；部署者 DashScope Key只用于 Markdown 文档图片识别，聊天图片不能静默回退并消耗部署者额度。
- 2026-09-03 官方 DeepSeek Vision 文档确认 `deepseek-v4-flash-vision-exp` 支持 Base64 data URL、外部 HTTP(S) URL 和 Files API；聊天图片可完全使用用户 BYOK Key。
- Markdown 图片识别也可优先使用用户 DeepSeek Key；Qwen-VL 后备必须显式配置，默认关闭，且其费用由部署者承担。
- DeepSeek Vision 仅允许图片出现在 user 消息，支持 JPEG/PNG/GIF/WebP；实现需限制 MIME 和请求体大小。

## 关键文件地图

| 文件 | 角色 | 下一步读取 | 原因 | 来源 |
| --- | --- | --- | --- | --- |
| `frontend/app.py` | 本地前端 | Yes | BYOK 门禁、MD 上传和聊天图片入口 | byok_frontend_api |
| `backend/app/api/chat.py` | SSE 聊天 API | Yes | 文本请求头与图片 multipart 接口 | byok_frontend_api |
| `backend/app/services/rag_service.py` | 模型与双模式编排 | Yes | 消除 Key 缓存风险 | byok_frontend_api |
| `backend/app/services/agent_service.py` | 快速 Agent | Yes | 消除首个用户模型跨请求复用 | byok_frontend_api |
| `backend/app/api/notes.py` | Markdown 导入 | Yes | 插入图片增强并保持失败回滚 | markdown_images |
| `backend/app/services/note_loader.py` | 文本加载 | Yes | 确认增强文本进入切分的位置 | markdown_images |
| `backend/app/core/config.py` | 后端私密配置 | Yes | OSS/Qwen 配置集中读取 | config_tests_docs |
| `requirements.txt` / `.env.example` | 依赖与模板 | Yes | 增加 oss2 和部署者配置 | config_tests_docs |
| `streamlit_demo.py` | 云端单进程版 | Maybe | 判断是否需同步支持或明确边界 | config_tests_docs |
| `tests/` | 回归验证 | Yes | 新增全 Mock 安全与多模态测试 | config_tests_docs |
