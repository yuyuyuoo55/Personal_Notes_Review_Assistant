# Progress

## 2026-08-19

- 确认项目目录为空，仅保留用户的 `.idea/` 配置。
- 创建可运行的 FastAPI 健康检查、Streamlit 空壳、pytest 验证、uv 配置与本地密钥模板。
- 未写入真实密钥，未导入个人笔记。
- 首次复制失败：PowerShell 的 `Copy-Item -LiteralPath` 不展开 `*`，目标目录未收到项目文件，`uv sync` 因而找不到 `pyproject.toml`。下一次改为先枚举子项再逐项复制。
- 修正复制方式后，已写入目标项目；`uv sync --group dev` 成功创建 `.venv` 与 `uv.lock`。
- 验证通过：`uv run pytest -q` 为 1 passed；`python -m compileall` 通过。FastAPI 的 TestClient 给出第三方弃用警告，不影响当前健康检查。
- 新增 `启动项目.cmd` 与 `start_all.ps1`：检测 8000/8501 端口后在后台启动服务、写入日志并打开前端页面。
- 启动器首次实测失败：Windows PowerShell 将脚本中的中文提示错误解码，触发字符串解析错误；已改为脚本内部仅用 ASCII 提示，准备重新验证。
- 直接运行 `.ps1` 被系统 Execution Policy 拦截；用户双击的 `.cmd` 已显式传入 `-ExecutionPolicy Bypass`，下一次按真实双击入口验证。
- 双击入口验证成功：`启动项目.cmd` 能启动后端与前端；`GET /api/health` 返回 status=ok，前端 `http://127.0.0.1:8501` 返回 HTTP 200。
- 安装完整 RAG 依赖时，`pkuseg==0.0.25` 构建失败：缺少声明的构建期 numpy。已在项目配置中增加 uv 的 pkuseg/numpy 构建依赖，待重新同步。
- 修正配置后，RAG 依赖同步成功：LangChain、Chroma、DashScope、BM25S、pkuseg、sentence-transformers、Torch 等均可导入；pytest 仍为 1 passed。
- 清理已确认损坏的旧 `websockets-16.1.1.dist-info` 记录后，`uv sync` 已无重复卸载警告，最终 websockets 为 15.0.1。
- 新增 `backend/app/core/config.py`：全项目集中加载 `.env`，提供 DashScope/DeepSeek Key 与 Chroma 持久化路径；未写入真实密钥。
- 新增 `backend/app/core/logger.py`：提供 `get_logger(__name__)`，统一写入控制台与 `logs/app.log`，日志文件按 2MB 滚动并保留 3 个备份。
- Chroma 持久化目录改为优先读取 `.env` 的 `CHROMA_PERSIST_DIRECTORY`，未配置时默认使用项目内 `data/chroma`。
- 开发流程已复核并修正：明确纯向量检索只用于开发基线；最终线上链路先查询改写再检索。补充了稳定 `chunk_id`；移除过早抽象的 `retrieval_pipeline.py`，改为直接在现有 `vector_retriever.py` 中接入查询改写。
# 2026-08-22

- 新增快速模式与精确查找：快速模式仅执行向量检索；精确查找保留查询改写、BM25、RRF、Cross-Encoder 完整链路。
- 问答 API DTO 增加 `mode`，前端已可选择并随请求传递。
- 基础问候不进入检索；旧 Chroma 片段缺少 `chunk_id` 时使用稳定兜底 ID，避免 RRF 内部异常暴露给用户。
- 下一步：用固定问题集记录两种模式的来源命中与分阶段耗时，再讨论参数优化。

# 2026-09-03 BYOK 与多模态升级

- 已读取现有规划文件并确认工作树当前无未提交改动。
- 已建立阶段 7 实施计划，准备分片核对前端/API、Markdown 导入和部署配置。
- 已完成三路只读梳理并形成关键文件地图；确认 BYOK 的模型缓存风险、Markdown 本地图片限制和双部署入口差异。
- 已核对 DeepSeek 官方 Vision 文档并采用 `deepseek-v4-flash-vision-exp`；聊天图片使用用户 Key，Qwen-VL 仅作默认关闭的部署者后备。
- 已完成后端请求级模型隔离、图片 SSE 直答、OSS/VLM 文档增强与安全降级。
- 已完成双进程前端和单进程 Demo 的 BYOK 门禁、聊天图片上传及 Markdown-only 上传约束。
- 已更新依赖、配置模板、评测脚本与 README；第二轮全 Mock 测试为 `9 passed`。
- 已安装并验证 `oss2==2.19.1`；`oss2.Bucket` 支持配置连接超时。
- 真实页面验证通过：Key 输入框为 password；无 Key 时聊天、Markdown 上传和图片上传均禁用；两类上传控件格式互不混用。
- 最终测试 `9 passed`；敏感模式命中 0；`.env` 未被 Git 跟踪；依赖导入和编译通过。
