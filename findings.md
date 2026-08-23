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
