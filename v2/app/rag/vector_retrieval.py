
"""
【模块功能】
连接 Chroma 向量数据库，并根据用户问题检索相关笔记。

【主要流程】
1. 加载环境变量。
2. 初始化 DashScope Embedding 模型。
3. 连接 Chroma 向量数据库。
4. 根据问题执行向量检索。
5. 返回笔记片段和检索距离。
"""

import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma


# =============== Step 1：加载环境变量 ===============

load_dotenv()


# =============== Step 2：配置向量数据库 ===============

# 1. 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 2. 设置 V2 的 Chroma 数据保存路径
DEFAULT_DIRECTORY = PROJECT_ROOT / "v2" / "data" / "chroma"

# 3. 设置集合名称
COLLECTION_NAME = "v2_notes"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024


# =============== Step 3：定义检索结果的数据结构 ===============

class VectorResult(TypedDict):
    """一条向量检索结果包含的数据。"""

    id: str | None
    content: str
    metadata: dict[str, Any]
    distance: float


# =============== Step 4：初始化向量数据库 ===============

def create_vector_store(directory=None, *, api_key=None) -> Chroma:
    """
    【业务需求】
    初始化 Embedding 模型和独立的 V2 Chroma 向量数据库。
    【输入】可选数据库目录与已解析的密钥，默认使用 v2/data/chroma。

    【执行步骤】
    1. 使用已有环境配置创建 Embedding 模型。
    2. 确定 V2 数据库保存路径。
    3. 使用 LangChain Chroma 连接数据库并自动获取或创建集合。
    【输出】可用于写入和查询的 Chroma 对象。
    """
    if api_key is None:
        # 独立脚本也遵循系统凭据优先、环境配置回退；不访问其他服务的凭据。
        from ..api.credentials import KeyManager, WindowsCredentials
        api_key = KeyManager(WindowsCredentials() if os.name == "nt" else None).effective("dashscope")
    # 1. 创建 Embedding 模型，沿用写入和查询共用的模型配置。
    embedding = DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=api_key,
    )

    # 2. 默认使用独立 V2 目录；测试可传入临时目录。
    database_path = Path(directory) if directory is not None else DEFAULT_DIRECTORY

    # 3. Chroma 自动处理底层连接和集合创建。
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(database_path),
    )
    return vector_store


# =============== Step 5：执行向量检索 ===============

def retrieve_vectors(
    query: str,
    vector_store: Chroma,
    top_k: int = 3
) -> list[VectorResult]:
    """
    【业务需求】
    根据用户改写后的问题，从 Chroma 中检索最相关的笔记片段。

    【输入】
    query：改写后的用户问题。
    vector_store：已经初始化的 Chroma 对象。
    top_k：最多返回多少条笔记。

    【执行步骤】
    1. 检查问题是否合法。
    2. 检查 top_k 是否为正整数。
    3. 执行向量相似度检索。
    4. 遍历检索结果。
    5. 提取笔记内容、元数据和距离。
    6. 返回检索结果列表。
    """

    # 1. 检查用户问题
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串")

    if not query.strip():
        raise ValueError("query 不能为空")

    # 2. 检查 top_k
    if type(top_k) is not int:
        raise ValueError("top_k 必须是整数")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 3. 执行向量检索
    # 内部自动将 query 转成向量，再到 Chroma 中检索
    matches = vector_store.similarity_search_with_score(
        query=query.strip(),
        k=top_k
    )

    # 4. 创建结果列表
    results = []

    # 5. 遍历检索结果
    for document, distance in matches:

        # 获取当前笔记片段的基本信息
        result = {
            "id": document.id,
            "content": document.page_content,
            "metadata": dict(document.metadata),
            "distance": float(distance)
        }

        # 将当前结果添加到列表
        results.append(result)

    # 6. 返回检索结果
    return results
