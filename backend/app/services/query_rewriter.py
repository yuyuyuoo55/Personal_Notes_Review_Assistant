from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# 只有带前文指代的问题才需要额外模型调用补全；完整问题直接检索更快。
CONTEXTUAL_MARKERS = ("刚才", "前面", "上面", "后面", "它", "那个", "这两个")


def query_rewrite(query: str, llm) -> str:
    """将口语化问题改写成适合检索的关键词；失败时返回原问题。"""

    if not any(marker in query for marker in CONTEXTUAL_MARKERS):
        logger.info("问题完整，跳过查询改写")
        return query

    rewrite_prompt = f"""
将以下问题改写为适合检索的关键词形式。
提取核心概念，用空格分隔。
只输出关键词，不要解释。

问题：{query}
关键词：
"""

    try:
        result = llm.invoke(rewrite_prompt)
        rewritten_query = result.content.strip()

        # 模型返回空内容时，也回退原问题
        return rewritten_query or query

    except Exception:
        # 不记录第三方 SDK 原始异常，避免认证信息或请求细节进入日志。
        logger.warning("查询改写失败，已回退原问题")
        return query
