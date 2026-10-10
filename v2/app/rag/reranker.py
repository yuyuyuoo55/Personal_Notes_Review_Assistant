"""Phase 8：使用 Cross Encoder 对 RRF 候选重排序，不生成答案。"""

from functools import lru_cache

from .rrf_fusion import FusedResult


RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


class RerankedResult(FusedResult):
    """保留全部融合字段，额外记录 Cross Encoder 相关性分数。"""

    rerank_score: float


@lru_cache(maxsize=1)
def get_reranker_model():
    """
    【业务需求】
    首次使用时加载 Cross Encoder，后续查询复用模型。
    【输入】无参数，使用与 V1 相同的模型名称。
    【执行步骤】
    1. 在需要精排时导入 CrossEncoder。
    2. 优先加载本机缓存，缓存缺失时下载同一个模型。
    【输出】缓存的 CrossEncoder 对象；加载失败时抛出异常。
    """
    # 1. 延迟导入，Daily Chat 和空候选无需加载模型依赖。
    from sentence_transformers import CrossEncoder

    # 2. 沿用 V1 的本机缓存优先方式，lru_cache 负责复用成功加载的模型。
    try:
        return CrossEncoder(RERANKER_MODEL_NAME, local_files_only=True)
    except OSError:
        return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_candidates(
    query: str,
    fused_results: list[FusedResult],
    top_k: int = 3,
) -> list[RerankedResult]:
    """
    【业务需求】
    对 RRF 候选重新评分，选出最相关的 TopK 片段，不修改原始候选。
    【输入】改写后的问题、RRF 融合列表、TopK（沿用 V1 实际值 3）。
    【执行步骤】
    1. 空候选直接返回；检查问题和 TopK。
    2. 获取缓存模型，构建问题与片段正文的输入对。
    3. 调用 predict 获取每个输入对的相关性分数。
    4. 复制每个候选并加入 rerank_score，保留全部原始字段。
    5. 按新分数降序排序并截取 TopK。
    【输出】含 rerank_score 的片段列表，不调用生成式模型。
    """
    # 1. 没有候选时不加载模型；有候选时检查必要输入。
    if not fused_results:
        return []
    if not isinstance(query, str) or not query.strip():
        raise ValueError("重排序需要非空的 rewritten_query。")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数。")

    # 2. 复用模型，将每条候选与同一个改写问题配对。
    cross_encoder = get_reranker_model()
    pairs = []
    for chunk in fused_results:
        pairs.append([query, chunk["content"]])

    # 3. predict 按输入顺序返回相关性分数，越大越相关。
    score_list = cross_encoder.predict(pairs)

    # 4. 创建新字典，避免把 rerank_score 写入 State 原有的 fused_results。
    scored_results = []
    for index, chunk in enumerate(fused_results):
        result = dict(chunk)
        result["metadata"] = dict(chunk["metadata"])
        result["rerank_score"] = float(score_list[index])
        scored_results.append(result)

    # 5. 只按重排序分数排序，不增加阈值或新的检索策略。
    reranked_results = sorted(
        scored_results,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )
    return reranked_results[:top_k]
