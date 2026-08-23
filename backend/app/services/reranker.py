from sentence_transformers import CrossEncoder

def cross_encoder_reranker_index(query:str,
                                 rrf_results:list[dict],
                                 cross_encoder:CrossEncoder,# 已经初始化好的模型
                                 top_k:int=5)->list[dict]:

    # 1.将用户问题和对应的片段组成一个列表返回
    pairs= []

    # 二维列表。
    # 每个内部列表代表一组“问题 + 候选片段”
    for chunk in rrf_results:
        pair=[
            query,
            chunk["content"]
        ]
        pairs.append(pair)

    # 2.将用户问题和检索片段发送给大模型判断相关性
    score_list=cross_encoder.predict(pairs)

    # 3.将分数写回对应的chunk
    for i,chunk in enumerate(rrf_results):
        chunk["rerank_score"]=score_list[i]

    # 4.根据rerank_score进行重排
    rerank_results=sorted(
        rrf_results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # 5.返回前 top_n 个
    return rerank_results[:top_k]


if __name__ == "__main__":
    # 1. 初始化 Cross-Encoder 模型。
    # 第一次运行会从 Hugging Face 下载模型，后续会使用本机缓存。
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

    # 2. 模拟 RRF 融合后的候选片段。
    query = "为什么 RRF 不直接相加向量分数和 BM25 分数？"
    rrf_results = [
        {
            "id": "chunk-1",
            "content": "RRF 只根据每一路检索的排名计算融合分数，避免直接比较不同尺度的原始分数。",
            "metadata": {"source": "rag.md"},
            "rrf_score": 0.032,
        },
        {
            "id": "chunk-2",
            "content": "Chroma 是一个本地向量数据库，可以保存文本向量和元数据。",
            "metadata": {"source": "rag.md"},
            "rrf_score": 0.031,
        },
        {
            "id": "chunk-3",
            "content": "向量检索和 BM25 的分数尺度不同，不能直接相加。RRF 通过排名完成融合。",
            "metadata": {"source": "rag.md"},
            "rrf_score": 0.030,
        },
    ]

    # 3. 执行精排，只保留分数最高的前两条。
    rerank_results = cross_encoder_reranker_index(
        query=query,
        rrf_results=rrf_results,
        cross_encoder=cross_encoder,
        top_k=2,
    )

    # 4. 验证结果数量与精排分数字段。
    assert len(rerank_results) == 2
    assert all("rerank_score" in item for item in rerank_results)

    # 5. 打印精排结果。
    print(f"问题：{query}")

    for rank, result in enumerate(rerank_results, start=1):
        print(f"\n===== Top {rank} =====")
        print(f"chunk_id：{result['id']}")
        print(f"精排分数：{result['rerank_score']:.4f}")
        print(f"RRF 分数：{result['rrf_score']:.4f}")
        print(f"内容：{result['content']}")
