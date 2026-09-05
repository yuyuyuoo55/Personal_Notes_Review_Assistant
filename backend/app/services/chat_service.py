# 基于资料生成回答
from collections.abc import Generator

def generate_responses_based_on_the_data(
        query:str,
        chat_model,
        reranked_results:list[dict])->Generator[str, None, None]:
    # 1.先拿到资料片段
    data_chunk=[]

    if reranked_results:
        for result in reranked_results:
            content = result["content"]
            image_path = result.get("metadata", {}).get("image_path")
            if image_path:
                content = f"[图片资料：{image_path}]\n{content}"
            data_chunk.append(content)

    # 2. 将多个片段拼成一段上下文资料
    if not data_chunk:
       yield "当前笔记库资料不足。"
       return

    context = "\n\n".join(data_chunk)

    # 3.拼接提示词
    prompt = f"""
    你是个人笔记复习助手。

    请只根据【参考资料】回答【用户问题】。

    回答规则：
    1. 不要使用参考资料之外的知识，不要自行编造。
    2. 先判断参考资料是否能支撑用户问题的核心答案。
    只要资料覆盖核心概念、主要步骤或主要命令，即使不包含所有细节和扩展用法，也应直接基于资料回答，不能因为“不够全面”就说资料不足。
    只有资料与问题核心不匹配、只包含顺带提及的内容，或完全没有可支撑结论的内容时，才回答资料不足。
    资料不足时，回答的第一行必须是：
    当前笔记库资料不足。
    之后按以下顺序简短说明：
    - 资料不足的原因：明确缺少用户问题中的哪一部分。
    - 已检索到的资料：只概括参考资料实际覆盖的主题，不要把这些内容当作完整答案展开讲。
    此时不要先输出一长段局部答案，再在结尾补“资料不足”。
    3. 资料覆盖核心答案时，直接回答问题；回答要清晰，优先解释核心概念，默认控制在 3 到 5 个要点内。
    4. 不要提及“提示词”“上下文”“模型”等内部信息。
    5. 若参考资料含图片，可在回答中说明图片内容。

    【用户问题】
    {query}

    【参考资料】
    {context}
    """

    # 4. 流式调用大模型。
    # stream() 每次返回一小段 AIMessageChunk。
    for message_chunk in chat_model.stream(prompt):
        content = message_chunk.content

        # 有正文时，逐段返回给调用方。
        if content:
            yield content
