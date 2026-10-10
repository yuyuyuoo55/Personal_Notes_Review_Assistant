"""只为回答节点调用真实model.stream，增量交给Graph custom通道。"""
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.utils import message_chunk_to_message
from langgraph.config import get_stream_writer


class StreamingAnswerModel:
    def __init__(self, model):
        self.model = model

    def invoke(self, messages):
        """【业务需求】在完整回答前发送真实模型增量，保留原回答校验入口。
        【输入】Daily/RAG原有prompt消息。
        【执行步骤】逐块调用stream、发送custom事件、累积AIMessage；异常原样上抛。
        【输出】完整AIMessage供原业务检查；不将完整回答切片伪造Token。
        """
        writer = get_stream_writer()
        combined = None
        for chunk in self.model.stream(messages):
            if not isinstance(chunk, AIMessageChunk) or not isinstance(chunk.content, str):
                raise ValueError("回答流必须是文本AIMessageChunk。")
            combined = chunk if combined is None else combined + chunk
            if chunk.content:
                writer({"event": "token", "text": chunk.content})
        if combined is None:
            raise ValueError("模型未返回回答增量。")
        return message_chunk_to_message(combined)
