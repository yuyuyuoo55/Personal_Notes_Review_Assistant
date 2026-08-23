from pydantic import BaseModel
from typing import Literal

class ImportResult(BaseModel):
    """导入一份笔记后的结果。"""
    file_name: str # 成功或失败的原始文件名
    chunk_count: int # 本次切分后写入的 Chunk 数量；失败时为 0
    status: Literal["success", "failed"] # 	导入最终状态，避免随意传字符串
    error_msg: str | None

class NoteSummary(BaseModel):
    """笔记列表中展示的一条摘要。"""
    note_id: str # 系统内笔记唯一标识，后续删除/重建索引时使用
    file_name: str # 前端展示的笔记文件名
    chunk_count: int # 该笔记当前包含的 Chunk 数量

class SourceChunk(BaseModel):
    """回答引用的一段原始笔记。"""
    file_name: str # 来源文件名
    header_path: list[str] # 标题层级，例如 ["项目开发计划", "目标"]
    chunk_id: str # 命中的稳定片段标识，用于追溯
    content_preview: str # 在前端来源卡片中展示的原文预览

# if __name__ == "__main__":
#     import_result = ImportResult(
#         file_name="task_plan.md",
#         chunk_number=2,
#         status="success",
#     )
#
#     note_summary = NoteSummary(
#         note_id="note-001",
#         note_name="task_plan.md",
#         chunk_number=2,
#     )
#
#     source_chunk = SourceChunk(
#         chunk_id="task_plan-001",
#         file_name="task_plan.md",
#         header_path=["项目开发计划", "目标"],
#         content_preview="搭建个人笔记复习助手的可运行基础环境……",
#     )
#
#     print("===== ImportResult =====")
#     print(import_result.model_dump_json(indent=2))
#
#     print("\n===== NoteSummary =====")
#     print(note_summary.model_dump_json(indent=2))
#
#     print("\n===== SourceChunk =====")
#     print(source_chunk.model_dump_json(indent=2))