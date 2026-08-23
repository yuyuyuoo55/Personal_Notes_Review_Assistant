from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# 读取笔记
def load_notes(file_path: str)->list[Document]:
    """接收 Markdown 文件路径，返回 LangChain Document 列表。"""
    text_loader = TextLoader(file_path, encoding="utf-8")
    docs = text_loader.load()

    return docs

# 测试通过
# if __name__ == "__main__":
#     file_path = (
#         r"D:\codex\project\vibe-coding\Personal_Notes_Review_Assistant\task_plan.md"
#     )
#
#     docs = load_notes(file_path)
#
#     print(f"加载了 {len(docs)} 个文档")
#     print(f"内容：{docs[0].page_content}")
#     print(f"元数据：{docs[0].metadata}")