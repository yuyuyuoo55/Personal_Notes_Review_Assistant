from hashlib import sha256

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# 把笔记切成多个 Document,每个片段保留文件名、标题路径和内容。
def split_documents(docs: list[Document])->list[Document]:

    # 切分依据，这里是按照三级标题
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 1. 初始化 Markdown 标题切分器。
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    # 切分文档
    # 一个完整 Document 切成多个 Chunk

    chunks_list=[]

    #第一层循环里切分的根本原因是：docs 是 list[Document]，以后可能一次加载多个文件。
    for doc in docs:
        chunks=markdown_splitter.split_text(doc.page_content)

        # 2. 遍历标题切分得到的每个 Chunk添加metadata
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata.update(doc.metadata)

            # 3. 为每个 Chunk 生成 chunk_id，供向量检索、BM25、RRF 共同使用。
            # 先把三个信息拼成一段字符串：
            # source：来自哪个文件，例如 D:\...\Git.md
            # index：它是这个文件切出的第几个 Chunk
            # chunk.page_content：这个 Chunk 的正文

            source = str(chunk.metadata.get("source", "unknown"))
            raw_value = f"{source}:{index}:{chunk.page_content}"
            # sha256 会把这段任意长度的文本，计算成一串固定长度的哈希值。它不是加密给人看的，而是相当于根据内容生成“指纹”
            #hexdigest()：把哈希结果转成十六进制字符串；
            #[:16]：只取前 16 位，作为较短的 ID。
            chunk.metadata["chunk_id"] = sha256(
                raw_value.encode("utf-8")
            ).hexdigest()[:16]

        chunks_list.extend(chunks)

    return chunks_list

if __name__ == "__main__":
    from note_loader import load_notes

    file_path = (
        r"D:\codex\project\vibe-coding\Personal_Notes_Review_Assistant\task_plan.md"
    )

    docs = load_notes(file_path)
    chunks = split_documents(docs)
    print(chunks)
    print("*"*50)

    print(f"原始文档数量：{len(docs)}")
    print(f"切分后 Chunk 数量：{len(chunks)}")

    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\n===== Chunk {index} =====")
        print(f"内容：\n{chunk.page_content[:200]}")
        print(f"元数据：{chunk.metadata}")
