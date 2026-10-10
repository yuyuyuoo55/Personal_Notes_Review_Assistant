"""仅联调测试使用的真实 HTTP 服务：Fake 模型、Mock 供应商、临时 Chroma。"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api.credentials import KeyManager, WindowsCredentials, verify_key
from app.api.balance import fetch_balance
from app.api.main import create_app
from app.api.runtime import Runtime
from app.knowledge.note_import import import_markdown
from app.quiz.schemas import GeneratedQuiz, QualityReview, ShortAnswerAssessment, ReportContent
from smoke_image_import import OfflineEmbedding, FakeVision
import app.graph as graph_module


def mock_probe(request):
    # 真正经过验证代码的 HTTP 协议路径，但 HTTP Transport 完全离线。
    if "bad" in request.headers.get("Authorization",""):
        return httpx.Response(401,json={"error":"test authentication rejected"})
    if request.url.path=="/user/balance":
        if (Path(os.environ.get("V2_E2E_ROOT","."))/"balance-failure").exists():
            return httpx.Response(503,json={"error":"Mock only"})
        return httpx.Response(200,json={"is_available":True,"balance_infos":[{
            "currency":"CNY","total_balance":"12.50","granted_balance":"2.50","topped_up_balance":"10.00"}]})
    if "dashscope" in request.url.host:
        return httpx.Response(200,json={"output":{"embeddings":[{"embedding":[1.0]+[0.0]*1023,"text_index":0}]},"usage":{"total_tokens":8}})
    return httpx.Response(200,json={"choices":[{"message":{"content":"OK"}}],"usage":{"total_tokens":8}})


class E2EModel:
    """可预测的集成夹具，不声称验证真实模型质量。"""
    failed_answers = set()
    def with_structured_output(self,schema,method):
        class Structured:
            def bind(self,**kwargs):
                return self

            def invoke(self,messages):
                if schema.__name__=="RouteResult":
                    return schema(route="daily" if "你好" in messages[-1].content else "knowledge")
                payload=json.loads(messages[-1].content)
                if schema is GeneratedQuiz:
                    questions=[]
                    chunks=payload["chapter"]
                    for kind in ("single_choice","short_answer"):
                        for number in range(payload["requested"][kind]):
                            index=len(questions)%len(chunks)
                            chunk=chunks[index]
                            quote=next(line.strip() for line in chunk["content"].splitlines()
                                       if line.strip() and not line.strip().startswith("#"))
                            point=" / ".join(chunk["headings"].values())+str(len(questions)+1)
                            questions.append({"question_type":kind,"question":f"结合笔记，{point} 的要点是什么？",
                                "knowledge_point":point,"source_chunk_ids":[chunk["chunk_id"]],
                                "evidence":[{"chunk_id":chunk["chunk_id"],"quote":quote}],
                                "options":{"A":"依据原文理解术语关系","B":"忽略原文中的限定条件","C":"交换相关步骤的先后","D":"混淆两个不同知识概念"} if kind=="single_choice" else {},
                                "correct_answer":"A" if kind=="single_choice" else "",
                                "explanation":"应依据原文理解，不能交换步骤或忽略条件。" if kind=="single_choice" else "",
                                "reference_answer":quote if kind=="short_answer" else "",
                                "key_points":[quote] if kind=="short_answer" else []})
                    return schema(questions=questions,insufficient_reason="")
                if schema is QualityReview:
                    return schema(decisions=[{"candidate_id":q["candidate_id"],"approved":True,"reason":"Fake 审核"} for q in payload["candidates"]])
                if schema is ShortAnswerAssessment:
                    if "首次失败" in payload["user_answer"] and payload["user_answer"] not in E2EModel.failed_answers:
                        E2EModel.failed_answers.add(payload["user_answer"])
                        raise RuntimeError("Mock grading first request failure")
                    if "遗漏条件" in payload["user_answer"]:
                        return schema(grade="良好",feedback="离线夹具：核心意思正确，但遗漏一个条件。",
                                      correct_points=["说明概念核心作用"],missing_points=["遗漏原文中的限定条件"],
                                      suggestion="对照参考答案补充限定条件。")
                    return schema(grade="优秀",feedback="离线夹具：表达核心含义，允许不同措辞。",correct_points=["准确说明概念"],
                                  missing_points=[],suggestion="用自己的例子再巩固一次。")
                if schema is ReportContent:
                    chunk=payload["chapter"][0]
                    quote=next(line.strip() for line in chunk["content"].splitlines() if line.strip() and not line.strip().startswith("#"))
                    evidence=[{"chunk_id":chunk["chunk_id"],"quote":quote}]
                    needs=payload["needs_review_question_ids"]
                    if any("长目录" in str(c["headings"]) for c in payload["chapter"]):
                        # 长报告只存在于离线夹具，仍经过正式报告 Schema 和证据校验。
                        return schema(core_points=[{"text":f"核心知识 {i}："+"分清输入、处理和输出。"*22,"evidence":evidence} for i in range(7)],
                            possible_confusions=[{"text":f"潜在混淆 {i}："+"比较术语的作用，不等于本次答错。"*16,"evidence":evidence} for i in range(5)],
                            memory_methods=[{"text":f"记法 {i}："+"先理解，再复述，再举例。"*18,"evidence":evidence} for i in range(4)],
                            review_suggestions=[{"focus":"加强" if q["question_id"] in needs else "巩固",
                                "question_ids":[q["question_id"]],"suggestion":"复习实际题目对应的知识。"*25} for q in payload["questions"]])
                    return schema(core_points=[{"text":"依据笔记准确区分步骤与作用。","evidence":evidence}],
                        possible_confusions=[{"text":"步骤顺序可能混淆，这是潜在易错点。","evidence":evidence}],
                        memory_methods=[{"text":"先说作用，再用一个例子解释。","evidence":evidence}],
                        review_suggestions=[{"focus":"加强" if q["question_id"] in needs else "巩固",
                            "question_ids":[q["question_id"]],"suggestion":"重读相关章节并比较容易混淆的选项。" if q["question_id"] in needs else "用自己的语言复述核心概念。"}
                            for q in payload["questions"]])
                raise AssertionError("未登记的 Fake Schema")
        return Structured()

    def invoke(self,messages):
        return AIMessage(content=messages[-1].content)

    def stream(self,messages):
        text="你好，欢迎继续复习。" if "你好" in messages[-1].content else "根据笔记，RRF 使用排名融合检索结果。[1]"
        if "BM25" in messages[-1].content:
            text="**BM25** 侧重词项匹配，向量检索寻找语义相关片段。两者从不同角度提供候选资料。[1]"
        if "代码" in messages[-1].content:
            text="**阅读代码时**，先确认输入和输出，再看处理步骤。\n\n```python\nquestion = '如何复习笔记？'\nprint(question)\n```\n\n这是离线展示示例，不调用真实模型。[1]"
        if "无引用测试" in messages[-1].content:
            text="当前资料没有支持本次问题的依据，请补充相关笔记。"
        if "仅网页测试" in messages[-1].content:
            text="网页摘要说明 RRF 使用排名融合检索结果。[W1]"
        for index in range(0,len(text),5):
            time.sleep(.18)
            yield AIMessageChunk(content=text[index:index+5])
            if "模拟失败" in messages[-1].content:
                raise RuntimeError("test streaming failure")


def make_app():
    root=Path(os.environ["V2_E2E_ROOT"]).resolve()
    root.mkdir(parents=True,exist_ok=True)
    store=Chroma(collection_name="e2e_notes",persist_directory=str(root/"chroma"),embedding_function=OfflineEmbedding())
    note=root/"联调学习笔记.md"
    if not note.exists():
        note.write_text("# 检索学习\n## RRF\nRRF 使用排名融合检索结果。\n## BM25\nBM25 根据词项匹配计算相关性。\n## 向量\n向量检索用于寻找语义相关的片段。",encoding="utf-8")
    import_markdown(note,store)
    if os.environ.get("V2_E2E_LONG_UI")=="1":
        long_note=root/"长目录联调笔记.md"
        long_note.write_text("# 长目录\n"+"".join(f"## 章节 {i:02d}\n### 知识点\n离线知识 {i}：理解输入、处理和输出。\n" for i in range(80)),encoding="utf-8")
        import_markdown(long_note,store)
    if (root/"empty-library").exists():
        # 浏览器空库验收使用另一个实际为空的临时集合，不删除原测试数据。
        store=Chroma(collection_name="e2e_empty",persist_directory=str(root/"chroma"),embedding_function=OfflineEmbedding())
    class OfflineEnvironment:
        def get(self,name,default=""):
            if name == "DASHSCOPE_API_KEY" and (root/"dashscope-env-fallback").exists():
                return "offline-environment-key"
            return default
    manager=KeyManager(WindowsCredentials(prefix=os.environ["V2_E2E_CREDENTIAL_PREFIX"]),environment=OfflineEnvironment(),
                       validator=lambda provider,key:verify_key(provider,key,transport=httpx.MockTransport(mock_probe)))
    runtime=Runtime(store_factory=lambda:store,model_factory=E2EModel,vision=FakeVision(),
                    upload_root=root/"uploads",keys=manager,settings_token=os.environ["V2_SETTINGS_TOKEN"],test_mode=True,
                    balance_reader=lambda key:fetch_balance(key,transport=httpx.MockTransport(mock_probe)))
    # 避免真实重排模型下载和任何 Tavily 调用，仅改变测试依赖。
    def offline_rerank(query,candidates,top_k=3,**kwargs):
        # 只在真实检索候选中优先展示匹配的夹具正文，不补造候选或来源。
        terms=[term for term in ("RRF","BM25") if term in query]
        ordered=sorted(candidates,key=lambda c:not any(term in c["content"] for term in terms))
        return [{**c,"rerank_score":1.0} for c in ordered[:top_k]]
    graph_module.rerank_candidates=offline_rerank
    async def offline_search(query):
        if "仅网页测试" in query:
            return [{"url":"https://example.org/reference","title":"离线网页夹具","content":"RRF 根据排名融合检索结果。"},
                    {"url":"https://example.org/unused","content":"未引用网页夹具。"}]
        return []
    graph_module.search_web=offline_search
    return create_app(runtime)
