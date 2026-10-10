"""Daily 联网、状态隔离、真实增量与引用离线契约；不访问供应商。"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from app.graph import build_graph, web_search_node
from app.api.main import create_app
from app.api.chat import chat_events
from app.api.runtime import Runtime
from app.api.schemas import ChatRequest
from app.integrations.tavily_mcp import search_web
from app.rag.rag_answer import valid_web_results


WEB = [{"title": "测试天气", "url": "https://example.org/weather", "content": "模拟今天广州有雨。"},
       {"url": "https://example.org/unused", "content": "未引用资料。"}]
NOTE = {"id": "rrf-1", "content": "RRF 按排名融合。", "metadata": {"source": "test/rrf.md"}}


def parse_events(body):
    result = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        result.append((lines[0][7:], json.loads(lines[1][6:])))
    return result


class FakeModel:
    """按明确输入返回意图，验证真实工作流而非调用真实分类模型。"""
    def __init__(self):
        self.answer = None
        self.fail = False
        self.gate = None
        self.finished = False
        self.prompts = []

    def with_structured_output(self, schema, method):
        class Router:
            def bind(self, **kwargs):
                return self

            def invoke(self, messages):
                text = messages[-1].content
                note = "笔记" in text or (text == "它的优点呢？" and any("笔记" in m.content for m in messages[:-1]))
                realtime = any(word in text for word in ("天气", "最新", "联网查"))
                return schema(route="knowledge" if note else "daily", requires_web=realtime)
        return Router()

    def invoke(self, messages):
        if "你负责查询改写" in messages[0].content:
            return AIMessage(content=messages[-1].content)
        return AIMessage(content="".join(chunk.content for chunk in self.stream(messages)))

    def stream(self, messages):
        self.prompts.append(messages)
        if self.fail:
            raise RuntimeError("private-test-secret")
        text = self.answer
        if text is None:
            rules = messages[0].content
            if "当前负责日常联网信息查询" in rules:
                text = "测试网页显示今天广州有雨。[W1]"
            elif "负责根据参考笔记" in rules:
                text = "RRF 按排名融合。[1]"
                if "【参考网页开始】" in messages[-1].content:
                    text += "网页补充。[W1]"
            else:
                text = "你好，稳定知识可以解释。"
        yield AIMessageChunk(content=text[:3])
        if self.gate and not self.gate.wait(3):
            raise RuntimeError("stream gate timeout")
        yield AIMessageChunk(content=text[3:])
        self.finished = True


class DailyWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = FakeModel()
        self.runtime = Runtime(model_factory=lambda: self.model,
                               store_factory=lambda: self.fail_store(),
                               upload_root=Path(self.temp.name), timeout=10)
        self.client = TestClient(create_app(self.runtime))
        self.addCleanup(self.client.close)
        self.web = self.enterContext(patch("app.graph.search_web", new_callable=AsyncMock, return_value=WEB))
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden")))
        self.enterContext(patch("httpx.AsyncClient.send", side_effect=AssertionError("network forbidden")))

    def fail_store(self):
        raise AssertionError("Daily must not open Chroma")

    def ask(self, question, enabled=False, history=None):
        parsed = parse_events(self.client.post("/chat/stream", json={"question": question,
                             "web_enabled": enabled, "history": history or []}).text)
        return parsed

    def test_greetings_emotions_and_stable_knowledge_never_force_search(self):
        for question in ("你好", "今天有点累", "RRF 是什么？"):
            for enabled in (False, True):
                with self.subTest(question=question, enabled=enabled):
                    parsed = self.ask(question, enabled)
                    self.assertEqual(parsed[-1][0], "done")
                    self.assertEqual(parsed[-1][1]["route"], "daily")
                    self.assertNotIn("笔记库资料不足", parsed[-1][1]["answer"])
        self.web.assert_not_awaited()

    def test_latest_off_has_honest_answer_without_model_or_sources(self):
        for question in ("今天广州天气？", "最新新闻是什么？"):
            parsed = self.ask(question)
            self.assertIn("未开启联网", parsed[-1][1]["answer"])
            self.assertEqual(parsed[-2], ("sources", {"items": []}))
            self.assertFalse(any(n == "token" for n, _ in parsed))
        self.assertEqual(self.model.prompts, [])
        self.web.assert_not_awaited()

    def test_latest_on_streams_and_only_returns_cited_web_source(self):
        for question in ("今天广州天气？", "最新新闻是什么？", "联网查 Python 用法"):
            parsed = self.ask(question, True)
            self.assertEqual(parsed[-1][0], "done")
            self.assertEqual("".join(d["text"] for n, d in parsed if n == "token"), parsed[-1][1]["answer"])
            sources = parsed[-2][1]["items"]
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["source_type"], "web")
            self.assertEqual(sources[0]["url"], WEB[0]["url"])
            self.assertEqual(sources[0]["reference"], "W1")
        self.assertEqual(self.web.await_count, 3)

    def test_note_rag_keeps_full_chain_and_distinct_web_sources(self):
        for enabled in (False, True):
            with patch("app.graph.retrieve_vectors", return_value=[NOTE]) as vector, \
                 patch("app.graph.retrieve_bm25", return_value=[NOTE]) as bm25, \
                 patch("app.graph.rerank_candidates", side_effect=lambda q, rows: rows) as rerank:
                parsed = self.ask("根据我的笔记解释 RRF", enabled)
            vector.assert_called_once()
            bm25.assert_called_once()
            rerank.assert_called_once()
            self.assertEqual(parsed[-1][1]["route"], "knowledge")
            types = {s["source_type"] for s in parsed[-2][1]["items"]}
            self.assertEqual(types, {"note", "web"} if enabled else {"note"})
        self.assertEqual(self.web.await_count, 1)

    def test_empty_invalid_or_failed_web_does_not_claim_live_information(self):
        for results in ([], [{"content": "missing URL"}], [{"url": "https://", "content": "invalid host"}]):
            self.web.return_value = results
            parsed = self.ask("最新新闻？", True)
            self.assertEqual(parsed[-1][0], "done")
            self.assertIn("未返回可用网页", parsed[-1][1]["answer"])
            self.assertEqual(parsed[-2][1]["items"], [])
            self.assertNotIn("笔记", parsed[-1][1]["answer"])
        self.web.side_effect = RuntimeError("private-test-secret")
        parsed = self.ask("今天广州天气？", True)
        self.assertIn("联网搜索失败", parsed[-1][1]["answer"])
        self.assertTrue(any(n == "status" and d.get("warning") for n, d in parsed))
        self.assertNotIn("private-test-secret", str(parsed))
        self.assertEqual(self.model.prompts, [])

    def test_note_search_failure_continues_with_local_sources_and_warning(self):
        self.web.side_effect = RuntimeError("private-test-secret")
        with patch("app.graph.retrieve_vectors", return_value=[NOTE]), \
             patch("app.graph.retrieve_bm25", return_value=[]), \
             patch("app.graph.rerank_candidates", side_effect=lambda q, rows: rows):
            parsed = self.ask("根据笔记解释 RRF", True)
        self.assertEqual(parsed[-1][0], "done")
        self.assertEqual({s["source_type"] for s in parsed[-2][1]["items"]}, {"note"})
        self.assertTrue(any(n == "status" and d.get("warning") for n, d in parsed))

    def test_reused_state_and_switches_never_keep_old_sources(self):
        graph = build_graph(self.model)
        state = {"messages": [], "answer_sources": [{"reference": "W9"}],
                 "reranked_results": [NOTE], "web_results": WEB, "web_error": "old"}
        for enabled, question in ((True, "今天广州天气？"), (False, "最新新闻？"), (True, "你好"),
                                  (True, "最新新闻？"), (False, "RRF 是什么？")):
            state["messages"] += [HumanMessage(content=question)]
            state["web_enabled"] = enabled
            state = asyncio.run(graph.ainvoke(state))
            self.assertEqual(state["reranked_results"], [])
            self.assertEqual(state["vector_results"], [])
            self.assertIsNone(state["web_error"])
            if not enabled or question == "你好":
                self.assertEqual(state["answer_sources"], [])
                self.assertEqual(state["web_results"], [])
        self.assertEqual(self.web.await_count, 2)

    def test_previous_citation_cannot_be_reused_in_current_daily_answer(self):
        self.model.answer = "旧结论。[W1]"
        parsed = self.ask("你好", True, [{"role": "assistant", "content": "旧网页。[W1]"}])
        self.assertEqual(parsed[-1][0], "error")
        self.assertFalse(any(n in {"done", "sources"} for n, _ in parsed))
        self.web.assert_not_awaited()

    def test_web_fabricated_note_or_web_reference_fails_without_done(self):
        for answer in ("虚构引用。[W9]", "错误笔记引用。[1]", "今天有雨但没有引用来源。", "只有代码引用：`[W1]`"):
            self.model.answer = answer
            parsed = self.ask("最新新闻？", True)
            self.assertTrue(any(n == "token" for n, _ in parsed))
            self.assertEqual(parsed[-1][0], "error")
            self.assertFalse(any(n in {"done", "sources"} for n, _ in parsed))

    def test_python_code_lists_and_indexes_are_not_source_citations(self):
        self.model.answer = "取首项用 `items[0]`。\n```python\nitems = [1]\nprint(items[0])\n```"
        parsed = self.ask("Python 如何取列表首项？")
        self.assertEqual(parsed[-1][0], "done")
        self.assertEqual(parsed[-1][1]["answer"], self.model.answer)
        self.assertEqual(parsed[-2][1]["items"], [])

    def test_web_reference_inside_code_alone_is_not_a_source(self):
        self.model.answer = "天气结论。[W1]\n```text\n[W2]\n```"
        parsed = self.ask("今天广州天气？", True)
        self.assertEqual(parsed[-1][0], "done")
        self.assertEqual([s["reference"] for s in parsed[-2][1]["items"]], ["W1"])

    def test_daily_web_model_failure_sends_safe_error(self):
        self.model.fail = True
        parsed = self.ask("最新新闻？", True)
        self.assertEqual(parsed[-1][0], "error")
        self.assertNotIn("private-test-secret", str(parsed))
        self.assertFalse(any(n in {"done", "sources"} for n, _ in parsed))

    def test_daily_web_token_arrives_before_model_finishes(self):
        self.model.gate = Event()
        class Connected:
            async def is_disconnected(self):
                return False
        async def consume():
            parsed = []
            async for block in chat_events(Connected(), self.runtime, ChatRequest(question="最新新闻？", web_enabled=True)):
                parsed.extend(parse_events(block))
                if "event: token" in block and not self.model.gate.is_set():
                    self.assertFalse(self.model.finished)
                    self.model.gate.set()
            return parsed
        self.assertEqual(asyncio.run(consume())[-1][0], "done")

    def test_personal_note_followup_keeps_rag_route(self):
        with patch("app.graph.retrieve_vectors", return_value=[NOTE]), \
             patch("app.graph.retrieve_bm25", return_value=[]), \
             patch("app.graph.rerank_candidates", side_effect=lambda q, rows: rows):
            parsed = self.ask("它的优点呢？", False, [{"role": "user", "content": "根据笔记解释 RRF"},
                                                       {"role": "assistant", "content": "RRF。[1]"}])
        self.assertEqual(parsed[-1][1]["route"], "knowledge")

    def test_search_node_itself_respects_disabled_switch(self):
        result = asyncio.run(web_search_node({"rewritten_query": "latest", "web_enabled": False}))
        self.assertEqual(result, {"web_results": [], "web_error": None})
        self.web.assert_not_awaited()


class TavilyParsingTests(unittest.TestCase):
    def test_valid_and_empty_results_and_invalid_containers(self):
        class Tool:
            name = "tavily_search"
            def __init__(self, response):
                self.response = response
            async def ainvoke(self, payload):
                return self.response
        for payload in ({"results": WEB}, {"results": []}):
            response = [{"type": "text", "text": json.dumps(payload)}]
            with patch("app.integrations.tavily_mcp.get_tavily_tools", new=AsyncMock(return_value=[Tool(response)])):
                self.assertEqual(asyncio.run(search_web("latest")), payload["results"])
        for response in ({"results": []}, ["invalid"], [{"type": "text", "text": "[]"}],
                         [{"type": "text", "text": '{"results": ["invalid"]}'}]):
            with patch("app.integrations.tavily_mcp.get_tavily_tools", new=AsyncMock(return_value=[Tool(response)])):
                with self.assertRaises(ValueError):
                    asyncio.run(search_web("latest"))

    def test_invalid_web_sources_are_not_used(self):
        rows = [None, {"url": "javascript:alert(1)", "content": "x"},
                {"url": "https://", "content": "x"}, {"url": "https://user:password@example.org", "content": "x"}]
        self.assertEqual(valid_web_results(rows), [])


if __name__ == "__main__":
    unittest.main()
