"""Phase19+20 安全配置、整篇小测与前端 SSE 离线回归。"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.api.credentials import CredentialRecord, KeyManager, WindowsCredentials, verify_key
from app.api.main import create_app
from app.api.runtime import Runtime
from app.api.schemas import QuizRequest
from app.quiz.chapter import read_chapter, read_note, NOTE_MATERIAL_BUDGET
from app.quiz.schemas import GeneratedQuiz
from app.quiz.service import generate_quiz, _structured_invoke
from app.quiz.store import QuizStore
from frontend.client import ApiClient, BackendError, parse_sse
from e2e_server import E2EModel, mock_probe
from smoke_quiz import FakeStore, SOURCE, HEADINGS, FakeModel, choice, generation, review


class MemoryCredentials:
    def __init__(self):
        self.rows={}
    def read(self,name):
        return self.rows.get(name)
    def write(self,name,secret,metadata):
        self.rows[name]=CredentialRecord(secret,dict(metadata))
    def delete(self,name):
        self.rows.pop(name,None)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.storage=MemoryCredentials()
        self.keys=KeyManager(self.storage,environment={"DEEPSEEK_API_KEY":"offline-environment-key"},
            validator=lambda p,k:verify_key(p,k,transport=httpx.MockTransport(mock_probe)))
        self.runtime=Runtime(keys=self.keys,settings_token="test-token",store_factory=FakeStore,model_factory=E2EModel)
        self.client=TestClient(create_app(self.runtime),base_url="http://127.0.0.1",client=("127.0.0.1",5000))
        self.addCleanup(self.client.close)
        self.headers={"X-V2-Settings-Token":"test-token"}

    def save(self,key="offline-valid-key",provider="deepseek",consent=True):
        return self.client.put("/settings/providers/"+provider,headers=self.headers,
            json={"api_key":key,"paid_validation_consent":consent,"max_requests":1})

    def test_environment_status_is_not_verified(self):
        state=self.client.get("/settings/providers",headers=self.headers).json()["data"]["deepseek"]
        self.assertEqual(state["validation_state"],"not_verified")
        self.assertNotIn("offline-environment-key",json.dumps(state))

    def test_save_mask_priority_and_failed_replace(self):
        self.assertEqual(self.save().status_code,200)
        self.assertEqual(self.keys.effective("deepseek"),"offline-valid-key")
        failure=self.save("bad-secret-that-must-not-leak")
        self.assertEqual(failure.status_code,502)
        self.assertNotIn("bad-secret",failure.text)
        self.assertEqual(self.keys.effective("deepseek"),"offline-valid-key")

    def test_dashscope_validation_keeps_existing_collection(self):
        self.assertEqual(self.save(provider="dashscope").status_code,200)
        self.assertEqual(self.client.get("/health").json()["data"]["collection"],"v2_notes")

    def test_dashscope_delete_returns_real_environment_fallback(self):
        self.keys.environment['DASHSCOPE_API_KEY']='offline-env-fallback'
        self.save(provider='dashscope')
        reply=self.client.request('DELETE','/settings/providers/dashscope',headers=self.headers,json={'confirm':True})
        self.assertEqual(reply.status_code,200)
        state=reply.json()['data']
        self.assertEqual(state['source'],'environment')
        self.assertEqual(state['validation_state'],'not_verified')
        self.assertTrue(state['configured'])
        self.assertNotIn('offline-env-fallback',reply.text)

    def test_consent_and_request_budget(self):
        self.assertEqual(self.save(consent=False).status_code,422)
        reply=self.client.put("/settings/providers/dashscope",headers=self.headers,json={"api_key":"test","paid_validation_consent":True,"max_requests":2})
        self.assertEqual(reply.status_code,422)
        self.assertEqual(self.storage.rows,{})

    def test_delete_requires_confirmation_and_disables_environment(self):
        self.save()
        self.assertEqual(self.client.request("DELETE","/settings/providers/deepseek",headers=self.headers,json={"confirm":False}).status_code,422)
        self.assertEqual(self.client.request("DELETE","/settings/providers/deepseek",headers=self.headers,json={"confirm":True}).status_code,200)
        self.assertEqual(self.keys.effective("deepseek"),"")
        self.assertEqual(KeyManager(self.storage,environment={"DEEPSEEK_API_KEY":"other"}).effective("deepseek"),"")
        self.assertEqual(self.save().status_code,200)

    def test_local_token_host_origin_protection(self):
        self.assertEqual(self.client.get("/settings/providers").status_code,403)
        for headers in [{**self.headers,"host":"evil.example"},{**self.headers,"origin":"https://evil.example"}]:
            self.assertEqual(self.client.get("/settings/providers",headers=headers).status_code,403)
        remote=TestClient(create_app(self.runtime),base_url="http://127.0.0.1",client=("203.0.113.1",1000))
        with remote:
            self.assertEqual(remote.get("/settings/providers",headers=self.headers).status_code,403)

    def test_disabled_settings_refuse(self):
        self.runtime.settings_token=""
        self.assertEqual(self.client.get("/settings/providers",headers=self.headers).status_code,503)

    def test_busy_configuration_does_not_change_key(self):
        self.runtime.job_lock.acquire()
        try:
            self.assertEqual(self.save().status_code,409)
        finally:
            self.runtime.job_lock.release()
        self.assertEqual(self.storage.rows,{})

    def test_validation_types_dont_echo_secret(self):
        value=self.client.put("/settings/providers/deepseek",headers=self.headers,json={"api_key":"secret", "paid_validation_consent":"yes","max_requests":1})
        self.assertEqual(value.status_code,422)
        self.assertNotIn('secret',value.text)

    def test_model_factory_uses_saved_changed_and_disabled_key(self):
        runtime=Runtime(keys=self.keys,settings_token="test-token")
        with patch("app.api.runtime.create_model") as factory:
            runtime.model_factory()
            self.assertEqual(factory.call_args.kwargs["api_key"],"offline-environment-key")
            self.keys.save("deepseek","offline-new-key")
            runtime.model_factory()
            self.assertEqual(factory.call_args.kwargs["api_key"],"offline-new-key")
            self.keys.delete("deepseek")
            runtime.model_factory()
            self.assertEqual(factory.call_args.kwargs["api_key"],"")

    def test_deepseek_probe_exactly_one_http(self):
        calls=[]
        def probe(request):
            calls.append(request)
            return mock_probe(request)
        result=verify_key("deepseek","test-key",transport=httpx.MockTransport(probe))
        self.assertEqual(len(calls),1)
        self.assertEqual(result["http_requests"],1)
        self.assertNotIn("vision",result["capabilities"])

    def test_dashscope_probe_exactly_one_http(self):
        calls=[]
        def probe(request):
            calls.append(request)
            return mock_probe(request)
        result=verify_key("dashscope","test-key",transport=httpx.MockTransport(probe))
        self.assertEqual(len(calls),1)
        self.assertEqual(result["http_requests"],1)

    def test_whole_note_api_and_answer_isolation(self):
        reply=self.client.post("/quiz/generate",json={"source":SOURCE,"scope":"note","single_choice":1,"short_answer":0})
        self.assertEqual(reply.status_code,200)
        data=reply.json()["data"]
        self.assertEqual(data["scope"],"note")
        self.assertEqual(set(data["questions"][0]),{"question_id","question_type","question","options"})
        self.assertEqual(self.client.post("/quiz/generate",json={"source":SOURCE,"scope":"note","headings":HEADINGS}).status_code,422)


class ScopeTests(unittest.TestCase):
    def test_empty_headings_remains_unheaded(self):
        self.assertEqual(read_chapter(FakeStore(),SOURCE,{}).status,"chapter_not_found")
        self.assertEqual(read_note(FakeStore(),SOURCE).status,"ok")

    def test_long_note_uniform_complete_chunks_and_budget(self):
        rows=[]
        for index in range(1,81):
            rows.append((f"c-{index}",f"原文{index}。"+"知识"*250,{"source":SOURCE,"chunk_id":f"c-{index}","chunk_index":index,"Header 1":f"章节{index}"}))
        result=read_note(FakeStore(rows),SOURCE)
        self.assertEqual(result.status,"ok")
        self.assertTrue(result.coverage["sampled"])
        indices=[c.chunk_index for c in result.chunks]
        self.assertIn(1,indices)
        self.assertIn(80,indices)
        self.assertLessEqual(len(json.dumps([c.model_dump() for c in result.chunks],ensure_ascii=False)),NOTE_MATERIAL_BUDGET)
        self.assertTrue(all(c.content==rows[c.chunk_index-1][1] for c in result.chunks))

    def test_oversize_chunk_no_silent_prefix(self):
        record=FakeStore().records[0]
        self.assertEqual(read_note(FakeStore([(record[0],"文"*20000,record[2])]),SOURCE).status,"empty_chapter")

    def test_note_cross_chapter_generates_grounded_questions(self):
        record=FakeStore().records[0]
        rows=[record,("c2","BM25 根据词项匹配。",{"source":SOURCE,"chunk_id":"c2","chunk_index":2,"Header 1":"BM25"})]
        store=QuizStore()
        result=generate_quiz(FakeStore(rows),SOURCE,{},1,1,model=E2EModel(),quiz_store=store,scope="note")
        self.assertEqual(result.status,"ready")
        self.assertEqual(result.coverage["tested_sections"],2)
        self.assertEqual(store.get(result.quiz_id).chapter.scope,"note")

    def test_invalid_scope_no_model_call(self):
        model=FakeModel([])
        self.assertEqual(generate_quiz(FakeStore(),SOURCE,{},1,0,scope="all",model=model).status,"invalid_request")
        self.assertEqual(model.calls,[])

    def test_single_section_questions_return_partial_coverage(self):
        record=FakeStore().records[0]
        rows=[record,("c2","BM25 根据词项匹配。",{"source":SOURCE,"chunk_id":"c2","chunk_index":2,"Header 1":"BM25"})]
        model=FakeModel([generation([choice(1),choice(2)]),review([True,True])])
        result=generate_quiz(FakeStore(rows),SOURCE,{},2,0,model=model,quiz_store=QuizStore(),scope="note")
        self.assertEqual(result.status,"partial")
        self.assertEqual(result.actual["single_choice"],2)
        self.assertEqual(result.coverage["tested_sections"],1)
        self.assertIn("章节",result.shortage_reason)

    def test_serialized_request_guard_before_invoke(self):
        model=FakeModel([])
        with self.assertRaises(ValueError):
            _structured_invoke(model,GeneratedQuiz,[HumanMessage(content="x"*60001)])
        self.assertEqual(model.calls,[])


class ClientTests(unittest.TestCase):
    def test_sse_comments_and_frames(self):
        frames=list(parse_sse([": ping","event: token",'data: {"text":"中"}',"","event: done",'data: {"status":"completed","answer":"中文"}',""]))
        self.assertEqual(frames[0],("token",{"text":"中"}))

    def test_sse_reject_truncated_bad_json_unknown(self):
        for lines in [["event: token",'data: {"text":"a"}'],["event: token","data: not-json",""],["event: private","data: {}",""]]:
            with self.assertRaises(BackendError):
                list(parse_sse(lines))

    def test_client_reject_nonlocal_backend(self):
        with self.assertRaises(ValueError):
            ApiClient("https://example.com")

    def test_stream_requires_done_and_rejects_error(self):
        for content in ['event: token\ndata: {"text":"draft"}\n\n',
                        'event: token\ndata: {"text":"draft"}\n\nevent: error\ndata: {"message":"failed"}\n\n']:
            transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={"content-type":"text/event-stream"},text=content))
            original=httpx.Client
            with patch("frontend.client.httpx.Client",side_effect=lambda **kw:original(transport=transport,**kw)):
                with self.assertRaises(BackendError):
                    list(ApiClient().stream({}))

    @unittest.skipUnless(os.name=="nt","Windows only")
    def test_windows_credentials_reopen_and_delete_dummy_only(self):
        prefix="PersonalNotesReviewAssistant/Test/"+uuid4().hex
        store=WindowsCredentials(prefix)
        self.addCleanup(lambda:store.delete("deepseek"))
        store.write("deepseek","offline-dummy-key",{"capabilities":["mock"]})
        reopened=WindowsCredentials(prefix).read("deepseek")
        self.assertEqual(reopened.secret,"offline-dummy-key")
        self.assertEqual(reopened.metadata["capabilities"],["mock"])
        store.delete("deepseek")
        self.assertIsNone(store.read("deepseek"))


if __name__=="__main__":
    unittest.main(verbosity=2)
