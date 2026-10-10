"""Cloud 隔离与内部真实 HTTP/SSE 离线回归，供应商完全 Mock。"""
import asyncio
import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from uuid import uuid4
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import httpx
from fastapi.testclient import TestClient
from langchain_chroma import Chroma

from app.api.cloud import CloudSessions, CloudSessionGuard
from app.api.credentials import verify_key
from app.api.runtime import Runtime, ApiError
from app.api.main import create_app
from smoke_api import ChatModel, events
from smoke_image_import import OfflineEmbedding
from e2e_server import mock_probe
from cloud_backend import start_backend


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.now = 100.0
        self.model = ChatModel()
        def factory(keys, token):
            keys.validator = lambda p,k: verify_key(p,k,transport=httpx.MockTransport(mock_probe))
            store = Chroma(collection_name="cloud_test_"+uuid4().hex,
                persist_directory=str(Path(self.temp.name)/"chroma"), embedding_function=OfflineEmbedding())
            return Runtime(keys=keys, settings_token=token, store_factory=lambda:store,
                           model_factory=lambda:self.model)
        self.sessions = CloudSessions(runtime_factory=factory,clock=lambda:self.now)
        self.addCleanup(lambda: __import__('shutil').rmtree(self.sessions.root,ignore_errors=True))
        self.a, self.b = self.sessions.register(), self.sessions.register()
        app = create_app()
        app.add_middleware(CloudSessionGuard,sessions=self.sessions)
        self.client = TestClient(app,base_url="http://127.0.0.1",client=("127.0.0.1",4000))
        self.addCleanup(self.client.close)
        self.addCleanup(patch.stopall)
        patch('requests.sessions.Session.request',side_effect=AssertionError('paid HTTP forbidden')).start()

    def headers(self, token):
        return {"X-V2-Session-Token":token,"X-V2-Settings-Token":token}

    def test_unknown_and_missing_tokens_rejected_on_every_route(self):
        for path in ('/health','/notes','/settings/providers','/quiz/result/guess','/docs'):
            self.assertEqual(self.client.get(path).status_code,401)
            self.assertEqual(self.client.get(path,headers=self.headers('unknown')).status_code,401)

    def test_health_session_unique_and_no_model_call(self):
        a=self.client.get('/health',headers=self.headers(self.a)).json()['data']
        b=self.client.get('/health',headers=self.headers(self.b)).json()['data']
        self.assertNotEqual(a['instance_id'],b['instance_id'])
        self.assertTrue(a['cloud_demo']);self.assertTrue(a['settings_enabled'])
        self.assertEqual(self.model.stream_calls,0)

    def test_env_key_never_used_and_delete_has_no_fallback(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'sentinel-private-key','DASHSCOPE_API_KEY':'sentinel-private-embedding'}):
            for token in (self.a,self.b):
                runtime=self.sessions.rows[token]['runtime']
                for provider in ('deepseek','dashscope'):
                    self.assertEqual(runtime.keys.effective(provider),'')
                    self.assertFalse(runtime.keys.status(provider)['configured'])
            reply=self.client.put('/settings/providers/dashscope',headers=self.headers(self.a),
                json={'api_key':'offline-key','paid_validation_consent':True,'max_requests':1})
            self.assertEqual(reply.status_code,200)
            self.assertEqual(reply.json()['data']['source'],'session_memory')
            deleted=self.client.request('DELETE','/settings/providers/dashscope',headers=self.headers(self.a),json={'confirm':True})
            self.assertFalse(deleted.json()['data']['configured'])

    def test_save_credentials_only_in_owner_and_not_in_response(self):
        secret='offline-owned-key'
        reply=self.client.put('/settings/providers/deepseek',headers=self.headers(self.a),
            json={'api_key':secret,'paid_validation_consent':True,'max_requests':1})
        self.assertEqual(reply.status_code,200);self.assertNotIn(secret,reply.text)
        self.assertEqual(self.sessions.rows[self.a]['runtime'].keys.effective('deepseek'),secret)
        self.assertEqual(self.sessions.rows[self.b]['runtime'].keys.effective('deepseek'),'')

    def test_settings_cannot_use_other_session_token(self):
        headers=self.headers(self.a);headers['X-V2-Settings-Token']=self.b
        self.assertEqual(self.client.get('/settings/providers',headers=headers).status_code,403)

    def test_same_filename_upload_and_deletion_isolation(self):
        for token,text in ((self.a,'# A\n甲的笔记'),(self.b,'# B\n乙的笔记')):
            reply=self.client.post('/notes/upload',headers=self.headers(token),files={'file':('same.md',text.encode())})
            self.assertEqual(reply.status_code,200,reply.text)
        a=self.sessions.rows[self.a]['runtime'];b=self.sessions.rows[self.b]['runtime']
        self.assertNotEqual(a.upload_root,b.upload_root)
        self.assertIn('甲',a.store().get()['documents'][0]);self.assertNotIn('乙',a.store().get()['documents'][0])
        self.assertIn('乙',b.store().get()['documents'][0])
        source=self.client.get('/notes',headers=self.headers(self.a)).json()['data'][0]['source']
        reply=self.client.get('/notes/chapters',params={'source':source},headers=self.headers(self.b))
        self.assertEqual(reply.status_code,404)

    def test_cross_session_quiz_id_not_found(self):
        from smoke_quiz_grading import make_quiz
        store, public, quiz=make_quiz()
        self.sessions.rows[self.a]['runtime'].quiz_store=store
        reply=self.client.get('/quiz/result/'+public.quiz_id,headers=self.headers(self.b))
        self.assertEqual(reply.status_code,404)

    def test_tavily_explicit_key_and_no_environment_fallback(self):
        search=AsyncMock(return_value=[])
        with patch('app.api.cloud.search_web',search),patch.dict(os.environ,{'TAVILY_API_KEY':'sentinel-search-key'}):
            self.sessions.set_tavily(self.a,'owned-tavily')
            asyncio.run(self.sessions.rows[self.a]['runtime'].web_search('最新信息'))
            search.assert_awaited_with('最新信息',api_key='owned-tavily')
            asyncio.run(self.sessions.rows[self.b]['runtime'].web_search('最新信息'))
            search.assert_awaited_with('最新信息',api_key='')

    def test_empty_explicit_mcp_key_does_not_fallback(self):
        from app.integrations.tavily_mcp import get_tavily_tools
        with patch.dict(os.environ,{'TAVILY_API_KEY':'sentinel-search-key'}),patch('app.integrations.tavily_mcp.MultiServerMCPClient') as client:
            with self.assertRaises(ValueError):asyncio.run(get_tavily_tools(api_key=''))
            client.assert_not_called()

    def test_sse_done_and_error(self):
        for fail,expected in ((False,'done'),(True,'error')):
            self.model.fail=fail
            reply=self.client.post('/chat/stream',headers=self.headers(self.a),json={'question':'你好'})
            names=[name for name,data in events(reply.text)]
            self.assertIn(expected,names)
            self.assertNotIn('done' if fail else 'error',names)

    def test_expiry_clean_credentials_files_and_busy_request_protected(self):
        runtime=self.sessions.rows[self.a]['runtime'];runtime.upload_root.mkdir(parents=True)
        (runtime.upload_root/'temporary.md').write_text('public demo')
        storage=self.sessions.rows[self.a]['credentials'];storage.write('deepseek','offline-key',{})
        self.sessions.acquire(self.a)
        self.now+=1801;self.sessions.prune()
        self.assertIn(self.a,self.sessions.rows)
        self.sessions.release(self.a);self.sessions.prune()
        self.assertNotIn(self.a,self.sessions.rows);self.assertFalse(runtime.upload_root.exists());self.assertFalse(storage.rows)

    def test_capacity_and_rate_limit(self):
        self.sessions.max_sessions=2
        with self.assertRaises(ApiError):self.sessions.register()
        for _ in range(90):self.sessions.acquire(self.a);self.sessions.release(self.a)
        with self.assertRaises(ApiError):self.sessions.acquire(self.a)

    def test_oversized_upload_rejected(self):
        reply=self.client.post('/notes/upload',headers={**self.headers(self.a),'Content-Length':str(6*1024*1024)},content=b'x')
        self.assertEqual(reply.status_code,413)

    def test_internal_server_real_token_arrives_before_model_finishes(self):
        gate=Event();self.model.gate=gate
        sessions,url=start_backend(self.sessions)
        self.addCleanup(lambda:setattr(sessions.server,'should_exit',True))
        self.addCleanup(gate.set)
        with httpx.Client(base_url=url,timeout=10,trust_env=False) as client:
            with client.stream('POST','/chat/stream',headers=self.headers(self.a),json={'question':'你好'}) as reply:
                self.assertEqual(reply.status_code,200)
                found=False
                for line in reply.iter_lines():
                    if line=='event: token' and not found:
                        self.assertFalse(self.model.stream_finished)
                        found=True;gate.set()
                self.assertTrue(found)


if __name__=='__main__':unittest.main()
