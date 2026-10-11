"""单一 DashScope、密钥生命周期及现有向量库的离线验证。"""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from app.api.credentials import KeyManager, WindowsCredentials, verify_key
from app.api.runtime import Runtime
from app.rag.vector_retrieval import create_vector_store
from smoke_frontend_settings import MemoryCredentials
from e2e_server import mock_probe


class DashScopeTests(unittest.TestCase):
    def manager(self, storage=None, environment=None):
        return KeyManager(storage or MemoryCredentials(), environment=environment or {},
            validator=lambda p, k: verify_key(p, k, transport=httpx.MockTransport(mock_probe)))

    def test_documents_query_batch_and_persistence(self):
        calls=[]
        def sdk(**kwargs):
            calls.append(kwargs)
            texts=kwargs['input'] if isinstance(kwargs['input'],list) else [kwargs['input']]
            return SimpleNamespace(status_code=200,output={'embeddings':[
                {'embedding':[1.0]+[0.0]*1023} for text in texts]})
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch('dashscope.TextEmbedding.call',side_effect=sdk):
            store=create_vector_store(directory,api_key='offline-dummy-key')
            store.add_texts(['知识'+str(i) for i in range(11)],ids=[str(i) for i in range(11)],
                            metadatas=[{'source':'offline.md','Header 1':'学习'} for i in range(11)])
            self.assertEqual([len(c['input']) for c in calls],[10,1])
            self.assertTrue(all(c['text_type']=='document' and c['model']=='text-embedding-v4' for c in calls))
            self.assertEqual(len(store._embedding_function.embed_query('知识')),1024)
            self.assertEqual(calls[-1]['text_type'],'query')
            reopened=create_vector_store(directory,api_key='offline-dummy-key')
            self.assertEqual(len(reopened.get()['ids']),11)
            self.assertEqual(len(reopened.similarity_search('知识',k=2)),2)
            self.assertEqual(reopened._collection.name,'v2_notes')
            self.assertEqual(len(reopened._client.list_collections()),1)

    def test_first_save_replace_failed_replace_delete_and_env_fallback(self):
        manager=self.manager(environment={'DASHSCOPE_API_KEY':'offline-environment-key'})
        self.assertEqual(manager.status('dashscope')['source'],'environment')
        manager.save('dashscope','offline-first-key')
        manager.save('dashscope','offline-second-key')
        with self.assertRaises(ValueError):
            manager.save('dashscope','bad-candidate-secret')
        self.assertEqual(manager.effective('dashscope'),'offline-second-key')
        status=manager.delete('dashscope')
        self.assertEqual(status['source'],'environment')
        self.assertEqual(status['validation_state'],'not_verified')
        self.assertEqual(manager.effective('dashscope'),'offline-environment-key')
        self.assertNotIn('offline-environment-key',str(status))
        manager.environment.clear()
        self.assertFalse(manager.status('dashscope')['configured'])

    def test_runtime_and_standalone_key_update_and_cache_invalidation(self):
        manager=self.manager()
        manager.save('dashscope','offline-saved-key')
        runtime=Runtime(keys=manager)
        with patch('app.api.runtime.create_vector_store',side_effect=lambda **kw: object()) as factory:
            first=runtime.store()
            self.assertEqual(factory.call_args.kwargs['api_key'],'offline-saved-key')
            self.assertIs(runtime.store(),first)
            self.assertEqual(factory.call_count,1)
            manager.save('dashscope','offline-changed-key')
            runtime.invalidate_embedding('dashscope')
            self.assertIsNot(runtime.store(),first)
            self.assertEqual(factory.call_args.kwargs['api_key'],'offline-changed-key')
            self.assertEqual(factory.call_count,2)
        # 独立入口在所有平台均通过 KeyManager 解析密钥；注入真实内存管理器。
        with patch('app.api.credentials.KeyManager',return_value=manager) as manager_factory, \
                patch('app.api.credentials.WindowsCredentials',return_value=manager.storage) as storage_factory, \
                patch('app.rag.vector_retrieval.DashScopeEmbeddings') as embedding, \
                patch('app.rag.vector_retrieval.Chroma'):
            create_vector_store()
            self.assertEqual(embedding.call_args.kwargs['dashscope_api_key'],'offline-changed-key')
            manager_factory.assert_called_once_with(manager.storage if os.name=='nt' else None)
            self.assertEqual(storage_factory.call_count,1 if os.name=='nt' else 0)

    @unittest.skipUnless(os.name=='nt','Windows only: standalone Windows credential sharing')
    def test_windows_standalone_shares_saved_credentials(self):
        manager=self.manager()
        manager.save('dashscope','offline-saved-key')
        manager.save('dashscope','offline-changed-key')
        # 保留真实 KeyManager，只替换系统存储，核对 Windows 专属共享路径。
        with patch('app.api.credentials.WindowsCredentials',return_value=manager.storage), \
                patch('app.rag.vector_retrieval.DashScopeEmbeddings') as embedding, \
                patch('app.rag.vector_retrieval.Chroma'):
            create_vector_store()
            self.assertEqual(embedding.call_args.kwargs['dashscope_api_key'],'offline-changed-key')

    def test_validation_protocol_and_invalid_responses(self):
        calls=[]
        def probe(request):
            calls.append(request)
            return mock_probe(request)
        verify_key('dashscope','offline-probe',transport=httpx.MockTransport(probe))
        self.assertEqual(len(calls),1)
        import json
        body=json.loads(calls[0].content)
        self.assertEqual(body['parameters'],{'text_type':'query','dimension':1024,'output_type':'dense'})
        for vector in [[1.0]*1023,[True]*1024,['1']*1024,[math.inf]*1024]:
            transport=httpx.MockTransport(lambda r:httpx.Response(200,json={
                'output':{'embeddings':[{'embedding':vector}]}}))
            # JSON cannot encode infinity: transport exceptions must also become safe errors.
            with self.assertRaisesRegex(ValueError,'验证失败'):
                verify_key('dashscope','secret-never-expose',transport=transport)

    def test_only_supported_provider_can_access_storage(self):
        manager=self.manager()
        with patch.object(manager.storage,'read') as read:
            with self.assertRaises(ValueError):
                manager.effective('unsupported-provider')
            read.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows only: native credential persistence')
    def test_windows_restart_persistence_dummy_credentials_only(self):
        prefix='PersonalNotesReviewAssistant/Test/'+uuid4().hex
        storage=WindowsCredentials(prefix)
        self.addCleanup(lambda:storage.delete('dashscope'))
        self.addCleanup(lambda:storage.delete('disabled/dashscope'))
        manager=self.manager(storage)
        manager.save('dashscope','offline-restart-key')
        reopened=self.manager(WindowsCredentials(prefix))
        self.assertEqual(reopened.effective('dashscope'),'offline-restart-key')
        self.assertEqual(reopened.status('dashscope')['validation_state'],'verified')
        reopened.delete('dashscope')
        self.assertFalse(self.manager(WindowsCredentials(prefix)).status('dashscope')['configured'])


if __name__=='__main__':
    unittest.main(verbosity=2)
