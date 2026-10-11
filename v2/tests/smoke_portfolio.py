"""作品集体验离线验证：真实临时 Chroma/图片/ZIP，禁止外部连接和供应商请求。"""
import base64
import io
import json
import socket
import sys
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1] / "frontend"))

import httpx
from fastapi.testclient import TestClient
from PIL import Image
from langchain_core.messages import AIMessage

from app.api.cloud import CloudSessions, CloudSessionGuard
from app.api.cloud_example import ASSETS, load_example, CloudEmbedding
from app.api.main import create_app
from app.api.runtime import ApiError
from app.api.source_images import public_sources
from app.quiz.chapter import read_chapter, read_note
from smoke_image_import import FakeVision
from smoke_api import events
from client import ApiClient, BackendError
import pages
from presentation import answer_markdown


ORIGINAL_CONNECT = socket.socket.connect

def deny_network(sock, address):
    if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}:
        return ORIGINAL_CONNECT(sock, address)
    raise AssertionError("offline test forbids external network")


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('requests.sessions.Session.request', side_effect=AssertionError('paid HTTP forbidden')).start()
        patch('socket.socket.connect', new=deny_network).start()
        # 真实 Cloud 默认工厂、真实资产和临时独立集合，无模型/Embedding 初始化请求。
        self.sessions = CloudSessions(sponsor_keys={})
        self.a, self.b = self.sessions.register(), self.sessions.register()
        self.ra = self.sessions.rows[self.a]['runtime']
        self.rb = self.sessions.rows[self.b]['runtime']
        self.ra.vision = FakeVision()
        self.rb.vision = FakeVision()
        app = create_app()
        app.add_middleware(CloudSessionGuard, sessions=self.sessions)
        self.client = TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 4000))
        self.addCleanup(self.client.close)
        # 清理由原有过期机制执行，只删除本测试新建的独立目录/集合。
        self.addCleanup(self.expire_sessions)
        image = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(image, format='PNG')
        self.picture = image.getvalue()

    def expire_sessions(self):
        for row in self.sessions.rows.values():
            row['seen'] = -100000
        self.sessions.prune()

    def headers(self, token):
        return {'X-V2-Session-Token': token}

    def add_image(self, token=None, zipped=False):
        runtime = self.ra if token in {None, self.a} else self.rb
        # 仅用测试替身进行上传向量化；默认样例6个真实向量不替换。
        with patch.object(CloudEmbedding, 'embed_documents', side_effect=lambda texts: [[0.1]*1024 for _ in texts]):
            if zipped:
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, 'w') as archive:
                    archive.writestr('docs/note.md', '# RRF\n![图](../images/p.png)')
                    archive.writestr('images/p.png', self.picture)
                name, content = 'bundle.zip', stream.getvalue()
            else:
                name, content = 'picture.png', self.picture
            reply = self.client.post('/notes/upload', headers=self.headers(token or self.a), files={'file': (name, content)})
            self.assertEqual(reply.status_code, 200, reply.text)
            self.assertEqual(reply.json()['data']['status'], 'imported')
        rows = runtime.store().get()
        return next((key, meta) for key, meta in zip(rows['ids'], rows['metadatas']) if meta.get('is_image_chunk'))

    def test_default_example_real_vectors_and_chapters_no_key(self):
        for token, runtime in ((self.a, self.ra), (self.b, self.rb)):
            notes = self.client.get('/notes', headers=self.headers(token)).json()['data']
            self.assertEqual(len(notes), 1)
            source = notes[0]['source']
            self.assertTrue(Path(source).is_relative_to(runtime.upload_root))
            self.assertEqual(read_note(runtime.store(), source).status, 'ok')
            headings = self.client.get('/notes/chapters', params={'source': source}, headers=self.headers(token)).json()['data']
            self.assertEqual(len(headings), 7)
            for heading in headings[1:]:
                self.assertEqual(read_chapter(runtime.store(), source, heading).status, 'ok')
            records = runtime.store().get(include=['documents', 'embeddings'])
            self.assertEqual(len(records['ids']), 6)
            package = json.loads((ASSETS/'embeddings.json').read_text(encoding='utf-8'))
            by_content = dict(zip(records['documents'], records['embeddings']))
            for chunk in package['chunks']:
                self.assertEqual(list(by_content[chunk['content']]), chunk['embedding'])
            with self.assertRaises(ApiError):
                runtime.store().similarity_search('RRF')

    def test_example_session_isolation_and_upload_keeps_example(self):
        before = self.rb.store().get()
        self.add_image()
        self.assertEqual(self.ra.store()._collection.count(), 7)
        self.assertEqual(self.rb.store().get(), before)
        self.assertTrue(set(before['ids']).isdisjoint(self.ra.store().get()['ids']))
        self.assertNotEqual(self.ra.upload_root, self.rb.upload_root)
        self.assertTrue(Path(self.ra.store()._client.get_settings().persist_directory).is_relative_to(self.sessions.root))

    def test_single_image_owned_hash_and_no_cache(self):
        key, metadata = self.add_image()
        reply = self.client.get('/notes/images/'+key, headers=self.headers(self.a))
        self.assertEqual(reply.content, self.picture)
        self.assertEqual(reply.headers['cache-control'], 'private, no-store')
        self.assertEqual(reply.headers['content-type'], 'image/png')
        Path(metadata['image_path']).write_bytes(b'changed')
        self.assertEqual(self.client.get('/notes/images/'+key, headers=self.headers(self.a)).status_code, 404)

    def test_zip_reads_preserved_original_after_temp_cleanup(self):
        key, metadata = self.add_image(zipped=True)
        self.assertTrue(metadata['image_path'].startswith('zip:'))
        self.assertEqual(metadata['image_member'], 'images/p.png')
        reply = self.client.get('/notes/images/'+key, headers=self.headers(self.a))
        self.assertEqual(reply.content, self.picture)
        with zipfile.ZipFile(metadata['zip_path'], 'a') as archive:
            archive.writestr('../escape.png', self.picture)
        self.assertEqual(self.client.get('/notes/images/'+key, headers=self.headers(self.a)).status_code, 404)

    def test_cross_session_unknown_token_and_non_image_rejected(self):
        key, _ = self.add_image()
        self.assertEqual(self.client.get('/notes/images/'+key).status_code, 401)
        self.assertEqual(self.client.get('/notes/images/'+key, headers=self.headers(self.b)).status_code, 404)
        ordinary = next(key for key, meta in zip(self.ra.store().get()['ids'], self.ra.store().get()['metadatas']) if not meta.get('is_image_chunk'))
        self.assertEqual(self.client.get('/notes/images/'+ordinary, headers=self.headers(self.a)).status_code, 404)
        self.assertEqual(self.client.get('/notes/images/'+'f'*64, headers=self.headers(self.a)).status_code, 404)

    def test_metadata_escape_to_other_session_rejected(self):
        key, metadata = self.add_image()
        other, other_meta = self.add_image(self.b)
        metadata['image_path'] = other_meta['image_path']
        self.ra.store()._collection.update(ids=[key], metadatas=[metadata])
        reply = self.client.get('/notes/images/'+key, headers=self.headers(self.a))
        self.assertEqual(reply.status_code, 404)
        self.assertNotIn(str(self.sessions.root), reply.text)

    def test_zip_missing_member_and_source_mismatch_rejected(self):
        key, metadata = self.add_image(zipped=True)
        original = deepcopy(metadata)
        for change in ({'image_member': 'missing.png'}, {'image_member': '../escape.png'}, {'zip_path': str(self.rb.upload_root/'bundle.zip')}):
            self.ra.store()._collection.update(ids=[key], metadatas=[{**original, **change}])
            self.assertEqual(self.client.get('/notes/images/'+key, headers=self.headers(self.a)).status_code, 404)

    def test_image_symlink_escape_rejected(self):
        key, metadata = self.add_image()
        other, other_meta = self.add_image(self.b)
        link = self.ra.upload_root/'link.png'
        try:
            link.symlink_to(other_meta['image_path'])
        except OSError:
            self.skipTest('Windows未授予创建符号链接权限')
        self.ra.store()._collection.update(ids=[key], metadatas=[{**metadata, 'image_path': str(link)}])
        self.assertEqual(self.client.get('/notes/images/'+key, headers=self.headers(self.a)).status_code, 404)

    def test_actual_sse_keeps_only_cited_image_and_done(self):
        key, metadata = self.add_image(zipped=True)
        ordinary = next((id, meta) for id, meta in zip(self.ra.store().get()['ids'], self.ra.store().get()['metadatas']) if not meta.get('is_image_chunk'))
        sources = [{'id': key, 'metadata': metadata, 'reference': 1, 'cited': True, 'content': '图片原文', 'source_type': 'note'},
                   {'id': ordinary[0], 'metadata': ordinary[1], 'reference': 2, 'cited': False, 'content': '未引用', 'source_type': 'note'}]
        class Graph:
            async def astream(self, state, **kwargs):
                yield 'custom', {'event': 'token', 'text': '依据图片。[1]'}
                yield 'updates', {'rag_answer': {'messages': [AIMessage(content='依据图片。[1]')], 'answer_sources': sources, 'route': 'knowledge'}}
        self.ra.model_factory = lambda: Mock()
        with patch('app.api.chat.build_graph', return_value=Graph()):
            reply = self.client.post('/chat/stream', headers=self.headers(self.a), json={'question': '图片说明了什么？'})
        frames = events(reply.text)
        items = next(data['items'] for name, data in frames if name == 'sources')
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['image_chunk_id'], key)
        self.assertNotIn(str(self.sessions.root), reply.text)
        self.assertEqual(frames[-1][0], 'done')
        self.assertEqual(frames[-1][1]['status'], 'completed')

    def test_public_sources_cited_images_only_and_no_paths(self):
        key, metadata = self.add_image(zipped=True)
        item = {'id': key, 'metadata': metadata, 'reference': 1, 'cited': True, 'content': '真实正文', 'source_type': 'note'}
        safe = public_sources([item], cloud=True)[0]
        self.assertEqual(safe['image_chunk_id'], key)
        self.assertNotIn(str(self.sessions.root), json.dumps(safe))
        self.assertNotIn('zip_path', safe['metadata'])
        self.assertNotIn('image_chunk_id', public_sources([{**item, 'cited': False}])[0])
        ordinary = {'source_type': 'note', 'metadata': {'source': '/local/plain.md'}, 'cited': True}
        self.assertEqual(public_sources([ordinary]), [ordinary])

    def test_bad_asset_refused_without_embedding_fallback(self):
        with TemporaryDirectory() as temporary:
            assets = Path(temporary)
            assets.joinpath('ai_application.md').write_bytes((ASSETS/'ai_application.md').read_bytes())
            package = json.loads((ASSETS/'embeddings.json').read_text(encoding='utf-8'))
            for change in ('model', 'dimension', 'content', 'vector'):
                bad = deepcopy(package)
                if change == 'model': bad['model'] = 'another-model'
                elif change == 'dimension': bad['dimension'] = 3
                elif change == 'content': bad['chunks'][0]['content'] += 'tampered'
                else: bad['chunks'][0]['embedding'][0] = float('nan')
                assets.joinpath('embeddings.json').write_text(json.dumps(bad), encoding='utf-8')
                before = self.ra.store().get()
                with self.assertRaises(ValueError): load_example(self.ra, assets)
                self.assertEqual(self.ra.store().get(), before)

    def test_ui_only_fetches_actual_cited_image_and_inlines_bytes(self):
        key, metadata = self.add_image()
        source = public_sources([{'id': key, 'metadata': metadata, 'reference': 1, 'cited': True, 'content': '原文', 'source_type': 'note'}], cloud=True)[0]
        fake_st = MagicMock()
        fake_st.expander.return_value.__enter__ = Mock()
        fake_st.expander.return_value.__exit__ = Mock()
        with patch.object(pages, 'st', fake_st), patch.object(pages, 'ApiClient') as api:
            api.return_value.source_image.return_value = self.picture, 'image/png'
            pages.show_sources([source], '没有引用')
            api.assert_not_called()
            pages.show_sources([source], '正文。[1]')
            api.return_value.source_image.assert_called_once_with(key)
            html = fake_st.markdown.call_args.args[0]
            self.assertIn('data:image/png;base64,'+base64.b64encode(self.picture).decode(), html)
            self.assertNotIn('/media/', html)
            fake_st.image.assert_not_called()

    def test_frontend_fixed_path_session_header_and_redirect_rejected(self):
        api = ApiClient(base_url='http://127.0.0.1')
        api.session_token = self.a
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, content=self.picture, headers={'content-type': 'image/png'})
        original = httpx.Client
        with patch('client.httpx.Client', side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)):
            self.assertEqual(api.source_image('a'*64)[0], self.picture)
            self.assertEqual(requests[0].url.path, '/notes/images/'+'a'*64)
            self.assertEqual(requests[0].headers['x-v2-session-token'], self.a)
            with self.assertRaises(BackendError): api.source_image('https://other.invalid/p.png')
        requests.clear()
        def redirect(request):
            requests.append(request)
            return httpx.Response(302, headers={'location': 'https://other.invalid/p.png'})
        with patch('client.httpx.Client', side_effect=lambda **kwargs: original(transport=httpx.MockTransport(redirect), **kwargs)):
            with self.assertRaises(BackendError): api.source_image('a'*64)
        self.assertEqual(len(requests), 1)

    def test_model_image_paths_are_text_not_rendered_images(self):
        for syntax in ('![图](file:///private.png)', '![图](https://untrusted.invalid/p.png)', '![图][ref]', '![图]'):
            self.assertTrue(answer_markdown(syntax).startswith(r'\!['))
        ordinary = 'RRF 使用排名。[1]\n\n`![代码](p.png)`\n```python\n![代码](p.png)\n```'
        self.assertEqual(answer_markdown(ordinary), ordinary)
        self.assertEqual(answer_markdown(r'\![已有转义](p.png)'), r'\![已有转义](p.png)')
        self.assertEqual(answer_markdown(r'\\![两个斜线](p.png)'), r'\\\![两个斜线](p.png)')


if __name__ == '__main__':
    unittest.main()
