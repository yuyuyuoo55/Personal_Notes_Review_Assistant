"""Phase18 HTTP/SSE离线测试；Fake模型与Mock Embedding，不调用付费API。"""
import asyncio
import io
import json
import os
import sys
import time
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch, AsyncMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.documents import Document
from app.api.main import create_app
from app.api.runtime import Runtime
from app.api.chat import chat_events
from app.api.schemas import ChatRequest
from smoke_image_import import FakeVision, OfflineEmbedding
from smoke_quiz import FakeStore, FakeModel, SOURCE, HEADINGS, choice, generation, review
from smoke_quiz_grading import make_quiz, assessment, report_response, batch_payload, single_payload


class ChatModel:
    def __init__(self,route='daily',fail=False,gate=None,wrong_citation=False):
        self.route,self.fail,self.gate,self.wrong_citation=route,fail,gate,wrong_citation
        self.stream_finished=False
        self.stream_calls=0
        self.answer=None

    def with_structured_output(self,schema,method):
        parent=self
        class Router:
            def bind(self,**kwargs):return self
            def invoke(self,messages):return schema(route=parent.route)
        return Router()

    def invoke(self,messages):
        if '你负责查询改写' not in messages[0].content:
            raise AssertionError('回答不应调用完整invoke')
        return AIMessage(content=messages[-1].content)

    def stream(self,messages):
        self.stream_calls+=1
        if self.fail:raise RuntimeError('private-api-key-must-not-leak')
        if self.answer is not None:
            yield AIMessageChunk(content=self.answer)
            self.stream_finished=True
            return
        yield AIMessageChunk(content='RRF' if self.route=='knowledge' else '你好')
        if self.gate and not self.gate.wait(3):raise RuntimeError('gate timeout')
        yield AIMessageChunk(content=' 使用排名。[999]' if self.wrong_citation else ' 使用排名。[1]' if self.route=='knowledge' else '，今天加油。')
        self.stream_finished=True


def events(body):
    parsed=[]
    for block in body.strip().split('\n\n'):
        rows=block.splitlines()
        parsed.append((rows[0].removeprefix('event: '),json.loads(rows[1].removeprefix('data: '))))
    return parsed


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(prefix='v2-api-',ignore_cleanup_errors=True);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.store=Chroma(collection_name='api_test',persist_directory=str(self.root/'chroma'),embedding_function=OfflineEmbedding())
        self.vision=FakeVision()
        self.model=ChatModel()
        self.runtime=Runtime(store_factory=lambda:self.store,model_factory=lambda:self.model,
                             vision=self.vision,upload_root=self.root/'uploads',timeout=10)
        self.client=TestClient(create_app(self.runtime));self.addCleanup(self.client.close)
        picture=io.BytesIO();Image.new('RGB',(32,32),'white').save(picture,format='PNG');self.picture=picture.getvalue()
        # 若有漏注入的网络调用，立刻失败；不使用真实供应商做API测试。
        self.addCleanup(patch.stopall)
        patch('requests.sessions.Session.request',side_effect=AssertionError('paid HTTP forbidden')).start()

    def upload(self,name,content,**data):
        return self.client.post('/notes/upload',files={'file':(name,content,'application/octet-stream')},data=data)

    def bundle(self,text,images=None):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('docs/notes.md',text)
            for name,value in (images or {}).items():z.writestr(name,value)
        return stream.getvalue()

    def test_health_swagger_and_openapi_without_model(self):
        self.runtime.model_factory=lambda:(_ for _ in ()).throw(AssertionError('不能初始化模型'))
        self.assertEqual(self.client.get('/health').status_code,200)
        self.assertEqual(self.client.get('/docs').status_code,200)
        schema=self.client.get('/openapi.json').json()
        self.assertIn('/chat/stream',schema['paths']);self.assertIn('/quiz/answer',schema['paths'])

    def test_markdown_upload_list_chapters_and_repeat(self):
        content='# 学习\n## RRF\nRRF 使用排名。'
        first=self.upload('笔记.md',content.encode()).json()['data']
        self.assertEqual(first['status'],'imported');self.assertEqual(first['original_filename'],'笔记.md')
        self.assertTrue(Path(first['stored_path']).exists())
        self.assertEqual(self.upload('笔记.md',content.encode()).json()['status'],'unchanged')
        notes=self.client.get('/notes').json()['data'];self.assertEqual(len(notes),1)
        chapters=self.client.get('/notes/chapters',params={'source':notes[0]['source']}).json()['data']
        self.assertIn({'Header 1':'学习','Header 2':'RRF'},chapters)
        self.assertEqual(self.client.get('/notes/chapters',params={'source':'C:/secret.env'}).status_code,404)

    def test_image_upload_and_invalid_image(self):
        result=self.upload('diagram.png',self.picture)
        self.assertEqual(result.json()['data']['image_processed'],1)
        self.assertEqual(result.json()['status'],'imported')
        self.assertEqual(self.upload('bad.png',b'not image').json()['status'],'failed')

    def test_plain_markdown_image_references_warn_without_vision(self):
        content=b'# Note\nText.\n![missing](no-image.png)\n<img src="C:/outside.png"/>\n'
        self.vision.calls.clear()
        for data in ({},{'markdown_images':'false'}):
            result=self.upload('plain.md',content,**data).json()['data']
            self.assertIn(result['status'],('imported','unchanged'))
            self.assertEqual(result['image_references'],2)
            self.assertEqual(result['vision_calls'],0)
            self.assertEqual(result['image_processed'],0)
            self.assertIn('图文 ZIP',result['warnings'][0])
        self.assertEqual(self.vision.calls,[])

    def test_plain_markdown_code_image_examples_do_not_warn(self):
        content=b'# Code\n`![inline](a.png)`\n```markdown\n![example](b.png)\n<img src="c.png"/>\n```\n'
        result=self.upload('code.md',content).json()['data']
        self.assertEqual(result['image_references'],0)
        self.assertEqual(result['warnings'],[])
        self.assertEqual(self.vision.calls,[])

    def test_zip_upload_partial_warning_and_stable_provenance(self):
        content=self.bundle('# 笔记\n## 图\n<img src="a.png"/>\n![外](C:/private.png)',{'docs/a.png':self.picture,'unused.png':self.picture})
        result=self.upload('notes.zip',content).json()['data']
        self.assertEqual(result['status'],'partial');self.assertEqual(result['image_processed'],1)
        self.assertTrue(result['warnings'])
        rows=self.store.get(where={'source':result['source']},include=['metadatas'])
        metadata=next(m for m in rows['metadatas'] if m.get('image_member'))
        self.assertEqual(metadata['markdown_member'],'docs/notes.md');self.assertEqual(metadata['image_member'],'docs/a.png')
        self.assertTrue(Path(metadata['zip_path']).is_file())
        repeated=self.upload('notes.zip',content).json()['data']
        self.assertEqual(repeated['write_status'],'unchanged');self.assertEqual(repeated['vision_calls'],0)

    def test_markdown_images_and_external_path_rejected(self):
        self.upload('diagram.png',self.picture)
        r=self.upload('notes.md',b'# Picture\n![img](diagram.png)',markdown_images='true').json()
        self.assertEqual(r['data']['image_processed'],1)
        calls=len(self.vision.calls)
        r=self.upload('unsafe.md',b'# Unsafe\n![img](C:/secret.png)',markdown_images='true')
        self.assertEqual(r.json()['status'],'failed');self.assertEqual(len(self.vision.calls),calls)

    def test_file_names_formats_empty_and_size_limits(self):
        for name in ['../secret.md','C:/secret.md','CON.md','x.md:stream','a\\b.md']:
            with self.subTest(name=name):self.assertEqual(self.upload(name,b'# X').status_code,422)
        self.assertEqual(self.upload('paper.pdf',b'PDF').status_code,415)
        self.assertEqual(self.upload('empty.md',b'').status_code,422)
        with patch.dict('app.api.notes.LIMITS',{'.md':8}):
            self.assertEqual(self.upload('large.md',b'# X'+b'a'*50).status_code,413)
        self.assertEqual(self.client.post('/notes/upload',headers={'Content-Length':str(200*1024*1024)},content=b'x').status_code,413)

    def test_invalid_markdown_and_zip_do_not_claim_success(self):
        self.assertEqual(self.upload('bad.md',b'\xff').json()['status'],'failed')
        self.assertEqual(self.upload('bad.zip',b'bad').json()['status'],'failed')
        self.assertEqual(self.store.get()['ids'],[])

    def test_upload_symlink_outside_root_rejected(self):
        self.runtime.upload_root.mkdir()
        target=self.root/'outside.md';target.write_text('# Original')
        try:(self.runtime.upload_root/'link.md').symlink_to(target)
        except OSError:self.skipTest('Windows符号链接权限不足')
        self.assertEqual(self.upload('link.md',b'# Changed').status_code,422)
        self.assertEqual(target.read_text(),'# Original')

    def test_daily_sse_format_tokens_and_terminal_order(self):
        r=self.client.post('/chat/stream',json={'question':'你好'})
        self.assertTrue(r.headers['content-type'].startswith('text/event-stream'))
        parsed=events(r.text);names=[n for n,_ in parsed]
        self.assertEqual(names[0],'status');self.assertEqual(names[-2:],['sources','done'])
        self.assertEqual(''.join(d['text'] for n,d in parsed if n=='token'),'你好，今天加油。')
        self.assertEqual(parsed[-1][1]['answer'],'你好，今天加油。')
        self.assertEqual(self.model.stream_calls,1)

    def test_daily_does_not_initialize_vector_store(self):
        self.runtime.store_factory=lambda:(_ for _ in ()).throw(AssertionError('Daily不能初始化向量库'))
        self.runtime._store=None
        parsed=events(self.client.post('/chat/stream',json={'question':'你好'}).text)
        self.assertEqual(parsed[-1][0],'done')

    def test_rag_sources_web_switch_and_no_router_tokens(self):
        self.model.route='knowledge'
        self.store.add_documents([Document(page_content='RRF 使用排名。',metadata={'source':'test.md','chunk_id':'note-1','Header 1':'RRF'})],ids=['note-1'])
        with patch('app.graph.retrieve_bm25',return_value=[]),patch('app.graph.rerank_candidates',side_effect=lambda q,rows:rows),patch('app.graph.search_web',new_callable=AsyncMock) as web:
            web.return_value=[{'url':'https://example.org/reference','title':'Web','content':'RRF 使用排名。'}]
            r=self.client.post('/chat/stream',json={'question':'RRF是什么','web_enabled':True})
            self.assertEqual(web.await_count,1)
        parsed=events(r.text);source=next(d['items'] for n,d in parsed if n=='sources')
        self.assertTrue(any(s['source_type']=='note' and s['cited'] for s in source))
        self.assertFalse(any(s['source_type']=='web' for s in source))
        self.assertEqual(len(source),1)
        self.assertEqual(''.join(d['text'] for n,d in parsed if n=='token'),'RRF 使用排名。[1]')

    def test_stream_error_is_safe_and_has_no_done(self):
        self.model.fail=True
        r=self.client.post('/chat/stream',json={'question':'你好'})
        self.assertEqual(events(r.text)[-1][0],'error');self.assertNotIn('event: done',r.text)
        self.assertNotIn('private-api-key',r.text)
        self.assertNotIn('event: sources',r.text)

    def test_current_no_citation_and_daily_do_not_reuse_previous_sources(self):
        self.model.route='knowledge'
        self.store.add_documents([Document(page_content='RRF 使用排名。',metadata={'source':'real.md'})],ids=['real'])
        with patch('app.graph.retrieve_bm25',return_value=[]),patch('app.graph.rerank_candidates',side_effect=lambda q,rows:rows):
            first=events(self.client.post('/chat/stream',json={'question':'RRF'}).text)
            self.assertEqual(len(next(d['items'] for n,d in first if n=='sources')),1)
            self.model.answer='资料无法支持当前问题，未引用任何资料。'
            second=events(self.client.post('/chat/stream',json={'question':'其他问题','history':[{'role':'assistant','content':'RRF 使用排名。[1]'}]}).text)
        self.assertEqual(next(d['items'] for n,d in second if n=='sources'),[])
        self.model.route='daily';self.model.answer=None
        daily=events(self.client.post('/chat/stream',json={'question':'你好'}).text)
        self.assertEqual(next(d['items'] for n,d in daily if n=='sources'),[])

    def test_web_only_cited_source_not_notes_or_unused_web(self):
        self.model.route='knowledge';self.model.answer='网页说明 RRF 使用排名。[W1]'
        with patch('app.graph.retrieve_bm25',return_value=[]),patch('app.graph.rerank_candidates',return_value=[]),patch('app.graph.search_web',new_callable=AsyncMock) as web:
            web.return_value=[{'url':'https://example.org/real','content':'排名融合'}, {'url':'https://example.org/unused','content':'无关网页'}]
            parsed=events(self.client.post('/chat/stream',json={'question':'RRF','web_enabled':True}).text)
        sources=next(d['items'] for n,d in parsed if n=='sources')
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]['source_type'],'web')
        self.assertEqual(sources[0]['url'],'https://example.org/real')

    def test_api_filters_uncited_and_stale_graph_update_sources(self):
        class Graph:
            async def astream(self,*args,**kwargs):
                yield ('updates',{'rag_answer':{'messages':[AIMessage(content='本轮回答。[2]')],
                    'answer_sources':[{'reference':1,'source_type':'note','cited':True},
                                      {'reference':2,'source_type':'note','cited':True},
                                      {'reference':3,'source_type':'note','cited':False}]}})
        with patch('app.api.chat.build_graph',return_value=Graph()):
            parsed=events(self.client.post('/chat/stream',json={'question':'测试'}).text)
        self.assertEqual(next(d['items'] for n,d in parsed if n=='sources'),[{'reference':2,'source_type':'note','cited':True}])

    def test_invalid_citations_invalidate_stream(self):
        self.model.route='knowledge';self.model.wrong_citation=True
        self.store.add_documents([Document(page_content='RRF 使用排名。',metadata={'source':'test.md'})],ids=['one'])
        with patch('app.graph.retrieve_bm25',return_value=[]),patch('app.graph.rerank_candidates',side_effect=lambda q,rows:rows):
            parsed=events(self.client.post('/chat/stream',json={'question':'RRF'}).text)
        self.assertTrue(any(n=='token' for n,_ in parsed));self.assertEqual(parsed[-1][0],'error')
        self.assertFalse(any(n=='done' for n,_ in parsed))
        self.assertFalse(any(n=='sources' for n,_ in parsed))

    def test_real_incremental_before_model_finishes(self):
        gate=Event();self.model.gate=gate
        class Connected:
            async def is_disconnected(self):return False
        async def consume():
            async for block in chat_events(Connected(),self.runtime,ChatRequest(question='你好')):
                if 'event: token' in block:
                    self.assertFalse(self.model.stream_finished)
                    gate.set();break
        asyncio.run(consume())

    def test_disconnect_stops_without_done(self):
        class Disconnected:
            async def is_disconnected(self):return True
        async def consume():
            return [block async for block in chat_events(Disconnected(),self.runtime,ChatRequest(question='你好'))]
        blocks=asyncio.run(consume());self.assertFalse(any('event: done' in b for b in blocks))

    def test_empty_knowledge_has_done_without_fake_tokens(self):
        self.model.route='knowledge'
        with patch('app.graph.retrieve_bm25',return_value=[]),patch('app.graph.rerank_candidates',return_value=[]):
            parsed=events(self.client.post('/chat/stream',json={'question':'RRF'}).text)
        self.assertEqual(parsed[-1][0],'done')
        self.assertIn('资料不足',parsed[-1][1]['answer'])
        self.assertFalse(any(name=='token' for name,_ in parsed))
        self.assertEqual(self.model.stream_calls,0)
        self.assertEqual(next(d['items'] for n,d in parsed if n=='sources'),[])
        self.assertIn('导入',parsed[-1][1]['answer'])

    def test_stream_timeout_ends_with_safe_error(self):
        class SlowGraph:
            async def astream(self,*args,**kwargs):
                await asyncio.sleep(.1)
                yield ('updates',{})
        self.runtime.timeout=.03
        with patch('app.api.chat.build_graph',return_value=SlowGraph()):
            parsed=events(self.client.post('/chat/stream',json={'question':'你好'}).text)
        self.assertEqual(parsed[-1][0],'error')
        self.assertFalse(any(name=='done' for name,_ in parsed))

    def test_persisted_upload_survives_api_restart(self):
        r=self.upload('persist.md',b'# Persist\nContent').json()['data']
        reopened=Chroma(collection_name='api_test',persist_directory=str(self.root/'chroma'),embedding_function=OfflineEmbedding(),create_collection_if_not_exists=False)
        runtime=Runtime(store_factory=lambda:reopened,upload_root=self.root/'uploads')
        with TestClient(create_app(runtime)) as other:
            notes=other.get('/notes').json()['data']
        self.assertEqual(notes[0]['source'],r['source'])
        self.assertTrue(Path(r['stored_path']).exists())

    def test_upload_timeout_preserves_worker_owned_file(self):
        from app.knowledge.note_import import import_markdown
        completed=Event()
        def delayed(path,store):
            try:
                time.sleep(.08)
                return import_markdown(path,store)
            finally:
                completed.set()
        self.runtime.timeout=.02
        with patch('app.api.notes.import_markdown',side_effect=delayed):
            result=self.upload('slow.md',b'# Slow\nContent')
            self.assertEqual(result.status_code,504)
            self.assertTrue(completed.wait(2))
        self.assertTrue((self.runtime.upload_root/'slow.md').exists())
        self.assertEqual(len(self.store.get()['ids']),1)

    def test_chat_input_validation(self):
        for payload in [{'question':' '},{'question':'x','web_enabled':'true'},{'question':'x','history':[{'role':'system','content':'leak'}]}]:
            self.assertEqual(self.client.post('/chat/stream',json=payload).status_code,422)

    def test_quiz_generate_counts_and_answer_isolation(self):
        model=FakeModel([generation([choice()]),review([True])]);self.runtime.model_factory=lambda:model
        self.runtime._store=FakeStore()
        r=self.client.post('/quiz/generate',json={'source':SOURCE,'headings':HEADINGS,'single_choice':1,'short_answer':0})
        self.assertEqual(r.json()['status'],'ready')
        question=r.json()['data']['questions'][0]
        self.assertEqual(set(question),{'question_id','question_type','question','options'})
        quiz_id=r.json()['data']['quiz_id']
        result=self.client.get('/quiz/result/'+quiz_id).json()['data']
        self.assertEqual(result['results'],[]);self.assertIsNone(result['report'])
        self.assertNotIn('correct_answer',r.text);self.assertNotIn('内部解析哨兵',r.text)
        for choice_count,short_count in [(0,0),(6,0),(0,4),(-1,1)]:
            invalid=self.client.post('/quiz/generate',json={'source':SOURCE,'single_choice':choice_count,'short_answer':short_count})
            self.assertEqual(invalid.status_code,422)

    def test_quiz_choice_duplicate_conflict_and_forgery(self):
        qs,public,quiz=make_quiz(1,0);self.runtime.quiz_store=qs
        self.runtime.model_factory=lambda:(_ for _ in ()).throw(AssertionError('选择题不能初始化模型'))
        p=single_payload(quiz,quiz.questions[0],'B')
        first=self.client.post('/quiz/answer',json=p)
        self.assertFalse(first.json()['data']['result']['correct'])
        self.assertEqual(first.json()['data']['result']['correct_answer'],'A')
        self.assertEqual(self.client.post('/quiz/answer',json=p).json()['status'],'already_graded')
        self.assertEqual(self.client.post('/quiz/answer',json={**p,'user_answer':'A'}).status_code,409)
        self.assertEqual(self.client.post('/quiz/answer',json={**p,'correct_answer':'A'}).status_code,422)
        self.assertEqual(self.client.get('/quiz/result/absent').status_code,404)
        self.assertEqual(self.client.post('/quiz/answer',json={**p,'question_id':'absent'}).status_code,404)

    def test_quiz_full_report_and_pure_result_read(self):
        qs,public,quiz=make_quiz();self.runtime.quiz_store=qs
        model=FakeModel([assessment(),report_response(quiz,[quiz.questions[0].question_id])]);self.runtime.model_factory=lambda:model
        p=batch_payload(quiz,['B','利用名次融合，不用统一分数尺度。'])
        result=self.client.post('/quiz/submit',json=p)
        self.assertEqual(result.json()['status'],'completed')
        self.assertEqual(result.json()['data']['report']['overview']['single_choice_correct'],0)
        self.assertEqual(result.json()['data']['results'][1]['grade'],'优秀')
        count=len(model.calls);self.runtime.model_factory=lambda:(_ for _ in ()).throw(AssertionError('读取不能调用模型'))
        self.assertEqual(self.client.get('/quiz/result/'+quiz.quiz_id).json()['status'],'completed')
        self.assertEqual(len(model.calls),count)
        self.assertEqual(self.client.post('/quiz/submit',json=p).json()['status'],'completed')

    def test_single_short_answer_grade_failure_and_retry(self):
        qs,public,quiz=make_quiz(0,1);self.runtime.quiz_store=qs
        model=FakeModel([(assessment()[0],RuntimeError('private secret')),assessment()])
        self.runtime.model_factory=lambda:model
        payload=single_payload(quiz,quiz.questions[0],'使用名次，不用统一尺度。')
        failed=self.client.post('/quiz/answer',json=payload)
        self.assertEqual(failed.status_code,502);self.assertNotIn('private secret',failed.text)
        success=self.client.post('/quiz/answer',json=payload)
        self.assertEqual(success.json()['data']['result']['grade'],'优秀')
        self.assertEqual(self.client.post('/quiz/answer',json=payload).json()['status'],'already_graded')
        self.assertEqual(len(model.calls),2)

    def test_quiz_model_failure_and_partial_submit(self):
        qs,public,quiz=make_quiz();self.runtime.quiz_store=qs
        model=FakeModel([(assessment()[0],RuntimeError('secret'))]);self.runtime.model_factory=lambda:model
        r=self.client.post('/quiz/submit',json=batch_payload(quiz,['A','回答']))
        self.assertEqual(r.json()['status'],'partial_failure')
        self.assertEqual(len(r.json()['data']['results']),1);self.assertIsNone(r.json()['data']['report'])
        self.assertNotIn('secret',r.text)

    def test_job_timeout_busy_and_safe_error(self):
        from app.api.runtime import run_job, ApiError
        async def check():
            self.runtime.timeout=.01
            with self.assertRaises(ApiError) as failure:
                await run_job(self.runtime,lambda:time.sleep(.08))
            self.assertEqual(failure.exception.code,'timeout')
            with self.assertRaises(ApiError) as busy:
                await run_job(self.runtime,lambda:1)
            self.assertEqual(busy.exception.code,'busy')
            await asyncio.sleep(.1)
        asyncio.run(check())
        self.runtime.timeout=10
        self.runtime.store_factory=lambda:(_ for _ in ()).throw(RuntimeError('secret api key'))
        self.runtime._store=None
        r=self.client.get('/notes');self.assertEqual(r.status_code,500);self.assertNotIn('secret',r.text)


if __name__=='__main__':
    os.environ.update(LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',ANONYMIZED_TELEMETRY='false')
    unittest.main(verbosity=2)
