"""Phase17B 离线 ZIP 测试：Fake Vision / Mock Embedding，不访问付费接口。"""
import io
import json
import os
import stat
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from langchain_chroma import Chroma
from app.knowledge.zip_import import import_zip, inspect_zip
from app.quiz.chapter import list_notes, list_chapters, read_chapter
from app.quiz.service import generate_quiz
from app.quiz.store import QuizStore
from app.rag.vector_retrieval import retrieve_vectors
from app.rag.bm25_retrieval import retrieve_bm25
from app.rag.rrf_fusion import rrf_fusion
from smoke_quiz import FakeModel, generation, review
import re
from smoke_image_import import FakeVision, OfflineEmbedding, MemoryStore


class ZipImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="v2-zip-test-", ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.image = self.root / "diagram.png"
        Image.new("RGB", (32, 32), "white").save(self.image)
        self.store = MemoryStore()
        self.vision = FakeVision()
        self.bundle = self.root / "notes.zip"

    def make_zip(self, text='# 笔记\n## 流程\n正文\n![图](assets/a.png)', images=None):
        if images is None:
            images = {'docs/assets/a.png': self.image.read_bytes()}
        with zipfile.ZipFile(self.bundle, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('docs/notes.md', text)
            for name, content in images.items():
                archive.writestr(name, content)

    def test_zip_normal_chapter_and_provenance(self):
        self.make_zip()
        result = import_zip(self.bundle, self.store, vision=self.vision)
        self.assertEqual(result['status'], 'imported')
        self.assertEqual(result['vision_calls'], 1)
        chapter = read_chapter(self.store, result['source'], {'Header 1':'笔记','Header 2':'流程'})
        self.assertEqual(chapter.status,'ok')
        self.assertIn('倒数排名融合', chapter.content)
        metadata = next(d.metadata for d in self.store.rows.values() if d.metadata.get('is_image_chunk'))
        self.assertEqual(metadata['zip_path'],str(self.bundle))
        self.assertEqual(metadata['markdown_member'],'docs/notes.md')
        self.assertEqual(metadata['image_member'],'docs/assets/a.png')
        self.assertTrue(metadata['image_path'].startswith('zip:'))
        self.assertEqual(metadata['image_path'], 'zip:' + result['source'] + '!/docs/assets/a.png')
        self.assertEqual(metadata['source'],result['source'])

    def test_zip_multi_images_html_and_unused(self):
        self.make_zip('# 笔记\n## 一\n<img src="assets/a.png" />\n## 二\n![二](assets/b.png)\n![再次](assets/a.png)',
                      {'docs/assets/a.png':self.image.read_bytes(),'docs/assets/b.png':self.image.read_bytes(),'unused.png':self.image.read_bytes()})
        r=import_zip(self.bundle,self.store,vision=self.vision)
        self.assertEqual(r['image_references'],3); self.assertEqual(r['image_eligible'],2)
        self.assertEqual(len(self.vision.calls),2)
        for title in ['一','二']:
            c=read_chapter(self.store,r['source'],{'Header 1':'笔记','Header 2':title})
            self.assertIn('倒数排名融合',c.content)

    def test_zip_missing_and_external_partial_no_external_read(self):
        self.make_zip('# 笔记\n![外](C:/secret.png)\n![缺](absent.png)\n![越](../../escape.png)\n![图](assets/a.png)')
        r=import_zip(self.bundle,self.store,vision=self.vision)
        self.assertEqual(r['status'],'partial');self.assertEqual(r['write_status'],'imported')
        self.assertEqual(len(r['warnings']),3);self.assertEqual(r['vision_calls'],1)
        self.assertTrue(any('缺失' in w for w in r['warnings']))
        self.assertTrue(any('拦截' in w for w in r['warnings']))

    def test_zip_repeat_update_delete_and_source_isolation(self):
        self.make_zip();r=import_zip(self.bundle,self.store,vision=self.vision);old=set(self.store.rows)
        calls=len(self.vision.calls);writes=self.store.writes
        repeated=import_zip(self.bundle,self.store,vision=self.vision)
        self.assertEqual(repeated['status'],'unchanged');self.assertEqual(self.store.writes,writes)
        self.assertEqual(len(self.vision.calls),calls);self.assertEqual(set(self.store.rows),old)
        self.make_zip('# 笔记\n## 流程\n文字更新\n![图](assets/a.png)')
        self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'updated')
        self.assertEqual(len(self.vision.calls),calls)
        Image.new('RGB',(32,32),'red').save(self.image)
        self.make_zip();self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['vision_calls'],1)
        self.make_zip('# 笔记\n## 流程\n只留正文',{})
        self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'updated')
        self.assertFalse(any(d.metadata.get('is_image_chunk') for d in self.store.rows.values()))
        other=self.root/'other.zip';other.write_bytes(self.bundle.read_bytes())
        import_zip(other,self.store,vision=self.vision)
        self.assertEqual(len(list_notes(self.store)),2)

    def test_zip_failure_preserves_records(self):
        self.make_zip();import_zip(self.bundle,self.store,vision=self.vision);old=dict(self.store.rows)
        Image.new('RGB',(32,32),'red').save(self.image);self.make_zip()
        self.assertEqual(import_zip(self.bundle,self.store,vision=FakeVision(fail_at=1))['status'],'failed')
        self.assertEqual(self.store.rows,old)
        with patch.object(self.store,'add_documents',side_effect=RuntimeError('storage failed')):
            self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        self.assertEqual(self.store.rows,old)
        self.make_zip(images={'docs/assets/a.png':b'bad image'})
        before=len(self.vision.calls);self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        self.assertEqual(len(self.vision.calls),before)

    def test_zip_attack_paths_duplicates_and_special_files(self):
        for name in ['../escape.png','/absolute.png','C:/a.png','dir/CON.png','dir/a.png:stream','dir/a. /x.png','bad.exe','a/../../bad.png']:
            with self.subTest(name=name):
                self.make_zip(images={name:self.image.read_bytes()})
                self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        for names in [('a.png','A.png'),('a.png','a.png'),('a.png','a.png/x.png')]:
            with zipfile.ZipFile(self.bundle,'w') as z:
                z.writestr('notes.md','# 笔记')
                for name in names:z.writestr(name,self.image.read_bytes())
            self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        with zipfile.ZipFile(self.bundle,'w') as z:
            z.writestr('notes.md','# 笔记');i=zipfile.ZipInfo('link.png');i.create_system=3;i.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(i,'../../secret')
        self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        self.assertEqual(self.vision.calls,[])
        # 构造加密标记，检查目录信息时即拒绝，不尝试解密或调用模型。
        self.make_zip()
        raw = bytearray(self.bundle.read_bytes())
        for signature, offset in [(b'PK\x03\x04', 6), (b'PK\x01\x02', 8)]:
            start = raw.find(signature)
            raw[start + offset] |= 1
        self.bundle.write_bytes(raw)
        self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        self.assertEqual(self.vision.calls,[])

    def test_zip_limits_bomb_and_invalid_markdown(self):
        self.make_zip()
        for setting,value in [('MAX_ZIP_FILES',1),('MAX_ZIP_BYTES',1),('MAX_ZIP_TOTAL_BYTES',1),('MAX_ZIP_IMAGES',0)]:
            with patch('app.knowledge.zip_import.'+setting,value):
                self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        with zipfile.ZipFile(self.bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('notes.md','# 标题\n'+'A'*100000)
        self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        self.make_zip(text=b'\xff')
        self.assertEqual(import_zip(self.bundle,self.store,vision=self.vision)['status'],'failed')
        self.assertEqual(self.vision.calls,[])

    def test_zip_code_images_not_identified_and_relative_parent(self):
        self.make_zip('# 笔记\n```html\n<img src="bad.png"/>\n```\n`<img src="bad.png"/>`\n![图](../a.png)',{'a.png':self.image.read_bytes()})
        r=import_zip(self.bundle,self.store,vision=self.vision)
        self.assertEqual(r['status'],'imported');self.assertEqual(r['image_references'],1)

    def test_zip_chroma_reopen_and_chapter(self):
        self.make_zip();database=str(self.root/'zip-chroma')
        store=Chroma(collection_name='zip_test',persist_directory=database,embedding_function=OfflineEmbedding())
        r=import_zip(self.bundle,store,vision=self.vision)
        reopened=Chroma(collection_name='zip_test',persist_directory=database,embedding_function=None,create_collection_if_not_exists=False)
        self.assertEqual(r['status'],'imported');self.assertEqual(len(list_notes(reopened)),1)
        self.assertIn({'Header 1':'笔记','Header 2':'流程'},list_chapters(reopened,r['source']))
        headings = {'Header 1':'笔记','Header 2':'流程'}
        chapter = read_chapter(reopened,r['source'],headings)
        self.assertIn('倒数排名融合',chapter.content)
        vector = retrieve_vectors('RRF 排名融合', store, 3)
        with patch('app.rag.bm25_retrieval.get_segmenter') as segmenter:
            segmenter.return_value.cut.side_effect=lambda text: re.findall(r'RRF|排名|融合|正文',text)
            keyword = retrieve_bm25('RRF 排名融合',store,3)
        self.assertTrue(any(row['metadata'].get('is_image_chunk') for row in vector))
        self.assertTrue(any(row['metadata'].get('is_image_chunk') for row in keyword))
        self.assertTrue(rrf_fusion(vector,keyword))
        chunk = next(c for c in chapter.chunks if '倒数排名融合' in c.content)
        question={'question_type':'single_choice','question':'图中RRF是什么？','knowledge_point':'RRF',
                  'source_chunk_ids':[chunk.chunk_id],'evidence':[{'chunk_id':chunk.chunk_id,'quote':'RRF 是倒数排名融合'}],
                  'options':{'A':'倒数排名融合','B':'Git状态','C':'暂存区','D':'版本库'},'correct_answer':'A',
                  'explanation':'离线预设','reference_answer':'','key_points':[]}
        public=generate_quiz(store,r['source'],headings,1,0,model=FakeModel([generation([question]),review([True])]),quiz_store=QuizStore())
        self.assertEqual(public.status,'ready')
        self.assertNotIn('correct_answer',public.questions[0].model_dump())


if __name__=='__main__':
    os.environ.update(LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',ANONYMIZED_TELEMETRY='false')
    unittest.main(verbosity=2)
