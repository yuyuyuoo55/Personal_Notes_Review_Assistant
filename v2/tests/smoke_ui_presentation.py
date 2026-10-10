"""验证报告展示保持后端事实，不生成新错误、评分或参考答案。"""
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frontend.presentation import excerpt, mistake_view, visible_sources


class PresentationTests(unittest.TestCase):
    def test_sources_match_current_answer_and_hide_old_candidates(self):
        note={"reference":1,"id":"real","content":"正文","metadata":{"source":"real.md"},"source_type":"note","cited":True}
        unused={**note,"reference":2,"cited":False}
        original=deepcopy(note)
        self.assertEqual(visible_sources([note,unused],"真实回答。[1]"),[note])
        self.assertEqual(visible_sources([note],"日常聊天，没有引用。"),[])
        self.assertEqual(visible_sources([note],"新回答。[2]"),[])
        self.assertEqual(note,original)

    def test_web_only_valid_sources_and_malformed_sources(self):
        web={"reference":"W1","url":"https://example.org","content":"网页摘要","source_type":"web","cited":True}
        self.assertEqual(visible_sources([web],"网页结论。[W1]"),[web])
        for changed in ({"cited":"true"},{"url":"file:///secret"},{"content":""},{"source_type":"unknown"}):
            self.assertEqual(visible_sources([{**web,**changed}],"网页结论。[W1]"),[])
        self.assertEqual(visible_sources([None],"[1]"),[])

    def test_long_excerpt_keeps_original_prefix_and_full_input(self):
        text="只适用于指定场景。"*30
        self.assertEqual(excerpt(text,70),text[:70]+"…")
        self.assertEqual(excerpt("简短原文"),"简短原文")
        self.assertEqual(len(text),len("只适用于指定场景。"*30))

    def test_choice_uses_matching_actual_grade_not_other_question(self):
        mistake={"question_id":"q1","knowledge_point":"真实知识点","description":"保存的错误"}
        rows=[{"question_id":"q2","question_type":"single_choice","user_answer":"A","correct_answer":"A","explanation":"另一题解析"},
              {"question_id":"q1","question_type":"single_choice","user_answer":"B","correct_answer":"C","explanation":"这一题真实解析"}]
        result=mistake_view(mistake,rows)
        self.assertIn("B",result["reason"])
        self.assertIn("C",result["reason"])
        self.assertEqual(result["understanding"],"这一题真实解析")
        self.assertNotIn("粗心",str(result))

    def test_short_answer_uses_real_omissions_reference_and_preserves_input(self):
        mistake={"question_id":"q","knowledge_point":"术语","description":"遗漏"}
        rows=[{"question_id":"q","question_type":"short_answer","missing_points":["缺少限制条件"],
               "feedback":"真实反馈","reference_answer":"仅在该条件下成立。","grade":"良好"}]
        original=deepcopy(rows)
        result=mistake_view(mistake,rows)
        self.assertEqual(result["reason"],"缺少限制条件")
        self.assertEqual(result["understanding"],"仅在该条件下成立。")
        self.assertEqual(rows,original)
        self.assertNotIn("grade",result)

    def test_missing_grade_does_not_invent_correct_understanding(self):
        result=mistake_view({"question_id":"missing","knowledge_point":"章节","description":"原始错误说明"},[])
        self.assertEqual(result["reason"],"原始错误说明")
        self.assertEqual(result["understanding"],"")


if __name__=="__main__":
    unittest.main(verbosity=2)
