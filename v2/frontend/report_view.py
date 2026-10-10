"""学习报告阅读视图；先结论后详情，全部内容来自后端。"""
import streamlit as st
from presentation import excerpt, mistake_view


def evidence(points):
    for item in points:
        st.caption("原文片段 · "+item["chunk_id"])
        st.write(item["quote"])


def render_study_report(completion,questions):
    """【业务需求】十秒内看到表现、实际问题和下一步，完整信息仍可读取。
    【输入】后端完成结果与当前会话公共题目。
    【执行步骤】顶部统计；左侧巩固与考点、右侧记忆与复习；底部同级折叠全文和逐题结果。
    【输出】只展示真实评级和批改，不推断错误、不生成新答案；不使用嵌套折叠或内部小滚动。
    """
    report=completion["report"]
    overview=report["overview"]
    results=completion["results"]
    with st.container(key="report_overview"):
        a,b,c=st.columns(3)
        a.metric("完成题数",overview["actual_question_count"])
        b.metric("选择题正确",f"{overview['single_choice_correct']}/{overview['single_choice_count']}")
        c.metric("问答题评级"," / ".join(row["grade"] for row in overview["short_answer_ratings"]) or "不适用")

    left,right=st.columns([1.35,1],gap="medium")
    with left:
        with st.container(key="report_strengthen"):
            st.subheader("本次重点巩固")
            if not report["actual_mistakes"]:
                st.write("本次没有记录到实际错误。可按右侧建议继续巩固。")
            for number,mistake in enumerate(report["actual_mistakes"][:2],1):
                view=mistake_view(mistake,results)
                st.markdown(f"**{number}. "+excerpt(view["title"],42)+"**")
                st.write("问题："+excerpt(view["reason"],58))
                st.write("正确理解："+(excerpt(view["understanding"],68) or "见逐题批改详情。"))
            if report["actual_mistakes"]:
                st.caption("以上为实际答题问题；完整说明见下方逐题详情。")
        with st.container(key="report_core"):
            st.subheader("核心考点")
            for number,point in enumerate(report["core_points"][:3],1):
                st.write(f"{number}. "+excerpt(point["text"],48))
            st.caption("完整考点与潜在易错点见下方完整报告。")
    with right:
        with st.container(key="report_memory"):
            st.subheader("记忆技巧")
            for point in report["memory_methods"][:1]:
                st.write(excerpt(point["text"],65))
        with st.container(key="report_review"):
            st.subheader("下一步复习")
            for number,suggestion in enumerate(report["review_suggestions"][:2],1):
                st.write(f"{number}. "+excerpt(suggestion["suggestion"],65))

    # 全文只折叠一次，内部普通标题与正文；不把长内容塞入小滚动框。
    with st.container(key="report_full"),st.expander("完整学习报告"):
        st.caption("笔记来源："+overview["source"])
        st.caption("范围："+("整篇笔记" if overview.get("scope")=="note" else " → ".join(overview["headings"].values()) or "无标题正文"))
        st.markdown("### 实际错误 / 遗漏")
        if not report["actual_mistakes"]:
            st.write("本次没有记录到实际错误。")
        for mistake in report["actual_mistakes"]:
            st.markdown("**"+mistake["knowledge_point"]+"**")
            st.write(mistake["description"])
            st.caption("对应题目："+mistake["question_id"])
        for label,field in (("全部核心考点","core_points"),("潜在易错点 · 不代表本次答错","possible_confusions"),("全部记忆技巧","memory_methods")):
            st.markdown("### "+label)
            for number,point in enumerate(report[field],1):
                st.markdown(f"**{number}.** "+point["text"])
                evidence(point["evidence"])
        st.markdown("### 全部复习建议")
        for number,suggestion in enumerate(report["review_suggestions"],1):
            st.write(f"{number}. "+suggestion["suggestion"])
            st.caption(suggestion["focus"]+" · 对应题目："+"、".join(suggestion["question_ids"]))
        if overview.get("coverage"):
            st.caption("材料抽样范围")
            st.json(overview["coverage"])

    # 每题都是同级折叠，不再套一个总折叠框。
    with st.container(key="report_details"):
        st.subheader("逐题批改详情")
        public={question["question_id"]:question for question in questions}
        for number,result in enumerate(results,1):
            question=public.get(result["question_id"],{})
            status=("正确" if result["correct"] else "错误") if result["question_type"]=="single_choice" else result["grade"]
            with st.expander(f"第 {number} 题 · {status} · "+excerpt(question.get("question",result["question_id"]),55)):
                st.markdown("**题干**")
                st.write(question.get("question","原题未保存在当前前端会话；题目 ID："+result["question_id"]))
                for option,text in question.get("options",{}).items():
                    st.write(option+". "+text)
                st.markdown("**你的回答**")
                st.write(result["user_answer"])
                if result["question_type"]=="single_choice":
                    st.write(status+" · 标准答案："+result["correct_answer"])
                    st.write("解析："+result["explanation"])
                else:
                    st.write("评价："+result["grade"])
                    st.write("反馈："+result["feedback"])
                    st.write("正确点："+"；".join(result["correct_points"]))
                    st.write("遗漏点："+("；".join(result["missing_points"]) or "无"))
                    st.write("参考答案："+result["reference_answer"])
                    st.write("改进建议："+result["suggestion"])
