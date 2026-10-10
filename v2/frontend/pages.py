"""已确认的五类产品页面；只使用 HTTP 返回的公共数据。"""
import html
from datetime import datetime
from pathlib import PureWindowsPath

import streamlit as st

from client import ApiClient, BackendError
from report_view import render_study_report
from presentation import visible_sources


def title(eyebrow, heading, description):
    # 固定标题模板先转义，不把后端内容当 HTML 执行。
    st.markdown('<div class="page-heading"><h1>'+html.escape(heading)+'</h1><p>'+
                html.escape(description)+'</p></div>',unsafe_allow_html=True)


def show_error(error):
    st.error(str(error))
    if error.data:
        show_import(error.data)


def notes_data():
    try:
        return ApiClient().request("GET","/notes")["data"]
    except BackendError as error:
        show_error(error)
        return []


def heading_label(headings):
    return " → ".join(headings.values()) or "无标题正文"


def note_label(note):
    return note["filename"] + " · " + PureWindowsPath(note["source"]).parent.name


def show_sources(sources,answer):
    sources=visible_sources(sources,answer)
    if not sources:
        return
    with st.expander("查看回答来源（" + str(len(sources)) + "）"):
        for item in sources:
            # 不拼接模型返回的 HTML；URL 链接仅允许 HTTP(S)。
            metadata=item.get("metadata") if isinstance(item.get("metadata"),dict) else {}
            label = (item.get("title") or item["url"]) if item["source_type"]=="web" else (metadata.get("filename") or metadata.get("source") or item["id"])
            reference=str(item.get("reference",""))
            st.write("["+reference+"] "+label)
            st.caption("网页引用" if item["source_type"]=="web" else "笔记引用")
            url = item.get("url", "")
            if isinstance(url,str) and url.startswith(("https://","http://")):
                st.link_button("打开网页来源",url)
            headings={key:metadata[key] for key in ("Header 1","Header 2","Header 3") if key in metadata}
            if headings:
                st.caption(heading_label(headings))
            for value in (metadata.get("source") or item.get("source"), item.get("id") or item.get("chunk_id")):
                if value:
                    st.caption(str(value))
            if item.get("content"):
                content=item["content"]
                with st.container(height=220 if len(content)>700 else "content",border=False):
                    st.write(content)


def chat_page():
    """【业务需求】按连续对话展示笔记问答，只有 done 后保存回答。
    【输入】会话历史、原生聊天输入与联网开关。
    【执行步骤】历史在独立区内纵向显示；底部输入；消费真实 SSE，失败清除草稿。
    【输出】角色分明的对话、完成后的实际来源及安全失败提示。
    """
    title("","智能问答","与你的笔记对话")
    main,aside = st.columns([4.7,1],gap="medium")
    history = st.session_state.setdefault("chat_history",[])
    # 提交先进入下一轮 disabled 输入状态；导航中断也不会自动重发已取出的请求。
    pending=st.session_state.pop("chat_pending",None)
    with aside:
        with st.container(border=True,key="chat_settings"):
            st.subheader("对话设置")
            web = st.toggle("启用联网搜索",key="web_enabled",disabled=bool(pending))
            st.caption("日常问题按实时信息需求搜索，笔记问答可补充网页；联网会使用外部服务。")
            if st.button("清空当前聊天",width="stretch",disabled=bool(pending)):
                st.session_state["chat_history"] = []
                st.session_state.pop("chat_error",None)
                st.rerun()
            st.caption("聊天仅保留在本次会话。")
    with main:
        # 与 V1 一样分开历史与输入；原生 autoscroll 跟随新消息，侧栏不跟着滚动。
        messages = st.container(height=490,border=False,key="chat_history_panel",autoscroll=True)
        with messages:
            welcome=st.empty()
            if not history:
                with welcome.container(key="chat_welcome"):
                    st.subheader("今天想理解哪个知识点？")
                    st.write("提一个问题，我们一起从笔记中找答案。")
                    st.caption("你可以这样问")
                    for example in ("这两个概念有什么区别？","用一个例子解释这个知识点。","帮我梳理这一章的重点。"):
                        st.write("· "+example)
            for index,message in enumerate(history):
                with chat_bubble(message["role"],index):
                    st.markdown(message["content"])
                    if message["role"]=="assistant":
                        message["sources"]=visible_sources(message.get("sources",[]),message["content"])
                        show_sources(message["sources"],message["content"])
                        if index==len(history)-1:
                            st.caption("回答完成")
            if st.session_state.get("chat_error"):
                with chat_bubble("assistant",len(history)):
                    st.error(st.session_state["chat_error"])
                    st.caption("本轮没有保存助手回答；可调整问题后手动重试。")
        question = st.chat_input("输入你的问题，例如：这两个概念有什么区别？",max_chars=10000,disabled=bool(pending))
        if question:
            st.session_state["chat_pending"]=question
            st.session_state.pop("chat_error",None)
            st.rerun()
        if pending:
            question=pending
            welcome.empty()
            past = [{"role":m["role"],"content":m["content"]} for m in history[-20:]]
            history.append({"role":"user","content":question})
            with messages,chat_bubble("user",len(history)-1):
                st.markdown(question)
            with messages,chat_bubble("assistant",len(history)):
                state = st.empty()
                draft_area = st.empty()
                draft, sources, done = "", [], None
                try:
                    for name, data in ApiClient().stream({"question":question,"history":past,"web_enabled":web}):
                        if name=="status":
                            state.caption("处理中 · " + str(data.get("stage","")))
                            if data.get("warning"):
                                st.warning(str(data["warning"]))
                        elif name=="token":
                            text = data.get("text")
                            if not isinstance(text,str):
                                raise BackendError("回答增量无效，本轮草稿未保存。")
                            draft += text
                            draft_area.markdown(draft + " ▌")
                        elif name=="sources":
                            sources = data.get("items",[])
                        elif name=="done":
                            done = data
                    if not done:
                        raise BackendError("未收到回答完成标记。")
                    draft_area.markdown(done["answer"])
                    state.caption("回答完成")
                    sources=visible_sources(sources,done["answer"])
                    show_sources(sources,done["answer"])
                    history.append({"role":"assistant","content":done["answer"],"sources":sources})
                except BackendError as error:
                    draft_area.empty()
                    state.empty()
                    st.session_state["chat_error"]=str(error)
            st.rerun()


def chat_bubble(role,index):
    """原生聊天容器外加稳定的角色标识，仅用于布局；内容仍由 st.markdown 安全渲染。"""
    wrapper=st.container(key=f"chat_{role}_{index}",horizontal=role=="user",
                         horizontal_alignment="right" if role=="user" else "left")
    return wrapper.chat_message(role,avatar=":material/person:" if role=="user" else ":material/auto_stories:",
                                width="content" if role=="user" else "stretch")


def show_import(result):
    status = result.get("status","unknown")
    messages = {"imported":"导入成功","updated":"笔记已更新","unchanged":"内容未变化，未重复入库",
                "partial":"部分成功，请检查警告；不代表完整导入","failed":"导入失败，未确认完整写入"}
    text = messages.get(status,"导入状态：" + status)
    (st.warning if status=="partial" else st.error if status=="failed" else st.success)(text)
    metrics = st.columns(3)
    metrics[0].metric("Chunk 数量", result.get("chunk_count",0))
    metrics[1].metric("已处理图片", result.get("image_processed",0))
    metrics[2].metric("警告",len(result.get("warnings",[])))
    for warning in result.get("warnings",[]):
        st.warning(str(warning))
    if result.get("stored_path"):
        st.caption("服务端来源：" + result["stored_path"])


def knowledge_page():
    title("","知识库","添加学习资料，按章节浏览你的笔记。")
    left,right=st.columns([1,1.25],gap="medium")
    with left:
        with st.container(border=True,key="knowledge_import"):
            st.subheader("导入学习资料")
            kind = st.segmented_control("导入方式",["Markdown","单张图片","图文 ZIP"],default="Markdown",selection_mode="single",key="import_kind") or "Markdown"
            types = {"Markdown":["md","markdown"],"单张图片":["png","jpg","jpeg","webp"],"图文 ZIP":["zip"]}
            descriptions={"Markdown":"导入笔记文字。图片内容请使用图文 ZIP。",
                          "单张图片":"识别一张图片中的学习内容，保存为可检索的笔记。",
                          "图文 ZIP":"一起上传 Markdown 和图片，自动关联包内引用图片。"}
            st.write(descriptions[kind])
            # form 在同一次提交中发送文件与费用确认，避免独立 rerun 使用上一份文件。
            with st.form("upload_form_"+kind):
                uploaded = st.file_uploader("1 · 选择文件",type=types[kind],key="upload_"+kind,
                                            max_upload_size={"Markdown":2,"单张图片":10,"图文 ZIP":100}[kind])
                st.caption("选择上方文件后，再确认费用并导入。")
                st.markdown("**2 · 确认费用**")
                st.caption("仅调用文字向量化，不识别图片。" if kind=="Markdown" else
                           "调用图片识别及文字向量化。" if kind=="单张图片" else
                           "ZIP 可能按引用图片发出多个视觉请求，并调用向量化；图片越多，费用越高。")
                consent = st.checkbox("我确认本次导入可能调用模型并产生费用",key="upload_consent")
                clicked=st.form_submit_button("确认导入",type="primary",width="stretch")
            if clicked:
                if uploaded is None or not consent:
                    st.warning("请重新选择文件并确认导入费用。")
                    return
                try:
                    with st.spinner("正在导入，请勿重复提交…"):
                        result=ApiClient().request("POST","/notes/upload",files={"file":(uploaded.name,uploaded.getvalue(),uploaded.type)},
                                                  data={"markdown_images":"false"})
                    st.session_state["last_import"] = result["data"]
                    show_import(result["data"])
                except BackendError as error:
                    st.session_state["last_import"] = error.data
                    show_error(error)
            elif st.session_state.get("last_import"):
                show_import(st.session_state["last_import"])
            else:
                st.caption("尚未导入文件。选一份资料，即可开始建立自己的知识库。")
            with st.expander("文件要求与来源说明"):
                st.write("Markdown ≤2 MB · 图片 ≤10 MB · ZIP ≤100 MB。图文 ZIP 保留 Markdown 和引用图片之间的相对路径。")
                st.caption("原文件保存在本机后端，供来源溯源。外部或越界图片不会读取；警告会随结果返回。")
    with right:
        with st.container(border=True,key="knowledge_list"):
            notes = notes_data()
            st.subheader("已导入笔记 · " + str(len(notes)))
            if not notes:
                st.caption("暂无可读取笔记。请导入资料或检查后端配置。")
            else:
                mapping={note["source"]:note for note in notes}
                source=st.selectbox("选择笔记",list(mapping),format_func=lambda source:note_label(mapping[source]),key="knowledge_note")
                note=mapping[source]
                with st.expander("笔记来源"):
                    st.caption(note["source"])
                try:
                    chapters=ApiClient().request("GET","/notes/chapters",params={"source":note["source"]})["data"]
                    st.subheader("章节目录")
                    render_chapters(chapters)
                except BackendError as error:
                    show_error(error)
            if st.button("刷新笔记列表"):
                st.rerun()


def render_chapters(chapters):
    """【业务需求】长章节目录可搜索、折叠，且不会撑长页面。
    【输入】后端返回的真实标题路径字典。
    【执行步骤】按完整路径搜索；逐级分组；限高容器中显示章节树。
    【输出】只改变目录展示，不改变章节读取与小测参数。
    """
    query = st.text_input("搜索章节",placeholder="输入标题关键词",key="chapter_search").strip().casefold()
    selected = [chapter for chapter in chapters if query in heading_label(chapter).casefold()]
    st.caption(f"{len(selected)} / {len(chapters)} 个章节路径")
    tree = {}
    for chapter in selected:
        branch = tree
        labels = [chapter[key] for key in ("Header 1","Header 2","Header 3") if key in chapter] or ["无标题正文"]
        for label in labels:
            branch = branch.setdefault(label,{})
    def show(branch):
        for label,children in branch.items():
            if children:
                with st.expander(label,expanded=True):
                    show(children)
            else:
                st.write("• "+label)
    with st.container(height=min(310,max(160,len(selected)*38+60)) if selected else 96,border=False,key="chapter_tree"):
        if tree:
            show(tree)
        else:
            st.caption("没有匹配的章节。")


def reset_quiz():
    for key in ("active_quiz","completion","quiz_index","submitted_answers","report_id","quiz_locked"):
        st.session_state.pop(key,None)


def quiz_page():
    title("CHECK YOUR UNDERSTANDING","章节小测","根据真实笔记出题，检查理解，再得到有针对性的复习建议。")
    if st.session_state.get("quiz_error"):
        st.error(st.session_state.pop("quiz_error"))
    quiz=st.session_state.get("active_quiz")
    if quiz:
        return answer_page(quiz)
    with st.container(border=True,key="quiz_setup"):
        notes=notes_data()
        if not notes:
            st.info("请先在知识库导入笔记。")
            return
        mapping={note["source"]:note for note in notes}
        source=st.selectbox("选择学习笔记",list(mapping),format_func=lambda source:note_label(mapping[source]),key="quiz_note")
        note=mapping[source]
        mode=st.segmented_control("出题范围",["整篇笔记","指定章节"],default="整篇笔记",selection_mode="single",key="quiz_scope")
        headings={}
        if mode=="指定章节":
            try:
                chapters=ApiClient().request("GET","/notes/chapters",params={"source":note["source"]})["data"]
            except BackendError as error:
                show_error(error)
                return
            if not chapters:
                st.info("当前笔记没有可用章节。")
                return
            selected=st.selectbox("选择专项章节",range(len(chapters)),format_func=lambda i:heading_label(chapters[i]))
            headings=chapters[selected]
        else:
            st.caption("跨章节抽样，有限题量不保证覆盖全部知识。")
        a,b=st.columns(2)
        choice=a.number_input("选择题",min_value=0,max_value=5,value=2,step=1)
        short=b.number_input("问答题",min_value=0,max_value=3,value=1,step=1)
        st.caption("至少 1 道；依据不足时实际题量可能减少。")
        if not choice+short:
            st.warning("请至少选择一道题。")
        if st.button("生成小测",type="primary",disabled=not choice+short or mode is None,width="stretch"):
            try:
                with st.spinner("正在出题并审校…"):
                    result=ApiClient().request("POST","/quiz/generate",json={"source":note["source"],"scope":"note" if mode=="整篇笔记" else "chapter",
                        "headings":headings,"single_choice":choice,"short_answer":short})["data"]
                if result.get("questions"):
                    st.session_state["active_quiz"]=result
                    st.session_state["quiz_index"]=0
                    st.session_state["submitted_answers"]={}
                    st.rerun()
                else:
                    st.warning(result.get("message","未生成可靠题目。"))
            except BackendError as error:
                show_error(error)


def answer_page(quiz):
    questions=quiz["questions"]
    index=min(st.session_state.get("quiz_index",0),len(questions)-1)
    answers=st.session_state.setdefault("submitted_answers",{})
    submitted=st.session_state.get("quiz_locked",False)
    completion=st.session_state.get("completion")
    if completion and completion.get("status")!="completed":
        st.warning(completion.get("message","仍有题目或报告未完成。"))
    count=sum(bool((st.session_state.get("answer_"+q["question_id"],answers.get(q["question_id"],"")) or "").strip()) for q in questions)
    st.caption(("整篇笔记" if quiz.get("scope")=="note" else heading_label(quiz["headings"])) + " · 实际生成 " + str(len(questions)) + " 道")
    if quiz.get("status")=="partial":
        st.warning(quiz.get("shortage_reason") or "实际题量不足。")
    coverage=quiz.get("coverage",{})
    if coverage:
        st.caption(f"材料覆盖 {coverage.get('selected_sections',0)}/{coverage.get('total_sections',0)} 个章节分组；仅代表抽样范围。")
    st.progress(count/len(questions),text=f"已填写 {count}/{len(questions)} · 第 {index+1} 题")
    question=questions[index]
    with st.container(border=True,key="quiz_answer"):
        st.subheader(("选择题" if question["question_type"]=="single_choice" else "问答题") + f" {index+1}")
        st.write(question["question"])
        qid=question["question_id"]
        key="answer_"+qid
        if key not in st.session_state and answers.get(qid):
            st.session_state[key]=answers[qid]
        if question["question_type"]=="single_choice":
            value=st.radio("请选择一个答案",list(question["options"]),index=None,
                format_func=lambda option:option+". "+question["options"][option],key=key,disabled=submitted)
        else:
            value=st.text_area("用自己的话回答",height=150,max_chars=10000,key=key,disabled=submitted)
        answers[qid]=value or ""
        previous,next_col=st.columns(2)
        if previous.button("上一题",disabled=index==0,width="stretch"):
            st.session_state["quiz_index"]=index-1
            st.rerun()
        if next_col.button("下一题",disabled=index==len(questions)-1,width="stretch"):
            st.session_state["quiz_index"]=index+1
            st.rerun()
    completed_count=sum(bool(answers.get(q["question_id"],"").strip()) for q in questions)
    st.caption(f"当前已填写 {completed_count}/{len(questions)}。提交后不能更改原答案，失败可以重试同一份答案。")
    label="重试批改 / 报告" if submitted else "提交小测"
    if st.button(label,type="primary",disabled=completed_count!=len(questions),width="stretch"):
        st.session_state["quiz_locked"]=True
        payload={"quiz_id":quiz["quiz_id"],"answers":[{"question_id":q["question_id"],"question_type":q["question_type"],
                 "user_answer":answers[q["question_id"]]} for q in questions]}
        try:
            with st.spinner("正在批改与整理学习报告…"):
                result=ApiClient().request("POST","/quiz/submit",json=payload)["data"]
            st.session_state["completion"]=result
            if result["status"]=="completed" and result.get("report"):
                st.session_state["report_id"]=quiz["quiz_id"]
                st.switch_page(st.session_state["pages"]["report"])
            else:
                st.rerun()
        except BackendError as error:
            if error.status=="quiz_not_found":
                reset_quiz()
                st.session_state.pop("quiz_locked",None)
            st.session_state["quiz_error"]=str(error)+" 未自动重试；超时请先查询已保存结果。"
            st.rerun()
    if submitted and st.button("查询已保存结果"):
        try:
            result=ApiClient().request("GET","/quiz/result/"+quiz["quiz_id"])["data"]
            st.session_state["completion"]=result
            if result.get("report"):
                st.switch_page(st.session_state["pages"]["report"])
            else:
                st.warning(result["message"])
        except BackendError as error:
            show_error(error)
    if st.button("重新设置小测"):
        reset_quiz()
        st.session_state.pop("quiz_locked",None)
        st.rerun()


def report_page():
    completion=st.session_state.get("completion")
    if not completion or not completion.get("report"):
        title("STUDY REPORT","学习报告","完成小测并成功批改后，报告会显示在这里。")
        if st.button("前往章节小测"):
            st.switch_page(st.session_state["pages"]["quiz"])
        return
    report=completion["report"]
    overview=report["overview"]
    label="整篇笔记" if overview.get("scope")=="note" else heading_label(overview["headings"])
    title("YOUR NEXT STEP",label + " · 学习报告","根据本次真实提交和后端批改生成；潜在易错点不等于实际犯错。")
    render_study_report(completion,st.session_state.get("active_quiz",{}).get("questions",[]))
    st.caption("小测与报告保存在后端进程中，服务重启后丢失。")
    a,b=st.columns(2)
    if a.button("再做一次小测",type="primary",width="stretch"):
        reset_quiz()
        st.session_state.pop("quiz_locked",None)
        st.switch_page(st.session_state["pages"]["quiz"])
    if b.button("返回智能问答",width="stretch"):
        st.switch_page(st.session_state["pages"]["chat"])


def settings_page():
    health=st.session_state.get("health",{})
    title("","模型服务设置","本机服务已连接 · 管理访问凭据与账户余额" if health else "本机服务未连接")
    try:
        statuses=ApiClient().request("GET","/settings/providers")["data"]
    except BackendError as error:
        show_error(error)
        return
    fingerprint=tuple(statuses["deepseek"].get(field) for field in ("source","masked_key","validated_at"))
    if st.session_state.get("balance_configuration")!=fingerprint:
        st.session_state.pop("deepseek_balance",None)
        st.session_state.pop("balance_error",None)
        st.session_state["balance_configuration"]=fingerprint
    columns=st.columns(2,gap="medium")
    for column,(provider,label,use) in zip(columns,[("deepseek","DeepSeek API Key","智能问答 · 小测出题与批改 · 视觉识别"),
                                ("dashscope","阿里云百炼 API Key","笔记向量生成 · 知识库索引与检索")]):
        with column,st.container(border=True,key="service_"+provider):
            st.subheader(label)
            st.caption(use)
            current=statuses[provider]
            if current["configured"]:
                if current["validation_state"]=="verified":
                    status_text="已配置 · 服务端验证成功 · "+current["masked_key"]
                    status_class="verified"
                else:
                    status_text="已配置（.env）· 未验证 · "+current["masked_key"]
                    status_class="pending"
                st.markdown('<div class="service-state '+status_class+'" role="status">'+html.escape(status_text)+'</div>',unsafe_allow_html=True)
                a,b=st.columns(2)
                if a.button("更换 Key",key="change_"+provider,width="stretch"):
                    st.session_state["editing_"+provider]=True
                if b.button("删除配置",key="delete_"+provider,width="stretch"):
                    st.session_state["deleting_"+provider]=True
            else:
                status_text="未配置 · 已禁用 .env 回退" if current["source"]=="disabled" else "未配置"
                st.markdown('<div class="service-state missing" role="status">'+html.escape(status_text)+'</div>',unsafe_allow_html=True)
            if st.session_state.get("deleting_"+provider):
                st.warning("删除保存的 Key 后将回退 .env（若已配置）。确定删除吗？" if provider=="dashscope" else "删除后该服务将禁用，且不会自动使用 .env 的另一把 Key。确定删除吗？")
                a,b=st.columns(2)
                if a.button("确认删除",key="confirm_delete_"+provider):
                    try:
                        ApiClient().request("DELETE","/settings/providers/"+provider,json={"confirm":True})
                        st.session_state["deleting_"+provider]=False
                        st.session_state["editing_"+provider]=False
                        if provider=="deepseek":
                            st.session_state.pop("deepseek_balance",None)
                            st.session_state.pop("balance_error",None)
                        st.rerun()
                    except BackendError as error:
                        show_error(error)
                if b.button("取消",key="cancel_delete_"+provider):
                    st.session_state["deleting_"+provider]=False
                    st.rerun()
            if not current["configured"] or st.session_state.get("editing_"+provider):
                widget="key_"+provider
                if st.session_state.pop("clear_"+provider,False):
                    st.session_state[widget]=""
                with st.form("key_form_"+provider):
                    secret=st.text_input("输入 "+label,type="password",key=widget,max_chars=512)
                    st.caption("验证调用 1 次服务，可能收费；失败保留旧 Key。")
                    consent=st.checkbox("同意本次 1 次验证调用及可能费用",key="key_consent_"+provider)
                    clicked=st.form_submit_button("验证并保存",type="primary",width="stretch")
                if clicked:
                    if not secret.strip() or not consent:
                        st.warning("请填写 Key 并确认验证费用。")
                    else:
                        try:
                            with st.spinner("验证中，请勿重复提交…"):
                                ApiClient().request("PUT","/settings/providers/"+provider,json={"api_key":secret,
                                    "paid_validation_consent":True,"max_requests":1})
                            st.session_state["editing_"+provider]=False
                            st.session_state["clear_"+provider]=True
                            if provider=="deepseek":
                                st.session_state.pop("deepseek_balance",None)
                                st.session_state.pop("balance_error",None)
                            # 清空输入必须在下一轮创建 widget 之前执行。
                            st.rerun()
                        except BackendError as error:
                            show_error(error)
            if provider=="deepseek":
                render_balance(current["configured"])
            else:
                st.caption("当前 Embedding："+health.get("embedding_provider","未知"))
                st.write(health.get("embedding_model","未知"))
                st.caption("Collection："+health.get("collection","未知"))
    with st.expander("模型详情与安全说明"):
        st.write("文本模型："+health.get("text_model","未知"))
        st.write("视觉模型："+health.get("vision_model","未知"))
        st.caption("DeepSeek 文本验证不代表视觉权限已验收。Embedding 固定为 DashScope，不重建已有知识库。")
        st.caption("仅可信本机单用户。密钥保存在 Windows 系统凭据，不进入浏览器存储、Chroma、普通 JSON 或日志。")


def render_balance(configured):
    """【业务需求】醒目展示后端余额，只允许主动刷新。
    【输入】DeepSeek 是否已配置；后端余额存于当前会话。
    【执行步骤】点击才读取；失败清除旧金额；仅按真实币种展示可用余额与时间。
    【输出】未查询、成功或失败状态，不使用示例金额。
    """
    with st.container(border=False,key="deepseek_balance_card"):
        value,action=st.columns([3,1],vertical_alignment="center")
        amount=value.empty()
        if action.button("刷新余额",disabled=not configured,width="stretch"):
            try:
                with st.spinner("正在查询余额…"):
                    st.session_state["deepseek_balance"]=ApiClient().request("GET","/settings/deepseek/balance")["data"]
                st.session_state.pop("balance_error",None)
            except BackendError as error:
                st.session_state.pop("deepseek_balance",None)
                st.session_state["balance_error"]=str(error)
        data=st.session_state.get("deepseek_balance")
        if st.session_state.get("balance_error"):
            with amount.container():
                st.caption("当前可用余额 · 暂不可用")
            st.error(st.session_state["balance_error"])
        elif data:
            with amount.container():
                for info in data["balance_infos"]:
                    symbol={"CNY":"¥", "USD":"$"}[info["currency"]]
                    st.metric("当前可用余额",symbol+info["total_balance"])
            when=datetime.fromisoformat(data["queried_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            st.caption("查询于 "+when+" · "+("账户可用" if data["is_available"] else "服务返回账户不可用"))
        else:
            with amount.container():
                st.caption("当前可用余额")
                st.write("尚未查询" if configured else "配置后可查询")
        if st.session_state.get("health",{}).get("test_mode"):
            st.caption("Mock HTTP 测试金额，不代表实际账户余额。")
