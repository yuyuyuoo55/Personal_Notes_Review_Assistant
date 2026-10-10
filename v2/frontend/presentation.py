"""纯展示辅助：只截取已有文本、关联已保存批改，不生成评分或知识结论。"""
import re


def visible_sources(sources,answer):
    """【业务需求】旧会话候选也不能冒充本次有效引用。
    【输入】本条助手消息来源与回答正文。
    【执行步骤】核对 cited、原编号、类型及真实正文/标识；不补造缺失字段。
    【输出】可展示的实际引用；空列表不创建来源折叠框。
    """
    references=set(re.findall(r"\[(W?\d+)\]",answer))
    valid=[]
    for item in sources if isinstance(sources,list) else []:
        if not isinstance(item,dict) or item.get("cited") is not True or str(item.get("reference")) not in references:
            continue
        if not isinstance(item.get("content"),str) or not item["content"].strip():
            continue
        if item.get("source_type")=="note" and item.get("id") and isinstance(item.get("metadata"),dict):
            valid.append(item)
        elif item.get("source_type")=="web" and isinstance(item.get("url"),str) and item["url"].startswith(("https://","http://")):
            valid.append(item)
    return valid


def excerpt(text,limit=100):
    value=" ".join(str(text).split())
    if len(value)<=limit:
        return value
    return value[:limit].rstrip()+"…"


def mistake_view(mistake,results):
    """【业务需求】将真实错误与批改结果关联，提供可快速阅读的展示字段。
    【输入】ActualMistake 字典与本次已批改结果列表。
    【执行步骤】按 question_id 匹配；选择题使用真实选项/解析，问答使用遗漏/参考答案。
    【输出】标题、错误说明与正确理解参考的原始文本；不推断心理原因、不新增评分。
    """
    result=next((row for row in results if row["question_id"]==mistake["question_id"]),None)
    reason=mistake["description"]
    understanding=""
    if result and result["question_type"]=="single_choice":
        reason=f"本次选择 {result['user_answer']}；标准选项为 {result['correct_answer']}。"
        understanding=result["explanation"]
    elif result:
        reason="；".join(result["missing_points"]) or result["feedback"]
        understanding=result["reference_answer"]
    return {"title":mistake["knowledge_point"],"reason":reason,"understanding":understanding}
