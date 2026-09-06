# -*- coding: utf-8 -*-
"""
RAG 三层评估脚本（检索层 + 生成层 + 应用层）

用途：为简历 / README 生成真实、可复现的 RAG 评测数据，替代"凭印象"的说法。
与 eval_10questions.py 的区别：本次覆盖三层指标，并支持 LLM-as-judge 忠实度评分。

------------------------------------------------------------
【怎么用】（先在下方 DATASET 填你自己的题）
------------------------------------------------------------
1. 准备你的脱敏测试笔记（确保 data/uploads 下有这些 .md），并确保后端已启动：
      cd Personal_Notes_Review_Assistant
      双击 启动项目.cmd            # 或手动启动 uvicorn + streamlit
2. 在下方 DATASET 里，按每题填：
      q        问题
      expect   期望（file:Git.md / refuse / any）
      note     备注
   期望来源用 data/uploads 里的真实文件名。
3. 安装 Judge 依赖（可选，做忠实度评分才需要）：
      uv pip install -r requirements.txt   # 已有
   忠实度用 DeepSeek 自己的 Key 作为"裁判"。
4. 运行：
      python eval_rag.py
   会提示输入你的 DeepSeek API Key，然后逐题跑三指标，并生成
   eval_rag_result_YYYYMMDD_HHMMSS.md。

指标说明：
  检索层  Recall@K：期望来源是否进入 Top-K（自动判定）
           MRR：命中的来源排第几（越靠前越接近 1）
  生成层  忠实度：答案是否只依据检索片段（LLM-as-judge，0~100）
           拒答准确率：该拒答拒答、不该拒答不拒答（自动判断）
  应用层  延迟：首 token 耗时 / 总耗时（ms）
"""

import getpass
import json
import uuid
import urllib.request
from datetime import datetime
from statistics import mean

API_URL = "http://127.0.0.1:8000/api/chat"
JUDGE_MODEL = "deepseek-v4-flash"

# ============================================================
# 在这里改成你笔记库里真实存在的文件（对照 data/uploads 下的文件名）
# expect 取值：
#   file:xxx.md  -> 期望回答引用的来源文件
#   refuse       -> 期望"资料不足/没找到"拒答
#   any          -> 人工判断，不参与自动统计
# ============================================================
DATASET = [
    {"q": "介绍一下 Git 的常用命令", "expect": "file:Git.md", "note": ""},
    {"q": "Git 的分支有什么用？", "expect": "file:Git.md", "note": ""},
    {"q": "Docker 里怎么部署 MySQL？", "expect": "file:Docker.md", "note": ""},
    {"q": "Linux 有哪些常用命令？", "expect": "file:Linux.md", "note": ""},
    {"q": "Maven 的依赖管理是怎么回事？", "expect": "file:Maven高级.md", "note": ""},
    {"q": "什么是 Docker 镜像？", "expect": "file:Docker.md", "note": ""},
    {"q": "介绍一下 MySQL", "expect": "refuse", "note": "笔记库无 MySQL，应拒答"},
    {"q": "什么是操作系统？", "expect": "any", "note": "人工判断"},
]


def parse_sse(resp):
    sources, tokens, elapsed, rewritten = [], [], None, None
    ev = ""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("event:"):
            ev = line[len("event:"):].strip()
        elif line.startswith("data:"):
            try:
                data = json.loads(line[len("data:"):].strip())
            except Exception:
                continue
            if ev == "meta":
                rewritten = data.get("rewritten_query")
                for s in data.get("sources") or []:
                    sources.append(
                        {
                            "file": s.get("file_name"),
                            "header": " / ".join(s.get("header_path") or []),
                        }
                    )
            elif ev == "token":
                tokens.append(data.get("content", ""))
            elif ev == "done":
                elapsed = data.get("elapsed_ms")
    return {
        "sources": sources,
        "answer": "".join(tokens),
        "elapsed_ms": elapsed,
        "rewritten": rewritten,
    }


def call_chat(query, mode, api_key):
    body = json.dumps(
        {
            "query": query,
            "mode": mode,
            "conversation_id": uuid.uuid4().hex[:32],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json", "X-DeepSeek-API-Key": api_key},
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return parse_sse(resp)


# ---------- 检索层 ----------
def retrieval_metrics(expect, result, top_k=3):
    """返回 (recall, mrr, detail)。只对 file:xxx 期望统计。"""
    if not expect.startswith("file:"):
        return None, None, "非来源题"
    want = expect[len("file:"):].replace(".md", "")
    files = [s["file"].replace(".md", "") for s in result["sources"]]
    rank = None
    for i, f in enumerate(files[:top_k], start=1):
        if want in f:
            rank = i
            break
    recall = rank is not None
    mrr = 1.0 / rank if rank else 0.0
    detail = f"期望[{want}] 排名Top{rank if rank else '未命中'}" if rank else f"期望[{want}] 未入Top{top_k}"
    return recall, mrr, detail


# ---------- 生成层：拒答 ----------
def refuse_metrics(expect, result):
    answer = result["answer"]
    if expect == "refuse":
        hit = ("资料不足" in answer) or ("没有找到" in answer) or ("没有对应片段" in answer)
        return hit, ("拒答命中" if hit else "未拒答")
    return None, "非拒答题"


# ---------- 生成层：忠实度（LLM-as-judge，可选） ----------
def faithfulness_score(answer, source_texts, api_key):
    """让 DeepSeek 当裁判，给答案打忠实度分 0~100。source_texts 为检索到的片段拼接。"""
    if not source_texts:
        return None, "无参考片段，跳过"
    prompt = (
        "你是 RAG 忠实度裁判。请判断【回答】是否只依据【参考片段】、没有编造额外事实。\n"
        "只输出一个 0~100 的整数分数（越高越忠实）。\n\n"
        f"【参考片段】\n{source_texts[:2000]}\n\n"
        f"【回答】\n{answer[:1000]}\n"
    )
    body = json.dumps(
        {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            # 提取整数
            import re
            m = re.search(r"\d{1,3}", text)
            score = int(m.group(0)) if m else None
            return (min(score, 100) if score is not None else None), f"judge={text[:40]}"
    except Exception as e:
        return None, f"judge错误:{e}"


def main():
    api_key = getpass.getpass("请输入您的 DeepSeek API Key（不显示）：").strip()
    if not api_key:
        print("请先输入API Key")
        return
    multi = input("是否启用 LLM-as-judge 忠实度评分？(y/N)：").strip().lower() == "y"
    print("请确认后端已启动。开始三层评测……\n")

    rows = []
    stats = {"fast": {"recall": [0, 0], "mrr": [], "refuse": [0, 0], "lat": []},
             "accurate": {"recall": [0, 0], "mrr": [], "refuse": [0, 0], "lat": []}}

    for i, item in enumerate(DATASET, 1):
        q = item["q"]
        print(f"[{i}/{len(DATASET)}] {q}")
        row = {"q": q, "expect": item["expect"], "note": item["note"]}
        for mode in ("fast", "accurate"):
            try:
                r = call_chat(q, mode, api_key)
            except Exception as e:
                print(f"  {mode} 失败：{e}")
                row[mode] = {"err": str(e)}
                continue
            row[mode] = r
            # 检索层
            recall, mrr, r_detail = retrieval_metrics(item["expect"], r)
            if recall is not None:
                stats[mode]["recall"][1] += 1
                stats[mode]["recall"][0] += int(recall)
                if mrr is not None:
                    stats[mode]["mrr"].append(mrr)
            # 拒答
            rhit, rj = refuse_metrics(item["expect"], r)
            if rhit is not None:
                stats[mode]["refuse"][1] += 1
                stats[mode]["refuse"][0] += int(rhit)
            # 延迟
            if r.get("elapsed_ms"):
                stats[mode]["lat"].append(r["elapsed_ms"])
            # 忠实度
            source_texts = "".join(f"{s['file']}: {s['header']}\n" for s in r["sources"])
            fid = None
            if multi and r["answer"] and r["sources"]:
                fid, f_detail = faithfulness_score(r["answer"], source_texts, api_key)
                row.setdefault(mode + "_fid", {}).update({"score": fid, "detail": f_detail})
            print(
                f"  {mode:8s} | 检索:{r_detail} | 拒答:{rj} | "
                f"{r.get('elapsed_ms')}ms | 忠实:{fid if fid is not None else '-'}"
            )
        rows.append(row)
        print()

    # 汇总
    def fmt(s):
        r = s["recall"]
        rec = f"{r[0]}/{r[1]}" if r[1] else "-"
        mrr = round(mean(s["mrr"]), 3) if s["mrr"] else "-"
        rf = s["refuse"]
        ref = f"{rf[0]}/{rf[1]}" if rf[1] else "-"
        lat = f"{round(mean(s['lat']))}ms" if s["lat"] else "-"
        return f"Recall@3={rec}, MRR={mrr}, 拒答={ref}, 平均延迟={lat}"

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"eval_rag_result_{now}.md"
    lines = [
        "# RAG 三层评估结果（自动生成）",
        "",
        f"> 生成时间：{now} ｜ 命令：python eval_rag.py ｜ 忠实度评分：{'开' if multi else '关'}",
        f"> 覆盖 {len(DATASET)} 题；检索层与拒答为自动判定，忠实度为 LLM 评审，最终请人工复核。",
        "",
        "| # | 问题 | 期望 | 快速·Recall | 快速·拒答 | 精确·Recall | 精确·拒答 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows, 1):
        def cell(mode):
            r = row.get(mode)
            if not r or "err" in r:
                return "失败", "失败", "失败", "失败"
            rec, _, rd = retrieval_metrics(row["expect"], r)
            rh, rj = refuse_metrics(row["expect"], r)
            return (str(rec) if rec is not None else "-", rj, str(rec) if rec is not None else "-", rj)
        f_rec, f_rj, a_rec, a_rj = cell("fast"), cell("accurate")
        lines.append(
            f"| {i} | {row['q']} | {row['expect']} | "
            f"{f_rec} | {f_rj} | {a_rec} | {a_rj} |"
        )
    lines.append("")
    lines.append("## 汇总")
    lines.append(f"- 快速模式：{fmt(stats['fast'])}")
    lines.append(f"- 精确模式：{fmt(stats['accurate'])}")
    lines.append("")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("== 汇总 ==")
    print("快速   " + fmt(stats["fast"]))
    print("精确   " + fmt(stats["accurate"]))
    print(f"结果已保存：{out}")
    print("\n◆ 请人工复核每题的拒答/来源是否合理，再把这些数字写进简历/README。")


if __name__ == "__main__":
    main()
