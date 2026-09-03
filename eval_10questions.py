# -*- coding: utf-8 -*-
"""
10 题回归评测：快速模式（纯向量） vs 精确查找（混合检索 + 精排）

用途：为简历生成真实可写的数据（对应 HR 文档的"硬伤 3"）。
用法：
  1. 先启动后端：双击项目根目录的 启动项目.cmd（或手动启动 uvicorn）。
  2. 在项目根目录运行：python eval_10questions.py
输出：
  - 控制台打印对比表格
  - 自动生成 eval_result_YYYYMMDD_HHMMSS.md，可直接整理进简历

说明：
  - 每题自动给出"命中判断"仅供参考，请再人工看一遍回答内容确认后，
    在生成的 md 里把"判定"列改成最终 ✓ / ✗。
  - expect 取值：
      file:Git.md   -> 期望来源包含该笔记文件
      refuse        -> 期望资料不足拒答
      any           -> 期望不定，人工判断
"""

import json
import getpass
import uuid
import urllib.request
from datetime import datetime

API_URL = "http://127.0.0.1:8000/api/chat"

# 修改这里：换成你笔记库里真实存在的内容（对照 data/uploads 下的文件名）
QUESTIONS = [
    {"q": "介绍一下 Git 的常用命令", "expect": "file:Git.md", "note": "两模式都应命中"},
    {"q": "Git 的分支有什么用？", "expect": "file:Git.md", "note": "经典对比点：快速模式可能漏召回"},
    {"q": "Docker 里怎么部署 MySQL？", "expect": "file:Docker.md", "note": ""},
    {"q": "Linux 有哪些常用命令？", "expect": "file:Linux.md", "note": ""},
    {"q": "Maven 的依赖管理是怎么回事？", "expect": "file:Maven高级.md", "note": ""},
    {"q": "Vue 是什么？", "expect": "file:Vue.md", "note": "Vue.md 只有 1 个片段，召回可能弱"},
    {"q": "什么是 Docker 镜像？", "expect": "file:Docker.md", "note": ""},
    {"q": "介绍一下 MySQL", "expect": "refuse", "note": "笔记库无 MySQL 内容，应拒答"},
    {"q": "它和加权融合有什么区别？", "expect": "refuse", "note": "同时验证查询改写是否触发"},
    {"q": "什么是操作系统？", "expect": "any", "note": "边界题，看实际结果"},
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
                        f'{s.get("file_name")} · ' + "/".join(s.get("header_path") or [])
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
        headers={
            "Content-Type": "application/json",
            "X-DeepSeek-API-Key": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return parse_sse(resp)


def judge(expect, result):
    """返回 (是否通过, 说明)。expect 为 file:xxx / refuse / any。"""
    answer = result["answer"]
    if expect == "refuse":
        hit = ("资料不足" in answer) or ("没有找到" in answer) or ("没有对应片段" in answer)
        return hit, ("拒答命中" if hit else "未拒答")
    if expect.startswith("file:"):
        name = expect[len("file:"):].replace(".md", "")
        hit = any(name in s.split(" · ")[0] for s in result["sources"])
        return hit, ("来源命中" if hit else "来源未命中")
    return None, "人工判断"


def main():
    api_key = getpass.getpass("请输入您的 DeepSeek API Key（输入内容不会显示）：").strip()
    if not api_key:
        print("请先输入API Key")
        return
    print("请确认后端已启动（启动项目.cmd）。开始评测……\n")
    rows = []
    stats = {"fast": [0, 0], "acc": [0, 0]}  # [命中数, 可判断总数]

    for i, item in enumerate(QUESTIONS, 1):
        q = item["q"]
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        row = {"q": q, "expect": item["expect"], "note": item["note"],
               "fast": None, "acc": None, "fast_judge": "请求失败", "acc_judge": "请求失败"}
        for mode, key in (("fast", "fast"), ("accurate", "acc")):
            try:
                r = call_chat(q, mode, api_key)
            except Exception as e:
                print(f"  {mode} 请求失败：{e}")
                continue
            row[key] = r
            hit, detail = judge(item["expect"], r)
            row[f"{key}_judge"] = detail
            if hit is not None:
                stats[key][1] += 1
                if hit:
                    stats[key][0] += 1
                    row[f"{key}_judge"] += " ✓"
                else:
                    row[f"{key}_judge"] += " ✗"
            print(
                f"  {mode:8s} 来源 {len(r['sources'])} 个 | {r['elapsed_ms']}ms | "
                f"改写 {r['rewritten']!r} | {row[f'{key}_judge']}"
            )
            for s in r["sources"]:
                print(f"        - {s}")
        rows.append(row)
        print()

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"eval_result_{now}.md"
    lines = [
        "# 10 题回归评测结果（自动生成）",
        "",
        f"> 生成时间：{now} ｜ 运行命令：python eval_10questions.py",
        "> 说明：判定列为自动判断，请人工确认后把 ✓/✗ 作为最终结果。",
        "",
        "| # | 问题 | 期望 | 快速·来源数 | 快速·判定 | 精确·来源数 | 精确·判定 | 快速·耗时 | 精确·耗时 | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows, 1):
        fast = row["fast"] or {}
        acc = row["acc"] or {}
        lines.append(
            f"| {i} | {row['q']} | {row['expect']} | {len(fast.get('sources') or [])} | "
            f"{row['fast_judge']} | {len(acc.get('sources') or [])} | {row['acc_judge']} | "
            f"{fast.get('elapsed_ms')}ms | {acc.get('elapsed_ms')}ms | {row['note']} |"
        )
    lines.append("")
    lines.append("## 汇总（仅统计可自动判断的题）")
    lines.append(
        f"- 快速模式（Agent 按需向量检索 top-3）：{stats['fast'][0]}/{stats['fast'][1]} 命中期望"
    )
    lines.append(
        f"- 精确查找（混合检索+精排）：{stats['acc'][0]}/{stats['acc'][1]} 命中期望"
    )
    lines.append("")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已保存：{out}")
    print(f"汇总：快速 {stats['fast'][0]}/{stats['fast'][1]} ｜ 精确 {stats['acc'][0]}/{stats['acc'][1]}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError:
        print("\n连不上后端（127.0.0.1:8000）。请先运行 启动项目.cmd 再执行本脚本。")
