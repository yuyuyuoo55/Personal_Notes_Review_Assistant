# Phase 16：章节小测、批改与学习报告

16A 已实现章节读取与出题；16B 已实现答案提交、批改和学习报告。均为独立后端函数，尚未接入 FastAPI 或页面，不修改 Graph 主流程。

## 调用流程

1. `list_notes(vector_store)` 返回实际已导入的 `source` 和文件名。
2. `list_chapters(vector_store, source)` 返回完整标题字典。
3. `read_chapter(vector_store, source, headings)` 返回章节及按 `chunk_index` 排序的 Chunk。
4. `generate_quiz(vector_store, source, headings, single_choice, short_answer)` 返回 `QuizResult`。

例如标题选择为 `{"Header 1": "检索", "Header 2": "RRF"}`：匹配这两个父级标题，包含其三级子章节，排除相邻的 BM25 章节。`{}` 只选择无标题正文，不代表整篇笔记。跳级标题保留原始 `Header` 层级，不把三级标题冒充一级标题。

所有读取函数要求传入已有 V2 Chroma 对象，使用 `get(where={"source": source})`，不执行相似度检索，不创建另一套默认库。source 与 Markdown 导入采用相同的绝对路径及 Windows 大小写规范。

如果缺少 chunk_index/chunk_id、顺序不可靠或发现重复完整标题路径，返回明确问题。现有导入 metadata 没有独立章节实例 ID；目前通过非连续的同路径 Chunk 和保留在正文中的重复标题检测歧义，无法证明所有异常 Markdown 的章节身份。此阶段保留原导入逻辑；遇到不能可靠识别的笔记，应先核对原文及 metadata。

## 出题与错误结果

- 请求选择题 0～5、问答题 0～3，道数必须为整数且总数至少 1。
- Pydantic Structured Output 检查结构和题型；Python 核对真实 Chunk ID、逐字引用、基础选项、去重及题量。
- 第二次结构化模型调用审校原文支持、单选唯一答案、同义重复和展示题目泄露答案。模型复核仍可能判断错误，属于自动质量检查，不是事实正确性的数学保证。
- 不按字数强行估算题量。资料不足允许少出题；请求/实际数量分别返回。
- `ready`：达到请求题量；`partial`：保留已审校成功题目，并说明不足。
- `insufficient_content`：模型未找到可出题知识点；`chapter_unavailable`：无笔记/章节、空正文或 metadata 不可靠。
- `invalid_request`、`invalid_model_output`、`model_error` 分别表示请求错误、有限尝试仍未通过校验、模型调用失败。
- 最多两轮出题，每轮最多一次语义审校，总共最多四次模型调用。资料不足直接停止补题；调用失败也不无限重试。
- 单轮结构化输出无法解析时该轮不能当作可靠题目使用；前一轮已成功的题目仍可保留。不会向用户输出供应商原始异常或模型原始内部理由。

## 答案隔离

`QuizStore` 以随机小测 ID 保存完整题目、答案和章节副本。批改后端通过 `store.get(quiz_id)` 找到内部数据，此对象绝不能直接返回前端。

公共题目只包含题目 ID、题型、题干和选择题选项。结果另外提供小测 ID、来源/章节、请求/实际题量和安全的状态说明。解析、答案、得分点、知识点标签、依据原文均不提前返回。

这是进程内状态：进程重启后丢失，多个工作进程也不共享。随机 ID 降低猜测风险，但尚未实现用户身份和权限系统；未来接入接口时须明确访问范围。

## 离线验证

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_quiz.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_quiz_grading.py
```

使用 Fake Model 验证结构化流程，用现有 Markdown 导入和临时真实 Chroma 验证 metadata/章节选择；Embedding 为离线替身。不会调用 DeepSeek/DashScope/Tavily，也不会打开默认持久化库。

真实模型初始化复用 `app.model.create_model()`。真实出题可能消耗最多四次 DeepSeek 请求，应先说明测试用途和次数并取得确认；本阶段未执行真实模型质量验收。

## 需要理解的核心代码

`read_chapter`：来源过滤 → metadata 校验 → 完整标题前缀 → 原文顺序/歧义 → 章节结果。

`generate_quiz`：校验请求 → 读取章节 → Pydantic 出题 → Python 校验 → 模型审校 → 保存完整题目 → 公共白名单。

`validate_candidates`：来源/逐字依据 → 确定性去重 → 各题型题量上限。

`QuizStore.save/get`：后端保留标准答案，前端只拿 ID；保存与读取均复制数据。

16A 测试保留原范围；16B 测试单独验证批改和报告流程。按本次用户指令先完成开发，暂不安排源码学习或个人验收，不自动进入 Phase 17。

## 16B 调用与状态

单题提交入口在 `grading.py`：

```python
result = submit_answer(
    {"quiz_id": quiz_id, "question_id": question_id, "user_answer": "A"},
    quiz_store=store,
)
```

题型由后端保存的题目决定，可选传 `question_type` 作一致性检查。请求拒绝标准答案、参考答案等额外字段。选择题只接受 A/B/C/D（小写可规范化）；问答答案必须是非空字符串，最多 10000 字符。选择题使用 Python 判分，不创建或调用 LLM。问答题调用现有 DeepSeek 的 Pydantic Structured Output，结合出题时保存的相关原文、参考答案、关键点和用户答案，只用优秀/良好/中等/及格/不及格。

批改成功后仅返回该题的标准答案和反馈，不暴露尚未作答题目的标准答案。`QuizStore.get_progress()` 也是内部数据，不得直接返回前端。

整次提交入口同样在 `grading.py`：

```python
result = submit_quiz(
    {"quiz_id": quiz_id, "answers": [
        {"question_id": first_question_id, "user_answer": "A"},
        {"question_id": second_question_id, "user_answer": "我用自己的话解释……"},
    ]},
    quiz_store=store,
)
```

必须覆盖这次实际生成的全部题目，不按原请求数量要求补交不存在的题。批改前先检查所有题目 ID、重复 ID、题型、答案和已保存提交的一致性。成功题保存结果，失败题列出 `failed_question_ids`，整个小测保持未完成，不生成报告。

也可以逐题调用 `submit_answer()`，最后调用 `report.py` 的 `complete_quiz(quiz_id, quiz_store=store)`。所有实际题目均批改成功后，才生成报告并返回 `completed`。报告失败时返回明确错误及已有批改结果，`report` 为 None，不假装完成。

### 重复提交及失败重试

- 同一题相同规范化答案：返回保存的结果，不重复模型调用。
- 同一题不同答案：返回 `submission_conflict`，不允许提交后改答案。
- 首次模型失败也保留原提交，允许用相同答案显式重试。
- 同题正在批改或报告正在生成：返回 `in_progress`，不启动重复任务。
- 整次重复提交：成功题用缓存；失败题才再次批改。若只剩报告失败，只重试报告。
- 锁仅保护进程内状态切换，模型调用在锁外执行；不提供跨进程协调或长期存储。
- 每个问答题每次提交最多一次模型调用；报告每次尝试一次。无自动付费重试。新小测最多 3 次问答批改加 1 次报告调用；即使全部是选择题，报告仍需一次模型调用。

### 报告一致性

答题概况、选择题正确数、问答评级和实际错误由 Python 读取保存结果组装，模型不能更改这些数据。`actual_mistakes` 表示本次结果指出的错误/遗漏；`possible_confusions` 表示章节中可能混淆的知识，并不代表用户犯过。

模型生成核心考点、可能混淆点、记忆方法及建议，使用 Pydantic 结构化输出。原文引用必须属于保存的章节；建议必须覆盖真实题目，错误/遗漏题归为“加强”，已准确完整答对题归为“巩固”。语义正确性仍依赖模型，逐字引用并不能证明每条解释都正确。问答评级和报告的真实模型质量尚未验证，当前通过的是 Fake Model 流程测试。

16B 最值得理解的业务函数是 `submit_answer`、`grade_short_answer`、`submit_quiz`、`build_report` 及 QuizStore 的占用/保存操作。这里仅列出代码地图，不安排当前个人学习验收。
