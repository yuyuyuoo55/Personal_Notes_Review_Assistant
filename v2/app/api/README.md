# Phase18 FastAPI接口与SSE

从项目根目录启动（完整依赖安装见 [V2 README](../../README.md)）：

```powershell
.\.venv\Scripts\python.exe -B -m uvicorn app.api.main:app --app-dir v2 --host 127.0.0.1 --port 8000 --workers 1
```

Swagger：http://127.0.0.1:8000/docs；OpenAPI：http://127.0.0.1:8000/openapi.json。
本地单用户，不提供生产鉴权/分布式存储。不要增加workers；reload和服务重启均会丢失进程内小测。
启动/health/docs不触发模型。实际上传、聊天、出题/问答批改/报告可能调用付费接口，使用前请确认预算；图文 ZIP 的图片数量可能增加视觉请求费用。

## 路由与请求

| 方法/路径 | 请求 | data返回 |
|---|---|---|
| GET /health | 无 | service、quiz_store、single_user、instance_id、test_mode、embedding_provider、collection、settings_enabled |
| GET /notes | 无 | [{source,filename}] |
| GET /notes/chapters | query source，使用/notes返回值 | [{Header 1,Header 2,Header 3}]，仅存在的标题字段 |
| POST /notes/upload | multipart file；markdown_images=false默认 | 业务导入原结果＋original_filename/stored_path/file_bytes |
| POST /chat/stream | JSON question、history=[]、web_enabled=false | SSE见下方 |
| POST /quiz/generate | JSON source、scope=chapter默认、headings={}、single_choice=1、short_answer=0 | QuizResult及scope/coverage，仅PublicQuestion |
| GET /settings/providers | 本机设置令牌 | 两服务脱敏状态、来源及已验证能力 |
| PUT /settings/providers/{deepseek或dashscope} | api_key、paid_validation_consent=true、max_requests=1；本机令牌 | 验证成功后保存并返回脱敏状态；失败保留旧配置 |
| DELETE /settings/providers/{deepseek或dashscope} | confirm=true；本机令牌 | DeepSeek删除后禁用；DashScope删除保存配置后回退.env（若存在） |
| POST /quiz/answer | JSON quiz_id、question_id、user_answer、可选question_type | 原AnswerResult，成功后只返回该题答案及解析/评级 |
| POST /quiz/submit | JSON quiz_id、answers=[{question_id,user_answer,可选question_type}] | 原QuizCompletionResult与成功保存的章节报告 |
| GET /quiz/result/{quiz_id} | path quiz_id | 只读QuizCompletionResult，不调用模型/生成报告 |

成功/业务返回统一 `{ "status": "业务状态", "data": ... }`；错误有status、data=null、message，参数错误附字段location/type，不回显请求正文。
HTTP：通常200；in_progress=202；未找到404；冲突/忙409；校验/导入failed422；格式415；大小413；模型失败502；任务超时504；内部异常500。
partial、partial_failure可以是HTTP200，前端必须检查status/data，不得只凭HTTP状态当成完整成功。
原业务状态如unchanged、insufficient_content、incomplete如实保留。

## 上传及溯源

- 仅.md/.markdown、PNG/JPG/JPEG/WebP、ZIP；不支持PDF。文件内容仍由现有导入模块验证，MIME/扩展名不能替代图片字节验证。
- 默认持久化目录 `v2/data/uploads`，Chroma仍使用原 `v2/data/chroma` / `v2_notes`。
- 保留原始单个文件名；拒绝路径、盘符、ADS、Windows保留名、非法字符及越界符号链接。只接收上传字节，不接受任意服务器路径参数。
- 同名文件视为该来源更新，大小写行为依Windows文件系统；无需临时路径source。ZIP原文件保留，图片通过ZIP成员溯源，临时提取文件不作为长期来源。
- Markdown2MiB、图片10MiB、ZIP100MiB；multipart整体101MiB，含无Content-Length的有界接收。ZIP内部安全/像素/数量限制沿用17A/17B。
- markdown_images=false仅导入文字；发现图片引用时返回 image_references、image_processed=0、vision_calls=0 及使用图文ZIP的警告，不读取图片、不请求Vision。文字导入状态仍按实际结果返回。正式前端固定使用此模式。
- markdown_images=true保留Phase17A能力：识别文件引用的图片，相对路径只允许uploads内文件；ZIP支持整包图片关联。缺失/不安全图片按原导入结果返回警告及失败/部分成功，不读取外部绝对路径。
- 原文件先持久化再导入，失败时原始文件仍可能已保存，旧索引可能保留；文件落盘与Chroma写入不是事务。partial必须展示warnings，ZIP外部图不会被静默处理。
- 超时返回504，同步线程可能继续完成写入/模型任务；占用持续到线程真正结束，前端不要自动重发以免重复费用。可稍后查询notes/quiz结果。

## 问答SSE

请求示例：
```json
{"question":"根据我的笔记解释RRF","history":[],"web_enabled":false}
```
仅允许user/assistant历史，最多20条，每条和当前问题最多10000字符；无服务器会话存储/Checkpointer，前端自行维护history。

`daily` 包含自然聊天、一般知识与通用实时信息查询；`knowledge` 用于明确依赖个人笔记的问题和对应追问。
联网关闭绝不搜索。开启时，daily 仅在实时信息需求或明确搜索请求下联网，knowledge 在原检索链后补充网页。
日常实时问题在联网关闭、失败或空结果时明确说明未获取实时资料，不返回笔记库不足。
网页引用编号为 `[W1]`，笔记为 `[1]`；只发送本轮实际引用的来源。

每个事件是UTF-8文本 `event: 名称\ndata: JSON\n\n`，content-type=text/event-stream：

```text
event: status
data: {"stage":"processing"}

event: token
data: {"text":"RRF"}

event: sources
data: {"items":[{"reference":1,"source_type":"note","id":"实际ID","content":"实际片段","metadata":{},"cited":true}]}

event: done
data: {"status":"completed","answer":"最终完整回答","route":"knowledge"}

```

- status：processing，或真实节点完成后 `{stage:节点名,state:completed}`；联网失败有安全warning。
- token：Daily/RAG回答模型stream原始增量，可能是多个字符/空白，不保证每次一个词。Router、查询改写、结构化输出不发送。
- sources：只发送本次最终回答中实际标注且校验通过的引用，cited必须为true；保留本地/网页区分和原编号，不重排。未引用的检索候选仍在检索状态中，不进入回答来源。无有效引用时items为空；前端不创建空折叠框。
- done：成功终止，answer为最终已校验的完整正文。无资料固定拒答没有模型Token，直接通过done返回；不会拆字伪造流。
- error：安全message、status=failed，可选code；失败终止且不再发送done。客户端断开时停止传递并关闭Graph流，可能没有终止事件。

Token是未校验草稿；只有done后才能标记完成。最终引用校验失败时先前Token不可保留为成功回答；前端应丢弃或明确标为失败。POST SSE请用httpx/requests逐行读取，不使用只支持GET的EventSource；跨多个chunk缓冲到空行再解析JSON。
底层同步SDK或模型加载已开始时不能保证立即停止；模型现有60秒超时与SSE总等待180秒约束，不能声称断开就不再产生费用。

## 小测

选择题0～5、问答0～3、至少一题。scope默认chapter，headings只能是Header1/2/3；{}沿用无标题正文语义，不是整篇笔记。scope=note时headings只能省略或{}，跨章节代表性抽样，资料JSON≤12000字符，完整结构化请求≤60000字符，保留完整Chunk与原文证据；不保证题量覆盖全部知识。
出题返回questions仅question_id/question_type/question/options；完整答案仅QuizStore保存。
answer/submit拒绝前端correct_answer/reference_answer等伪造字段，Python判选择题不初始化模型，问答用原章节Structured Output五级评级。
result GET只投影已成功批改题与已有报告，不返回StoredQuiz/未批改标准答案/内部answers快照；报告未生成时返回incomplete，重新提交相同整次答案可按原业务机制重试未成功部分或报告。
QuizStore服务重启丢失，原文件和Chroma持久化；默认单用户，无鉴权隔离。结果只在当前进程中存在。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_api.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_quiz.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_quiz_grading.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_image_import.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_zip_import.py
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_graph.py --offline --note-import
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_graph.py --offline
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_graph.py --offline --acceptance
.\.venv\Scripts\python.exe -B -X utf8 v2\tests\smoke_graph.py --offline --web-switch
```

Fake模型验证API调用链与增量时序，不是实时DeepSeek质量或ZIP批量识别验收。正式HTTP前端与实际离线联调见../../frontend/README.md；推荐start_v2.bat生成安全本机设置令牌，手动Uvicorn无令牌时Key管理拒绝启用。

## 本机 Key 安全

`GET /settings/deepseek/balance`：复用下述本机保护和密钥优先级；不接受前端传 Key 或服务端点。返回 `{status:ok,data:{is_available,balance_infos:[{currency,total_balance,granted_balance,topped_up_balance}],queried_at,http_status,http_requests}}`。金额保留官方字符串与 CNY/USD 币种。固定请求 `https://api.deepseek.com/user/balance`，15 秒超时、一次 GET、无重试/重定向；无生成模型调用。未配置 422，忙碌 409，远端/解析失败 502；不返回异常正文或密钥。前端主动刷新，不随页面/状态查询调用。

系统Windows Generic Credential保存Key与非秘密验证信息，不写普通JSON。DeepSeek优先级：禁用标记 → 已验证用户凭据 → .env。DashScope优先级：已验证用户凭据 → .env；删除保存配置后恢复环境回退，环境Key显示已配置但未验证。验证请求固定官方端点，max_requests=1，无SDK自动重试，不验证视觉权限、不写Chroma。验证与更换可能产生少量费用，必须显式同意。DashScope验证使用原生文本Embedding接口，一个短查询文本、1024维，失败不覆盖旧配置。正式服务固定text-embedding-v4 / v2_notes。
管理接口必须通过随机X-V2-Settings-Token、loopback来源、Host和Origin校验；不信任代理头，一键启动仅绑定127.0.0.1。所有导入/模型任务及整段SSE占用同一业务锁，执行中更换/删除返回409。这里只支持可信本机单用户，不提供公网鉴权。
