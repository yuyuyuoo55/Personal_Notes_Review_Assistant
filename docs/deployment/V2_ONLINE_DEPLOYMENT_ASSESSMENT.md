# V2 在线部署可行性评估

评估日期：2026-10-10。代码基线：V2 发布提交 `30137fd`。本次仅阅读源码、依赖和平台官方资料；未部署、开放本机 API、创建收费资源或调用付费模型。本文的工作量是相对判断，未承诺工期或最低资源规格。

## 推荐结论

**要保留完整真实 V2 AI 交互，优先 C：同一容器环境运行 Streamlit 与 FastAPI，只公开前端。** 先做小规模、受限的作品集演示，使用经过审核的公开示例资料；不要求永久保存访客笔记。临时上传仍需会话隔离、定期清理与费用限制，不能直接公开当前单用户程序。

A 可作为 V1 历史备用入口；B 可行但需要更多连接和安全改造；D 适合补充作品集展示页，不适合原样运行 Python V2。当前没有已验证的“免费、稳定、完整 V2”方案，最终平台取决于预算与实测资源。

## 四条路线比较

| 路线 | 代码复用与修改工作量 | 免费 / 成本可能性 | HR 免登录与真实 AI |
| --- | --- | --- | --- |
| A. 恢复 V1 Streamlit Cloud | 复用发布版 `streamlit_demo.py` 和 `backend/`；相对较低，需要检查 Cloud 配置、Secret 和实际构建日志 | 托管可免费；Embedding、生成、视觉等 API 仍有成本 | Public app 可免登录；原版需访客填 DeepSeek Key，体验有门槛；只有 V1 功能 |
| B. Cloud 前端 + 独立 V2 后端 | 前端页面及后端业务可复用；相对较高，需要远端 HTTPS 客户端、服务鉴权、云设置模式、两个平台运维 | 前端可免费；重依赖后端可能需付费实例，另有 API 成本 | 可以免登录，但服务端必须负责访客配额和隔离；保留真实 V2 |
| C. 完整 V2 容器 | 业务复用最多；中等改造，需要 Docker、Linux 构建、进程管理、云启动与受限演示模式 | 可选低成本 CPU 平台；不能承诺免费，重依赖和磁盘可能增加费用 | 可以只公开前端并免登录；保留真实检索、联网、小测与报告 |
| D. ChatGPT Sites | 可复用截图、架构、文案；交互 UI 需另写，Python 服务不能原样搬入 | 取决于 Sites 账户方案和额度；外部 AI 后端仍有成本 | Public Site 可匿名访问；静态展示不等于真实 V2，完整 AI 仍依赖独立后端 |

Streamlit 公共应用分享规则见 [官方分享说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)。Sites 的公开访问也依赖账户/工作区允许，尚未核对本账户设置；见 [Sites 官方说明](https://learn.chatgpt.com/docs/sites)。

## 当前代码的部署阻塞

| 源码位置 | 已核对的行为 | 部署影响 |
| --- | --- | --- |
| `streamlit_demo.py` | V1 在 Streamlit 内直接调用 backend 业务，不需要单独启动 FastAPI | A 应使用发布代码；不能拿本地未提交重构冒充已发布入口 |
| `v2/frontend/client.py` | 虽读取 `V2_API_URL`，但只接受 HTTP + loopback；Streamlit 服务端消费 SSE | B 不能只填远端 URL；需允许受控 HTTPS 和服务认证。当前浏览器不直连 API，无需为了分开部署盲目增加 CORS |
| `v2/start_local.py` | 固定 localhost、8000/8501；生成本机设置令牌并打开浏览器 | C 需要独立云启动方式、可配置前端监听端口和进程生命周期 |
| `v2/app/api/runtime.py`、`credentials.py`、`settings.py` | Linux 可使用环境 Key；没有 Windows 凭据存储时设置接口禁用 | 云上可使用平台 Secret，但需调整设置页；不是整个后端无法在 Linux 运行 |
| `v2/app/api/notes.py`、`quiz.py` | 业务接口没有生产鉴权；文件按原名覆盖；Quiz 没有所有者校验 | 不能把 API 直接公开；共享资料或 ID 不构成访客隔离 |
| `v2/frontend/pages.py`、`client.py` | 页面经服务端调用 localhost，并可带全局 settings token | 只公开前端也不能保护设置页；远程访客仍可能改共享 Key，需关闭或授权管理设置 |
| `v2/app/api/runtime.py`、`quiz/store.py` | 全局 job_lock，一次一个任务；小测/报告在内存 | 保持单 worker；并发会忙碌返回，重启丢失结果；无需为短期演示先上分布式框架 |

当前发布树没有 V2 Dockerfile / Compose。原 `.devcontainer` 属于历史开发环境，其 Python 配置与当前 Python 3.12 要求有差异；不是已验收的云部署方案。

## RAM、CPU 与依赖

V2 requirements 包含 pkuseg、sentence-transformers、torch。Cross Encoder 首次非空候选精排加载 `BAAI/bge-reranker-base`，本地没有缓存时下载；daily 和空候选不加载精排。代码没有要求必须用 GPU，CPU 可作为候选，但**尚未测量实际 RSS、加载峰值、CPU 延迟和并发容量**。BM25 当前每次读取整个集合，资料量增加会带来额外负担。

完整 V2 不宜直接承诺能跑在 512 MB 实例上。可先以 2–4 GB 作为小样本 CPU 压测候选区间，4 GB 更值得优先验证；这只是估算，不是最低配置结论。镜像预缓存模型、锁定可安装依赖、测量上传/精排/批改峰值，再选择规格。不要通过静默删除 BM25 或精排声称部署了完整同款 V2。

V1 根依赖允许没有重精排库，但缺少 pkuseg 会跳过 BM25，缺少 sentence-transformers 会跳过精排。轻量恢复要准确标注降级能力；当前未读取 Cloud 构建日志，不能断言历史 Demo 故障原因。

## 平台成本与存储事实

- **Streamlit Community Cloud**：免费托管有资源限制和休眠。官方资源页的约 690 MB–2.7 GB RAM、0.078–2 CPU 数据注明来自 2024 年 2 月，且会变化，不能当成当前保证；12 小时无访问会休眠。见 [资源与休眠说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)。本地文件不保证持久保存，可能随时删除，见 [文件存储限制](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data)。A/B 均应在面试前实际验证冷启动。
- **Render**：Free 为 512 MB / 0.1 CPU，见 [Compute Plans](https://render.com/docs/compute-plans)。15 分钟无流量休眠，唤醒约一分钟；免费服务不支持持久磁盘，重启/休眠后的临时文件会丢失，见 [免费限制](https://render.com/docs/free)。适合轻量验证，不作为完整 V2 默认推荐；付费规格与磁盘需按 [价格](https://render.com/pricing) 单独核算。
- **Hugging Face Docker Spaces**：CPU Basic 为 2 vCPU / 16 GB RAM / 50 GB 临时磁盘；无按小时硬件费，但当前创建 Gradio/Docker 计算空间要求个人 PRO 或组织付费方案，Static 才对所有人免费。见 [Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview) 和 [创建规则](https://huggingface.co/docs/huggingface_hub/main/guides/manage-spaces)。内存更宽裕但不等于零成本或已验收。
- **HF 存储**：当前推荐 Storage Buckets 挂载卷，临时盘重启丢失，见 [存储说明](https://huggingface.co/docs/hub/en/spaces-storage)；旧的 persistent storage 配置已不再可用，见 [配置参考](https://huggingface.co/docs/hub/en/spaces-config-reference)。Bucket 挂载不能未经测试就认定满足 Chroma/SQLite 文件锁与一致性要求；短期 Demo 优先可重建临时库，需稳定数据库时选择经过验证的本地持久块存储。

模型 API、Embedding、视觉、Tavily 的额度/计费与托管分开，免费主机不意味着免费 AI。本文未验证供应商赠送额度，也没有创建账单资源。

## Chroma、上传文件与状态

默认数据在 `v2/data/chroma`（collection `v2_notes`）和 `v2/data/uploads`。存储构造器可传目录，Runtime 可注入 uploads，但现有启动没有完整的存储路径环境配置。容器可以挂载现有目录；无需为了 Demo 直接换数据库。

推荐两种明确选择：

1. **短期作品集 Demo**：审核公开种子笔记；重启可恢复示例库；访客上传临时保存并隔离、清理，显式告知丢失边界。首次重建 Embedding 可能收费，需预算或准备可复现的公开示例索引，不能复制私人本机数据库。
2. **需要保存个人资料**：为 Chroma、uploads 和模型缓存选独立挂载；备份、恢复及删除要验收。小测/报告仍需另做持久化，挂载 Chroma 不会让进程内 QuizStore 自动保存。

当前同名文件覆盖、全局 Key 与知识库共享，未经隔离的“匿名上传”可能让其他访客看到资料。最小公开 Demo 可先提供只读共享样例、保留问答/小测/报告；如必须开放真实上传，需要增加会话命名空间或限定受保护的单用户演示入口。

## 密钥、公网与免登录体验

密钥放平台 Secret / 环境变量，不能写入 README、镜像层、前端脚本或浏览器。V1 Cloud 现有配置读取 DashScope/OSS secrets，生成使用访客 DeepSeek Key；恢复需对照实际代码检查 Secret。V2 云模式可由部署者提供预算有限的服务端 Key，页面不得暴露修改共享配置的能力。

HR 免登录体验技术上可行：公开入口、审核示例笔记、预设问题、真实模型回答、小测和报告。**免登录不等于无限制**：上线前必须限制请求频率、输入/上传大小、每会话任务数和供应商预算；匿名会话不能访问他人资料/小测。先关闭高成本批量上传或默认联网也是可选演示策略，但需明确展示限制，不能用假回复冒充真实 AI。

C 中 FastAPI 应继续绑定内部 loopback，仅 Streamlit 经 HTTPS 对外；平台需支持 WebSocket。SSE 在 Streamlit 服务端读取，仍需验证后台响应超时和断流；如代理 API 则禁用响应缓冲。保持单 worker，管理两个进程的退出、健康检查和重启。HF Docker 默认公开一个配置端口，内部可多端口，见 [Docker Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-docker)。

## ChatGPT Sites 的适用范围

Sites 适合托管项目介绍、已审核截图、架构图、运行边界和真实 Demo 链接。官方描述支持静态站与全栈 JavaScript/TypeScript 应用，见 [Sites 应用说明](https://learn.chatgpt.com/use-cases/build-and-deploy-internal-apps)。由该运行形态判断，现有 Streamlit、FastAPI、torch 与本地 Chroma 不能原样部署到 Sites；D1/R2 也不是现有 Chroma 目录的直接替代。

如果另写 Sites 前端调用独立后端，会增加 API 鉴权、跨域、增量协议和页面迁移工作，相当于另做 UI，不是本次最小发布路径。Public Site 可匿名，私有分享可能要求登录；上线资格和额度需核对账户/工作区政策。无需登录的展示页可以作为补充，但不代表完整 V2 已上线。

## 执行部署前必须验证

1. 确认演示范围：只读样例还是允许临时上传；配置费用上限与访客限制，隐藏/授权设置页。
2. 为 Linux/容器补云启动、环境 Key 模式与依赖安装验证，预缓存模型；保持单 worker，后端不直接公开。
3. 实测内存、CPU、模型加载与长任务/断流，确认平台冷启动和出网可用。
4. 确认临时数据重建/清理，或验证持久卷、备份与恢复；上传和小测不能跨访客泄露。
5. 在另行取得授权后，用有限次数真实模型调用验收引用、联网失败、小测与报告，并从未登录浏览器实测。

目前 A 未恢复、B/C 未部署、D 未创建。下一步由项目所有者选择路线和预算后再实施。
