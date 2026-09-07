# Design QA

- source visual truth path: `C:\Users\ADMINI~1\AppData\Local\Temp\codex-clipboard-e5a5e45e-4c64-4985-b229-62316aa34ffd.png`
- current-page evidence path: `C:\Users\ADMINI~1\AppData\Local\Temp\codex-clipboard-421e8246-340f-4182-92bb-7c890b3c48f1.png`
- implementation URL: `http://127.0.0.1:8501`
- viewport: 1280 x 720 CSS px, density 1
- state: 智能问答首页、未配置 Key
- source pixels: 1536 x 1024
- provided current-page pixels: 2048 x 1116
- density normalization: 按桌面内容区域和相同未配置 Key 状态比较，浏览器 chrome 不计入差异

## Full-view comparison evidence

- 初次浏览器渲染确认：首页已显示四页导航、品牌、无 Key 提示、检索范围、模式切换、聊天区和折叠的“本次复习”。
- 初次截图发现品牌覆盖原生导航，属于 P1；随后根据真实 DOM 将品牌固定在左侧，并为原生顶部导航预留 250px 后居中。
- 修正后的页面刷新被浏览器安全策略拦截，无法取得同视口最终截图，因此不能把视觉 QA 标记为通过。

## Focused region comparison evidence

- 顶部区域：已测得 Streamlit 导航链接位于 `stToolbar` 下的 `.rc-overflow`，品牌初始范围为 x=22..246；修正 CSS 后导航容器左侧预留 250px，品牌符号和导航 emoji 已删除。
- 问答区：标题间距已压缩，模式按钮改为单行，聊天容器由 500px 调整为 420px，右侧复习卡改为默认折叠。

## Findings

- [P1 fixed] 品牌与导航重叠：品牌固定左侧，导航容器预留品牌宽度后居中。
- [P2 fixed] 首屏纵向信息过多：压缩标题与区块间距、简化模式按钮、缩短聊天区并折叠辅助信息。
- [P2 blocked] 缺少修正后同视口截图，无法最终确认导航视觉中心和 1280 x 720 下输入框露出程度。

## Required fidelity surfaces

- Fonts and typography: 延续现有中文系统字体、深色标题和层级；未引入新字体。
- Spacing and layout rhythm: 已按目标图收紧顶部、标题、模式和聊天区节奏；最终截图待补。
- Colors and visual tokens: 保留米白、墨绿、淡粉现有 token，不改变产品配色。
- Image quality and asset fidelity: 本次不新增图片资产；品牌符号按用户要求直接移除。
- Copy and content: 保留问答、来源、耗时、模式和 Key 提示，删除按钮副标题与冗余引导。

## Comparison history

1. 初次实现截图：品牌覆盖导航，问答页仍偏高。
2. 修正：品牌无符号固定左侧，导航项无 emoji 并居中；聊天区 420px；右侧信息折叠。
3. 修正后截图：浏览器策略阻止本地刷新，未能捕获。
4. 用户细节反馈：右侧说明不应折叠、图片上传不应独占输入框下方、未选模式不应保持橙色。
5. 修正：右侧改为常驻紧凑卡片；图片上传收进与输入框同行的“添加图片”弹出入口；模式按钮使用亮暗主题自适应白灰配色。

## Primary interactions tested

- 默认智能问答页成功运行。
- 四个页面 URL 已由 `st.navigation` 注册。
- 无 Key 时聊天输入和图片上传保持禁用。
- Python 编译与后端全量回归通过。
- AppTest 已确认两套入口均渲染快速/精确按钮、一个图片弹出入口、无折叠面板，并显示常驻“本次复习”说明。

## Console errors checked

- 修正后因浏览器刷新被阻止，无法完成最终 console 复查。

final result: blocked
