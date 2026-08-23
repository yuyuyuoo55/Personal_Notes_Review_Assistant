# 双模式 UI 验收记录

## 验收对象

- 参考稿：`C:\Users\Administrator\.codex\generated_images\01a005e7-b703-7af3-9daf-0a6882fbb69e\exec-f733663c-f82b-4481-8474-adec18fb11ba.png`
- 实现截图（快速模式）：`C:\Users\Administrator\.codex\visualizations\2026\08\15\01a005e7-b703-7af3-9daf-0a6882fbb69e\implementation-double-mode-v2.png`
- 对比图：`C:\Users\Administrator\.codex\visualizations\2026\08\15\01a005e7-b703-7af3-9daf-0a6882fbb69e\design-qa-comparison-v2.png`
- 本地预览：`http://127.0.0.1:8501`
- 验收视口：1280 × 720。

## 功能验收

| 项目 | 结果 | 说明 |
| --- | --- | --- |
| 快速模式默认选中 | 通过 | 显示“原问题 → 向量检索 → 基于 Top-3 片段回答”。 |
| 精确查找模式切换 | 通过 | 切换后显示“查询改写 → 向量 + BM25 → RRF → Cross-Encoder → 回答”。 |
| 模式说明同步更新 | 通过 | 右侧卡片会随模式展示速度或精度取舍。 |
| 聊天区域独立滚动 | 通过 | 问答历史使用固定高度容器，侧栏、顶部和输入框不会随历史一起滚动。 |

## 视觉验收

- 保留参考稿的暖白主背景、浅绿色侧栏、深蓝标题与鼠尾草绿强调色。
- 双模式选择器放在问答区顶部，用户能在提问前明确理解当前 RAG 链路。
- 已修复首次验收发现的选中按钮红色问题：主选中态改为鼠尾草绿色。
- 参考稿中示例含有已完成问答；本地截图为空会话，因此聊天容器内容不同，但布局、层级和模式入口一致。

## 控制台检查

- 页面未发现来自本项目的 JavaScript 错误。
- 浏览器仅记录一条 Chrome 扩展动态模块加载失败，与本地 Streamlit 应用无关。

## 最终结果

**passed**：双模式交互、布局和视觉方向均通过验收，可交由用户继续体验。
