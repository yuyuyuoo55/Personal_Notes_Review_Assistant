# 任务：UI 优化（导航卡顿根治 + 关于页 GitHub 图标 + 主页精简）

> 本文件是给 codex 的改造说明。目标：解决导航切换卡顿/点不到；关于页加 GitHub 图标并指向 GitHub；主页问答区精简。

---

## 0. 已确认决策

1. **导航卡顿根治**：把当前 `st.radio(key="active_page")` 切页，改为 **`st.navigation` + `st.Page`** 多页方案（Streamlit 1.60 完全支持）。
2. **关于页开源地址**：指向 **GitHub 主仓库** `https://github.com/yuyuyuoo55/Personal_Notes_Review_Assistant`，并加 **GitHub 图标**。
3. **主页问答区精简**：去掉大段 hero 介绍区（「从你的笔记里，重新理解知识」那套大标题/大段落），只留**简洁标题 + 当前检索范围**，像 UI 图一样紧凑。

---

## 1. 问题 1：导航切换卡顿 → 改用 st.navigation

### 现状
- `streamlit_demo.py` 用 `st.radio(["智能问答","知识库","设置","关于"], key="active_page")` 做切页。
- 问题：`radio` 切页会 rerun 整个脚本，且顶部 `list_notes()`（每次 rerun 都执行）重扫知识库 → 卡顿、偶尔点不到。
- 且 `if selected_page == "设置"/"关于": render_...(); st.stop()` 提前 stop，但知识库/问答页仍会先跑顶部重逻辑。

### 要改成 `st.navigation` + `st.Page`
Streamlit 1.60 支持 `st.navigation` + `st.Page`，它是**真·多页**：切页只跑当前页，不重跑其他页，显著更快更顺。

示例结构：
```python
pages = [
    st.Page(render_chat_page / page_func, title="智能问答", icon="💬", default=True),
    st.Page(render_library_page, title="知识库", icon="📚"),
    st.Page(render_settings_page, title="设置", icon="⚙️"),
    st.Page(render_about_page, title="关于", icon="ℹ️"),
]
nav = st.navigation(pages)
nav.run()
```
- 把现有 `render_settings_page` / `render_about_page` / `render_library_page` 以及「智能问答」主区，拆成独立页面函数，用 `st.Page` 注册。
- **`list_notes()` / 顶部重逻辑只在需要用到的页面内调用**（不要在 `st.navigation` 之前无条件跑），或在页面函数内 `@st.cache_data` 缓存。
- 顶部品牌区（`app-brand`）可作为 `st.navigation` 上方的小 header 保留，或每页顶部显示。

### 关键注意
- 拆分页面时**不要丢失现有逻辑**：问答、图片问答、资料导入、删除/清空、Key 跨页共享、来源及耗时展示都要保留。
- `st.session_state` 的 Key（`deepseek_api_key`/`validated_api_key`）跨页共享不变。
- 页面切换后状态保留（Key、已导入笔记列表、聊天历史）。

---

## 2. 问题 2：关于页加 GitHub 图标 + 指向 GitHub

- 当前 `render_about_page`（`streamlit_demo.py` 第 408-425 行）里开源地址是 `<a href='https://gitee.com/...'>查看项目仓库 ↗</a>`。
- **改成**：
  ```html
  <p><strong>开源地址</strong><br>
  <a href='https://github.com/yuyuyuoo55/Personal_Notes_Review_Assistant' target='_blank'>
    <span class='github-link'>GitHub</span> ↗</a></p>
  ```
- **加 GitHub 图标**：可用 SVG 内联或 Font Awesome。推荐内联一个常见 GitHub 标志 SVG（白/暗色自适应），放在链接文字前。示例：
  ```html
  <a href='https://github.com/yuyuyuoo55/Personal_Notes_Review_Assistant' target='_blank'>
    <svg width='16' height='16' viewBox='0 0 16 16' fill='currentColor'>...github octocat path...</svg>
    GitHub ↗</a>
  ```
- 若已用现有 `.tech-chip` 等样式，保持视觉统一。

---

## 3. 问题 3：主页问答区精简（去掉大 hero）

- 当前「智能问答」页顶部有一段大 hero：
  ```
  从你的笔记里，重新理解知识
  提出问题，系统只依据已导入的学习资料回答，并保留可回看的来源。
  当前检索范围 · X 份笔记 · Y 个片段
  ```
- **精简为**：去掉大标题「从你的笔记里，重新理解知识」和大段介绍，**保留简洁标题 + 「当前检索范围」**。像 UI 图那样紧凑：
  ```
  智能问答
  当前检索范围 · X 份笔记 · Y 个片段
  ```
- 可保留一行小副标题（可选），但**不要占很大版面**。
- 「本次复习」右侧栏可保留或精简，视需要。

---

## 4. 注意事项 / 不要踩的坑

1. **st.navigation 拆页时**：把每个页面逻辑移入独立函数；确保 `st.session_state` 的 Key 和笔记数据跨页共享不丢。
2. **不要破坏现有功能**：问答、图片问答、导入、删除/清空、Key 验证、图片压缩都要保留。
3. **性能**：`list_notes()` 用 `@st.cache_data` 缓存（知识库未变时复用），避免切页卡顿。
4. **点击不到**：`st.navigation` 的页签点击应稳定；若仍有问题，检查是否有 `st.rerun()`/`st.stop()` 干扰。
5. **兼容**：`streamlit_demo.py` 作为 Cloud 入口，改后要在 Streamlit Cloud 正常跑。
6. **分步做**：先拆页面函数 + st.navigation，再分别优化关于页/主页，每步确保 `python -c "import streamlit_demo"` 可过。

---

## 5. 完成标准（验收）

- [ ] 导航换成 `st.navigation` + `st.Page`，4 页切换流畅、不卡顿、点得到
- [ ] 切页只跑当前页，不重跑其他页；`list_notes` 缓存
- [ ] 关于页开源地址指向 GitHub 主仓库，带 GitHub 图标
- [ ] 主页问答区精简，去掉大 hero，只留简洁标题 + 检索范围
- [ ] 现有功能（问答/图片/导入/删除/Key验证/压缩）全部保留
- [ ] Key 跨页共享、笔记列表/聊天历史切页不丢
- [ ] `pytest tests/` 全绿（UI 之外逻辑不坏）
