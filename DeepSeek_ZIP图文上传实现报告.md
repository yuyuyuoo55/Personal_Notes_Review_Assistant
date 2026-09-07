# Markdown 图文 ZIP 上传实现报告

## 1. 本次目标

解决 Markdown 中包含本地图片时，云端无法读取用户电脑绝对路径的问题。

原有流程只上传 `.md` 文件。类似下面的图片地址只在用户电脑上有效：

```text
C:\Users\Administrator\Pictures\example.png
```

部署后的服务无法访问用户本机磁盘，因此 Markdown 文字可以导入，但本地图片只能跳过。

本次新增“Markdown + 配套图片 ZIP 整包上传”：用户将一份 Markdown 和图片目录一起压缩为 ZIP，系统在内存中读取、匹配并识别图片。

## 2. 用户使用方式

推荐目录：

```text
HTTP.zip
├─ HTTP.md
└─ images/
   ├─ request.png
   └─ response.png
```

Markdown 推荐使用相对路径：

```markdown
![HTTP 请求结构](images/request.png)
```

兼容规则：

- 图片目录可以使用 `images/`，也兼容已有资料中的 `assets/`。
- ZIP 中必须且只能包含一份 Markdown。
- 旧 Markdown 如果仍使用 `C:\...\example.png` 绝对路径，系统会在 ZIP 内按唯一文件名 `example.png` 兜底匹配。
- 上传格式必须是 `.zip`，暂不支持 `.rar`。

## 3. 对 `day01.rar` 的检查结果

提供的 `day01.rar` 内部包含：

- 1 份 Markdown：`苍穹外卖-day01.md`
- 77 个图片文件
- Markdown 中共 10 处图片引用
- 10 处引用都能在压缩包中按相对路径或唯一文件名找到

内部目录结构可以使用，但需要将 RAR 重新压缩为 ZIP 后再上传。

## 4. 主要实现

### 4.1 安全读取 ZIP

新增 `backend/app/services/markdown_bundle_service.py`：

- 使用 Python 标准库 `zipfile` 在内存中读取，不把整个压缩包直接解压到服务器目录。
- 找出 ZIP 中唯一的 Markdown。
- 收集 JPG、JPEG、PNG、GIF、WEBP 图片。
- 建立相对路径和唯一文件名索引，供 Markdown 图片引用匹配。
- 返回 Markdown 文本、Markdown 文件名和图片字节映射。

### 4.2 Markdown 图片匹配

扩展 `backend/app/services/markdown_image_service.py`：

- `data:image/...` 仍按原逻辑处理。
-公网 HTTPS 图片仍按原逻辑下载。
- ZIP 本地图片优先按相对路径匹配。
- 相对路径未命中时，按唯一文件名匹配，兼容旧绝对路径。
- 匹配后继续复用原有流程：校验图片、保存图片、调用视觉模型生成描述、写回 Markdown、生成图片知识片段。
- 单张图片失败不会中断整份 Markdown，继续记录 `image_skipped` 和安全提示。

### 4.3 双部署入口接入

已同步修改：

- `backend/app/api/notes.py`：FastAPI 双进程版本支持 `.zip`。
- `streamlit_demo.py`：Streamlit Cloud 单进程版本支持 `.zip`。
- `frontend/app.py`：上传控件允许选择 ZIP，并显示图文打包说明。

ZIP 导入成功后，知识库列表展示的是 ZIP 内 Markdown 的名称，而不是 ZIP 文件名。

### 4.4 前端提示

知识库上传区现在显示：

> 支持 Markdown / ZIP / JPG / PNG / WEBP。文档含图片时，请将图片放入同级 images 文件夹，与 Markdown 一起压缩为 ZIP 后上传。

## 5. 安全边界

ZIP 解析增加了以下限制：

- 最多 500 个文件。
- 解压后总大小最多 80MB。
- 拒绝加密 ZIP。
- 拒绝 `../`、绝对路径和 Windows 盘符等不安全成员路径，防止路径穿越。
- ZIP 必须且只能包含一份 Markdown。
- Markdown 必须使用 UTF-8 编码。
- 图片仍经过原有 MIME、文件签名和大小校验。
- 不支持 RAR，避免云端依赖 `unrar`、7-Zip 等额外系统程序。

## 6. 测试

新增 `tests/test_markdown_bundle.py`，覆盖：

1. 相对图片路径匹配。
2. Windows 本地绝对路径按唯一文件名匹配。
3. ZIP 路径穿越拦截。
4. 通过 `/api/notes/import` 完成 ZIP 图文导入。

最终验证结果：

```text
26 passed
```

相关 Python 文件编译通过，`git diff --check` 通过。

测试中的 3 条 warning 来自已有第三方依赖弃用提示和沙箱无法写入 `.pytest_cache`，不影响本次功能。

## 7. 本次涉及文件

| 文件 | 作用 |
| --- | --- |
| `backend/app/services/markdown_bundle_service.py` | 安全读取 ZIP、查找 Markdown、建立图片映射 |
| `backend/app/services/markdown_image_service.py` | 使用 ZIP 图片字节解析 Markdown 本地图片引用 |
| `backend/app/api/notes.py` | FastAPI ZIP 上传入口 |
| `streamlit_demo.py` | Cloud 单进程 ZIP 上传入口与提示 |
| `frontend/app.py` | 双进程前端 ZIP 选择与提示 |
| `tests/test_markdown_bundle.py` | ZIP 解析、安全和 API 回归测试 |
| `README.md` | 使用方式、支持范围与限制说明 |

## 8. 请 DeepSeek 重点检查

1. ZIP 路径规范化和路径穿越校验是否完整。
2. 解压文件数量、总大小限制是否合理，是否还需要限制压缩比或单文件大小。
3. 绝对路径按 basename 兜底匹配是否会产生歧义；当前只有文件名在 ZIP 中唯一时才建立 basename 映射。
4. ZIP 内图片全部保存在内存中是否需要改成按需读取，以降低大压缩包内存占用。
5. FastAPI 与 Streamlit Cloud 两套导入入口是否保持一致。
6. 图片识别中途失败时，已保存图片、Markdown 和向量数据的回滚是否还需要进一步加强。

## 9. 当前限制

- 不支持 RAR、7z。
- 一个 ZIP 只支持一份 Markdown。
- ZIP 中未被 Markdown 引用的图片不会进入知识库。
- 同名 Markdown 仍遵循原有规则：需要先删除旧笔记，不能直接覆盖。
- 图片识别会调用视觉模型，处理时间和费用随引用图片数量增加。
