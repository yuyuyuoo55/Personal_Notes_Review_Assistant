# 修复：图片导入超时 + 图片压缩（给 codex）

> 本文件是给 codex 的改造说明。目标：解决「导入图片时 DeepSeek Vision 识别超时/失败、被笼统提示『已跳过』」的问题。
> 根因：免费 Streamlit Cloud + DeepSeek Vision 识别图片，60s 超时不够；图片过大/尺寸过大也会导致慢或失败。

---

## 0. 已确认决策（不要再改）

1. **调大超时**：`VLM_TIMEOUT_SECONDS` 默认 60 → **180**。
2. **图片压缩**：导入图片时先用 Pillow 压缩/缩放，降低尺寸与体积，再发给 Vision，提升识别成功率。
3. **加 `pillow` 依赖**到 `requirements.txt`（云端需要真装，不能只在本地）。
4. **降级明确提示**：压缩后仍识别失败/超时，提示明确（如「图片过大或无法识别，请更换更清晰的小图」），不笼统说「超时」。

---

## 1. 要改的文件

### 文件 1：`backend/app/core/config.py`

- `VLM_TIMEOUT_SECONDS` 默认 60 → **180**：
  ```python
  VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "180"))
  ```
- `MAX_IMAGE_BYTES` 可保留 8MB，或调为更合理值（建议保留 `8 * 1024 * 1024`，压缩步骤会处理真正大图）。

### 文件 2：`.env.example`

- `VLM_TIMEOUT_SECONDS=60` → `180`，方便部署者知晓默认改为 180。

### 文件 3：`backend/app/services/multimodal_service.py` — 新增/调整图片压缩

新增一个**离线压缩函数**（用 Pillow），在 `save_image_to_local` 或 `build_image_chunk` 调用前对图片做处理：

```python
def _maybe_compress_image(image_bytes: bytes, content_type: str) -> bytes:
    """若图片过大/尺寸过大，用 Pillow 压缩并缩放到合理尺寸，返回压缩后的字节；其他情况原样返回。"""
    try:
        from PIL import Image
        import io
        # 1) 先读原始字节，判断尺寸/体积
        # 2) 若超过目标(如最长边 1600px 或 体积 > 2MB)，缩放 + 重存(JPEG/PNG)，控制体积
        # 3) 返回压缩后的 bytes；压缩后仍不达标则原样返回（交给上层提示）
    except Exception:
        return image_bytes  # 没有 Pillow 或处理失败时，保持原样，不中断
```

- **调用位置**：`build_image_chunk`（`multimodal_service.py`）在 `save_image_to_local` 之前，先对 `image_bytes` 做 `_maybe_compress_image`。
- 或放在 `save_image_to_local` 内部统一处理（更省事）。**建议在 `build_image_chunk` 里处理，统一影响所有图片导入路径。**

### 文件 4：`requirements.txt`

- 加 `pillow>=10,<12`（或与当前 Pillow 兼容版本），确保云端能装。

---

## 2. 降级提示（用户可读）

- 当 `describe_image_url` 抛 `TimeoutException`/`HTTPError` 后，`build_image_chunk` 或上层 `import_note` 的 except 里，把信息改成**明确、可操作**的中文：
  - 例如 `ImageProcessingError("图片较大或无法识别，已跳过。请更换更清晰、较小尺寸的图片后重试。")`
- 不要只显示「超时」这种笼统字样。当前 `describe_image_url` 里 `httpx.TimeoutException` → `ImageProcessingError("图片识别超时，请稍后重试")` 可改为更明确。

---

## 3. 注意事项 / 不要踩的坑

1. **Pillow 只在压缩时用**，`import PIL` 要放函数内或 try/except（本地有，云端要装，缺失时不能崩）。
2. **压缩不要过度**：目标足够识别即可（最长边 1600px、体积 ≤ 2MB 较稳），别压到看不清字。
3. **保留原图**：压缩只影响**发给 Vision 识别**的字节；**存到本地的原图仍可保留**（或存压缩版，选一种，确保 metadata 的 `image_path` 指向实际存在文件）。建议：存压缩后用于识别，`image_path` 指向该压缩图（更省空间）；若想保留原图，可另存。
4. **不改变 `MAX_IMAGE_BYTES` 校验语义**：上传超限仍提示，但压缩步骤让「够大但可处理」的图也能识别。
5. **两套入口同步**：`streamlit_demo.py`（单进程）与 `backend/app/api/notes.py`（双进程）的图片导入都要走压缩逻辑（`build_image_chunk` 统一，无需各自改）。
6. **测试**：用 mock 或假图验证 `_maybe_compress_image` 不抛异常、不破坏原流程。外部 API 用 mock，不产生费用。

---

## 4. 完成标准（验收）

- [ ] `VLM_TIMEOUT_SECONDS` 默认 180，`.env.example` 更新
- [ ] 图片导入前自动压缩（最长边限制 + 体积控制），用 Pillow
- [ ] `requirements.txt` 加入 `pillow`
- [ ] 识别失败/超时时提示明确、可操作（不是笼统「超时」）
- [ ] 不破坏现有图片导入、.md 导入、删除、检索功能
- [ ] 无 Pillow 环境（云端未装）时不崩，降级为原样识别
- [ ] `pytest tests/` 全绿（图片相关用 mock）
