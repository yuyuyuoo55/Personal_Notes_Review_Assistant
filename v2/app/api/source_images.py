"""仅从当前 Runtime 的真实 Chunk 读取原图；不接受图片路径或模型 URL。"""
import io
import re
import warnings
import zipfile
from hashlib import sha256
from pathlib import Path

from PIL import Image

from ..knowledge.image_import import read_image, IMAGE_FORMATS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
from ..knowledge.zip_import import inspect_zip
from .runtime import ApiError


def _owned_file(value, root):
    if not isinstance(value, str) or value.startswith(("//", "\\\\")):
        raise ValueError("不读取网络共享路径。")
    path = Path(value)
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("图片来源越界或不存在。")
    return resolved


def source_image(runtime, chunk_id):
    """【业务需求】回答来源原图只能在自己的访客知识库及上传目录内读取。
    【输入】当前鉴权 Runtime、真实 Chunk ID，不接收文件路径。
    【执行步骤】回查片段；限制上传边界；安全读取单图/ZIP；验证导入时的图片哈希。
    【输出】已验证的原图 bytes、MIME；失效或非法来源统一拒绝且不泄露路径。
    """
    if not re.fullmatch(r"[a-f0-9]{64}", chunk_id):
        raise ApiError(404, "image_not_found", "当前会话没有可读取的来源图片。")
    try:
        records = runtime.store().get(ids=[chunk_id], include=["metadatas"])
        if records["ids"] != [chunk_id]:
            raise ValueError("片段不存在。")
        metadata = records["metadatas"][0]
        if metadata.get("is_image_chunk") is not True or metadata.get("chunk_id") != chunk_id:
            raise ValueError("不是图片片段。")
        root = runtime.upload_root.resolve()
        source = _owned_file(metadata["source"], root)
        if "zip_path" in metadata:
            package = _owned_file(metadata["zip_path"], root)
            if package != source:
                raise ValueError("ZIP 来源不一致。")
            manifest = inspect_zip(package)
            member = metadata["image_member"]
            if member not in manifest["members"]:
                raise ValueError("图片成员不存在。")
            with zipfile.ZipFile(package) as archive, archive.open(manifest["members"][member]) as stream:
                raw = stream.read(MAX_IMAGE_BYTES + 1)
            if not raw or len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("图片体积超限。")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    expected = "JPEG" if Path(member).suffix.lower() in {".jpg", ".jpeg"} else Path(member).suffix[1:].upper()
                    if image.format not in IMAGE_FORMATS or image.format != expected or image.width * image.height > MAX_IMAGE_PIXELS or max(image.size) > 8192:
                        raise ValueError("图片格式或像素无效。")
                    mime = IMAGE_FORMATS[image.format]
                    image.verify()
        else:
            picture = _owned_file(metadata["image_path"], root)
            raw, mime, _ = read_image(picture)
        if sha256(raw).hexdigest() != metadata["image_hash"]:
            raise ValueError("原图已改变。")
        return raw, mime
    except ApiError:
        raise
    except Exception:
        raise ApiError(404, "image_not_found", "当前会话没有可读取的来源图片，可能已删除或变更。") from None


def public_sources(sources, *, cloud=False):
    """仅改变公共来源展示；内部检索和出题 metadata 保持不变。"""
    result = []
    for source in sources:
        item = dict(source)
        metadata = dict(item.get("metadata") or {})
        image = metadata.get("is_image_chunk") is True
        if item.get("source_type") == "note" and (image or cloud):
            # 白名单避免 ZIP/image_path/source 及其他服务器绝对路径进入公共 SSE。
            metadata = {key: value for key, value in metadata.items()
                        if key in {"Header 1", "Header 2", "Header 3", "filename", "chunk_index", "chunk_id", "is_image_chunk"}}
            item["metadata"] = metadata
            item.pop("source", None)
        if item.get("source_type") == "note" and image and item.get("cited") is True and re.fullmatch(r"[a-f0-9]{64}", str(item.get("id", ""))):
            item["image_chunk_id"] = item["id"]
        result.append(item)
    return result
