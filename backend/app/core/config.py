"""统一读取项目环境变量与本地存储路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

# config.py → core → app → backend → 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 全项目只在这里加载一次 .env；文件不存在时不会报错。
load_dotenv(PROJECT_ROOT / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEEPSEEK_TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")
DEEPSEEK_VISION_MODEL = os.getenv(
    "DEEPSEEK_VISION_MODEL",
    "deepseek-v4-flash-vision-exp",
)
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

# OSS 与文档图片识别均为部署者配置，绝不发送给前端。
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")
OSS_PUBLIC_BASE_URL = os.getenv("OSS_PUBLIC_BASE_URL")
OSS_OBJECT_PREFIX = os.getenv("OSS_OBJECT_PREFIX", "personal-notes/images").strip("/")
OSS_URL_EXPIRES_SECONDS = int(os.getenv("OSS_URL_EXPIRES_SECONDS", "3600"))
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
VLM_FALLBACK_TO_QWEN = os.getenv("VLM_FALLBACK_TO_QWEN", "false").lower() == "true"
VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "60"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))

# RAG 组件使用的名称与本地存储位置。
# 这些只是“配置值”；实际对象由 vector_store.py 和 rag_service.py 创建。
DASHSCOPE_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_EMBEDDING_MODEL",
    "text-embedding-v4",
)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "notes")

def _resolve_project_path(value: str) -> Path:
    """将相对路径按项目根目录解析，绝对路径则原样使用。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# 允许 .env 在不同环境指定 Chroma 数据目录；本机默认保存在项目内。
CHROMA_PERSIST_DIRECTORY = _resolve_project_path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")
)
# BM25 的索引与 corpus 会保存到此路径；相对路径同样基于项目根目录。
BM25_INDEX_PATH = _resolve_project_path(
    os.getenv("BM25_INDEX_PATH", "data/bm25/notes.bm25")
)
# 用户通过 FastAPI 上传的 Markdown 文件统一保存到项目内的此目录。
UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"
LOG_DIRECTORY = PROJECT_ROOT / "logs"


def require_dashscope_api_key() -> str:
    """在真正创建 Embedding 时校验 DashScope Key。"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未找到 DASHSCOPE_API_KEY，请在项目根目录的 .env 文件中配置。"
        )

    return DASHSCOPE_API_KEY
