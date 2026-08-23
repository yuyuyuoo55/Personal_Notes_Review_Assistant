"""统一日志工具，调用方式接近 Java 的 logger.info(...)。"""

import logging
from logging.handlers import RotatingFileHandler

from backend.app.core.config import LOG_DIRECTORY

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _configure_root_logger() -> logging.Logger:
    """初始化一次项目根日志器；重复调用不会重复添加 Handler。"""
    root_logger = logging.getLogger("personal_notes_assistant")

    if root_logger.handlers:
        return root_logger

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_DIRECTORY / "app.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """返回模块日志器，使用：logger = get_logger(__name__)。"""
    _configure_root_logger()
    return logging.getLogger(f"personal_notes_assistant.{module_name}")
