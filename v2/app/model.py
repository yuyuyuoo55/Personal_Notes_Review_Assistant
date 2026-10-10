
"""
【业务需求】
使用 LangChain 统一接口初始化 DeepSeek 大模型。

【执行步骤】
1. 加载 .env 环境变量。
2. 获取模型配置。
3. 使用 init_chat_model 创建模型。
4. 返回模型对象，供 LangGraph 使用。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# =============== Step 1：加载环境变量 ===============

load_dotenv()


# =============== Step 2：创建模型 ===============

def create_model(*, api_key=None):
    """创建并返回 LangChain 聊天模型。"""

    # 1. 获取 API Key
    api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY")

    # 2. 获取模型配置
    model_name = os.getenv(
        "DEEPSEEK_TEXT_MODEL",
        "deepseek-v4-flash"
    )

    base_url = os.getenv(
        "DEEPSEEK_API_BASE_URL",
        "https://api.deepseek.com"
    )

    # 3. 初始化模型
    model = init_chat_model(
        model=model_name,
        model_provider="deepseek",
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=0
    )

    # 4. 返回模型对象
    return model
