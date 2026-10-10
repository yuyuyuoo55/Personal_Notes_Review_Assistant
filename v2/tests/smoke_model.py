"""Phase 1 运行入口：发送一句问候，确认返回 AIMessage。"""

import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

# Direct script entrypoint: locate the V2 app package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import create_model


def main() -> None:
    """把固定问题交给模型，展示消息类型和正文，不打印模型对象或完整响应。"""
    # 此脚本只验证模型调用，不把本次运行交给外部追踪平台。
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    model = create_model()
    question = "你好，请用一句话介绍你自己"
    print("输入：", question)

    try:
        response = model.invoke(question)
    except Exception as error:
        # 不打印供应商原始异常，避免认证信息或请求详情进入终端。
        print("模型调用失败，异常类型：", type(error).__name__)
        print("请检查 Key、模型权限、API 地址和网络连接。")
        raise SystemExit(1) from None

    if not isinstance(response, AIMessage):
        raise RuntimeError("模型没有返回 AIMessage，请检查 SDK 和模型接口。")
    print("输出类型：", type(response).__name__)
    print("回答正文：", response.content)
    print("AIMessage 还可包含消息 ID、用量和响应元数据；本阶段只关注 content。")


if __name__ == "__main__":
    main()
