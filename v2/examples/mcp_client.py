"""Phase 12：独立的本地 MCP Client，不接入知识问答 Graph。"""

import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_test_tools():
    """【业务需求】连接本地测试 Server，获取可以调用的 MCP 工具。
    【输入】无参数；使用当前 Python 环境和 V2 内的测试 Server。
    【执行步骤】1. 定位 Server。2. 初始化 stdio Client。3. 获取工具。
    【输出】LangChain 工具对象组成的列表。
    """
    # 1. 定位 V2 的本地测试 Server。
    server_path = Path(__file__).resolve().with_name("mcp_test_server.py")

    # 2. 使用当前 Python 启动 Server，通过标准输入/输出通信。
    client = MultiServerMCPClient({
        "local_test": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-B", "-X", "utf8", str(server_path)],
        }
    })

    """
    参数	你需要理解的意思
    local_test	给这个 MCP Server 起个名字
    transport: "stdio"	使用标准输入/输出与 Server 通信
    command: sys.executable	使用当前 Python 解释器启动程序
    args	启动 Python 时传入的参数和 Server 文件路径
    合起来就是一句话：
    告诉 MCP Client：请使用当前 Python 启动本地 MCP Server，并通过 stdio 与它通信。
    这段是连接配置，不是你自己实现 MCP 通信协议。
    """

    # 3. 连接 Server，获取工具名称、描述、参数及调用能力。
    tools = await client.get_tools()
    return tools
