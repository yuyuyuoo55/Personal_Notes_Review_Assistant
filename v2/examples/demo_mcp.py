"""运行方式：.venv/Scripts/python.exe -B -X utf8 v2/examples/demo_mcp.py"""

import asyncio
import json

from mcp_client import get_test_tools


async def main():
    """【业务需求】演示工具发现、参数查看和真实 MCP 调用。
    【执行步骤】1. 获取工具。2. 查看并选择加法工具。3. 调用并检查结果。
    """
    # 1. 连接本地 Server，获取真实工具列表。
    tools = await get_test_tools()
    assert len(tools) == 1

    # 2. 查看工具名称、描述、参数，选择要调用的工具。
    selected_tool = None
    for tool in tools:
        print("工具类型：", type(tool).__name__)
        print("工具名称：", tool.name)
        print("工具描述：", tool.description)
        print("工具参数：", json.dumps(tool.args, ensure_ascii=False))
        if tool.name == "add_numbers":
            selected_tool = tool
    assert selected_tool is not None
    assert selected_tool.args["a"]["type"] == "integer"
    assert selected_tool.args["b"]["type"] == "integer"

    # 3. 使用参数字典异步调用工具，检查 Server 返回的实际结果。
    arguments = {"a": 7, "b": 5}
    result = await selected_tool.ainvoke(arguments)
    print("实际输入：", arguments)
    print("实际输出：", json.dumps(result, ensure_ascii=False))
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "12"
    print("MCP 连接、工具发现、参数与真实调用检查通过")


if __name__ == "__main__":
    asyncio.run(main())
