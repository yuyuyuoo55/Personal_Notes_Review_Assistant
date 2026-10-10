"""本地 MCP 测试 Server：只提供加法工具，不需要 API Key。"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("V2 Local Test")


@server.tool()
def add_numbers(a: int, b: int) -> int:
    """【业务需求】计算两个整数的和。
    【执行步骤】1. 将 a 和 b 相加。2. 返回整数结果。
    """
    # 1. 计算输入的两个整数之和。
    total = a + b
    # 2. 将实际计算结果返回给 Client。
    return total


if __name__ == "__main__":
    # stdio 用于 MCP 通信，不能在 Server 中用 print 输出业务日志。
    server.run(transport="stdio")
