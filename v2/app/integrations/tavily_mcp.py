"""通过 Tavily 官方远程 MCP 获取网页搜索结果。"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_tavily_tools():
    """【业务需求】安全连接 Tavily 官方 MCP，获取真实工具。
    【执行步骤】1. 加载根目录配置。2. 检查 Key。3. 通过请求头连接并获取工具。
    【输出】LangChain 工具对象列表。
    """
    # 1. 复用项目根目录 .env，不创建第二份配置。
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")
    # 2. 只读取密钥，不打印或将它放入 URL。
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("请在项目根目录 .env 配置 TAVILY_API_KEY。")
    # 3. 用官方远程 HTTP Server 获取工具。
    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/",
            "headers": {"Authorization": "Bearer " + api_key},
        }
    })
    return await client.get_tools()


async def search_web(query):
    """【业务需求】用改写后的问题执行真实网页搜索，保留实际来源。
    【输入】非空 rewritten_query 文本。
    【执行步骤】1. 检查问题。2. 获取搜索工具。3. 调用。4. 解析真实结果。
    【输出】包含实际 title、url、content 等字段的字典列表；失败抛出异常。
    """
    # 1. 不用空问题发起付费搜索。
    if not isinstance(query, str) or not query.strip():
        raise ValueError("联网搜索需要非空 rewritten_query。")
    # 2. 选择已确认的官方搜索工具，不交给模型自动选择。
    tools = await get_tavily_tools()
    search_tool = None
    for tool in tools:
        if tool.name == "tavily_search":
            search_tool = tool
    if search_tool is None:
        raise ValueError("Tavily MCP 没有提供 tavily_search 工具。")
    # 3. 使用基础搜索，最多返回三条，避免额外搜索消耗。
    response = await search_tool.ainvoke({
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
    })
    # 4. MCP 返回文本内容块；解码其中的 JSON，不补造来源字段。
    results = []
    parsed = False
    if not isinstance(response, list):
        raise ValueError("Tavily 搜索返回必须是内容块列表。")
    for block in response:
        if not isinstance(block, dict):
            raise ValueError("Tavily 搜索返回了无效内容块。")
        if block.get("type") == "text":
            payload = json.loads(block["text"])
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError("Tavily 搜索返回缺少 results。")
            parsed = True
            for item in payload["results"]:
                if not isinstance(item, dict):
                    raise ValueError("Tavily 搜索结果必须是字典。")
                result = {}
                for field in ["title", "url", "content", "score", "raw_content"]:
                    if field in item:
                        result[field] = item[field]
                results.append(result)
    if not parsed:
        raise ValueError("Tavily 搜索没有返回可解析的文本结果。")
    return results
