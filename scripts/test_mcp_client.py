#!/usr/bin/env python3
"""测试 MultiServerMCPClient"""
import asyncio
import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp import MultiServerMCPClient, create_tavily_client


async def test_manual_add():
    """测试手动添加 MCP Server"""
    print("\n" + "=" * 60)
    print("测试 1: 手动添加 Tavily MCP Server")
    print("=" * 60)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("⚠️ 未设置 TAVILY_API_KEY 环境变量，跳过此测试")
        return

    async with MultiServerMCPClient() as client:
        # 添加 Tavily
        success = await client.add_stdio_server(
            name="tavily",
            command="npx",
            args=["-y", "tavily-mcp"],
            env={"TAVILY_API_KEY": api_key},
        )

        if not success:
            print("❌ 连接 Tavily MCP Server 失败")
            return

        # 获取工具列表
        tools = client.get_tools()
        print(f"\n📋 可用工具 ({len(tools)} 个):")
        for tool in tools:
            print(f"  - {tool['full_name']}: {tool.get('description', '')[:50]}...")

        # 测试搜索
        print("\n🔍 测试搜索: 'Python asyncio tutorial'")
        result = await client.call_tool(
            "tavily",
            "search",
            {"query": "Python asyncio tutorial", "max_results": 3}
        )

        if result["success"]:
            print(f"✅ 搜索成功:")
            print(result["result"][:500] + "..." if len(result["result"]) > 500 else result["result"])
        else:
            print(f"❌ 搜索失败: {result['error']}")


async def test_load_from_config():
    """测试从配置文件加载"""
    print("\n" + "=" * 60)
    print("测试 2: 从配置文件加载 MCP Servers")
    print("=" * 60)

    config_path = os.getenv("MCP_CONFIG_PATH", "config/mcp_servers.json")
    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在: {config_path}")
        return

    # 替换环境变量
    import json
    with open(config_path, "r") as f:
        config_text = f.read()

    # 简单的环境变量替换
    for key, value in os.environ.items():
        config_text = config_text.replace(f"${{{key}}}", value)

    config = json.loads(config_text)

    async with MultiServerMCPClient() as client:
        count = await client.load_from_config(config)
        print(f"\n✅ 成功加载 {count} 个 MCP Server")

        # 显示状态
        status = client.get_all_status()
        for name, info in status.items():
            print(f"\n📊 {name}:")
            print(f"   - 传输: {info['transport']}")
            print(f"   - 连接状态: {'✅' if info['connected'] else '❌'}")
            print(f"   - 工具数量: {info['tools_count']}")


async def test_create_tavily_client():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("测试 3: 使用 create_tavily_client 便捷函数")
    print("=" * 60)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("⚠️ 未设置 TAVILY_API_KEY 环境变量，跳过此测试")
        return

    try:
        client = await create_tavily_client(api_key)

        # 获取 LLM 格式的工具 Schema
        schemas = client.get_tool_schemas_for_llm()
        print(f"\n📋 LLM Tool Schemas ({len(schemas)} 个):")
        for schema in schemas:
            print(f"  - {schema['function']['name']}")

        await client.close_all()
        print("\n✅ 测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("MultiServerMCPClient 测试")
    print("=" * 60)

    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    await test_manual_add()
    await test_load_from_config()
    await test_create_tavily_client()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
