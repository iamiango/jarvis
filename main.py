#!/usr/bin/env python3
"""Jarvis Agent 主入口 - 交互式命令行"""
import asyncio
import sys
from src import JarvisAgent, skill_manager, mcp_manager


def print_banner():
    """打印欢迎信息"""
    print("\n" + "=" * 50)
    print("  🤖 Jarvis Agent - LangGraph + Ollama")
    print("=" * 50)
    print("  Model: qwen3.5:9b (Ollama)")
    print("  Commands:")
    print("    /skills  - 列出可用 Skills")
    print("    /mcp     - 列出 MCP Servers")
    print("    /quit    - 退出")
    print("=" * 50 + "\n")


def list_skills():
    """列出可用的 Skills"""
    skills = skill_manager.list_skills()
    if skills:
        print("\n📦 可用 Skills:")
        for name in skills:
            skill = skill_manager.get(name)
            print(f"  - {name}: {skill.description}")
    else:
        print("\n⚠️  暂无注册的 Skills")
    print()


def list_mcp_servers():
    """列出 MCP Servers"""
    servers = mcp_manager.list_servers()
    if servers:
        print("\n🔌 已连接 MCP Servers:")
        for name in servers:
            client = mcp_manager.get_client(name)
            tools = client.get_tools()
            print(f"  - {name}: {len(tools)} tools")
    else:
        print("\n⚠️  暂无连接的 MCP Server")
    print()


async def main():
    """主函数"""
    print_banner()

    # 创建 Agent
    print("🔧 正在初始化 Agent...")
    try:
        agent = JarvisAgent()
        # 发现并注册子 Agent（如果有运行的话）
        print("🔍 正在发现子 Agent...")
        await agent.discover_agents()
        print("✅ Agent 初始化完成!\n")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        print("请确保 Ollama 已启动并且模型已下载")
        print("运行: ollama pull qwen3.5:9b")
        return

    # 交互循环
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() == "/quit":
                print("\n👋 再见!")
                break
            elif user_input.lower() == "/skills":
                list_skills()
                continue
            elif user_input.lower() == "/mcp":
                list_mcp_servers()
                continue
            elif user_input.startswith("/"):
                print("❓ 未知命令。使用 /skills, /mcp 或 /quit")
                continue

            # 运行 Agent
            print("\n🤔 思考中...")
            response = await agent.arun(user_input)
            print(f"\nJarvis: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
