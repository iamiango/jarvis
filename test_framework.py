#!/usr/bin/env python3
"""简单测试脚本 - 验证框架是否正常工作"""
import asyncio
from src import JarvisAgent, skill_manager
from src.skills import CalculatorSkill


async def test_skill():
    """测试 Skill 功能"""
    print("=" * 40)
    print("测试 1: Skill 直接调用")
    print("=" * 40)

    calc = CalculatorSkill()
    result = await calc.execute(operation="add", a=10, b=20)
    print(f"Calculator Skill (10 + 20) = {result.result}")
    print(f"Success: {result.success}")
    print()


async def test_skill_manager():
    """测试 Skill Manager"""
    print("=" * 40)
    print("测试 2: Skill Manager")
    print("=" * 40)

    # 注册 Skills
    skill_manager.register(CalculatorSkill())

    # 列出 Skills
    print(f"已注册 Skills: {skill_manager.list_skills()}")

    # 通过 Manager 调用
    result = await skill_manager.execute("calculator", operation="multiply", a=5, b=6)
    print(f"通过 Manager 调用 (5 * 6) = {result.result}")
    print()


async def test_agent_build():
    """测试 Agent 构建"""
    print("=" * 40)
    print("测试 3: Agent 构建")
    print("=" * 40)

    agent = JarvisAgent()
    agent.build()

    print(f"Agent Model: {agent.model_name}")
    print(f"Agent Base URL: {agent.base_url}")
    print(f"Graph 已编译: {agent.compiled_graph is not None}")
    print()


async def test_agent_run():
    """测试 Agent 运行（需要 Ollama 运行）"""
    print("=" * 40)
    print("测试 4: Agent 运行 (需要 Ollama)")
    print("=" * 40)

    try:
        agent = JarvisAgent()
        agent.build()

        # 简单测试
        response = agent.run("你好，请介绍一下你自己")
        print(f"Agent Response: {response[:200]}...")
    except Exception as e:
        print(f"⚠️  Agent 运行测试跳过: {e}")
        print("请确保 Ollama 已启动并运行 qwen3.5:9b 模型")
    print()


async def main():
    """运行所有测试"""
    print("\n🧪 Jarvis Agent 框架测试\n")

    await test_skill()
    await test_skill_manager()
    await test_agent_build()
    await test_agent_run()

    print("=" * 40)
    print("✅ 测试完成!")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
