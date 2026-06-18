#!/usr/bin/env python
"""FundAgent LLM 路由测试

测试 FundAgent v3.1.0 的 LLM Tool Calling 路由功能。
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_fund_analysis():
    """测试基金分析功能"""
    from src.agents.fund_agent import FundAgent
    from src.a2a import Task, Message, TextPart

    print("\n" + "=" * 70)
    print("  FundAgent 基金分析测试")
    print("=" * 70)

    agent = FundAgent()

    # 正确方式：先创建 Task，再添加用户消息到 history
    task = Task(id="test_fund_001")
    user_message = Message(
        role="user",
        parts=[TextPart(text="分析基金 008089")]
    )
    task.add_message(user_message)

    print("\n📊 测试输入: '分析基金 008089'")
    print("-" * 50)

    try:
        result = await agent.process_task(task)
        print(f"\n📋 任务状态: {result.status.state.value}")

        if result.artifacts:
            for artifact in result.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            print(f"\n📄 响应内容:\n{part.text[:800]}...")
                            break
        elif result.status.message:
            print(f"\n📄 状态消息: {result.status.message}")

        return result.status.state.value == "completed"
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fund_position():
    """测试持仓查询功能"""
    from src.agents.fund_agent import FundAgent
    from src.a2a import Task, Message, TextPart

    print("\n" + "=" * 70)
    print("  FundAgent 持仓查询测试")
    print("=" * 70)

    agent = FundAgent()

    task = Task(id="test_fund_002")
    user_message = Message(
        role="user",
        parts=[TextPart(text="我的基金持仓情况")]
    )
    task.add_message(user_message)

    print("\n📊 测试输入: '我的基金持仓情况'")
    print("-" * 50)

    try:
        result = await agent.process_task(task)
        print(f"\n📋 任务状态: {result.status.state.value}")

        if result.artifacts:
            for artifact in result.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            print(f"\n📄 响应内容:\n{part.text[:800]}...")
                            break
        elif result.status.message:
            msg = result.status.message
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'text'):
                        print(f"\n📄 状态消息: {part.text}")
                        break

        return result.status.state.value == "completed"
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_buy_fund():
    """测试买入基金功能"""
    from src.agents.fund_agent import FundAgent
    from src.a2a import Task, Message, TextPart

    print("\n" + "=" * 70)
    print("  FundAgent 买入基金测试")
    print("=" * 70)

    agent = FundAgent()

    task = Task(id="test_fund_003")
    user_message = Message(
        role="user",
        parts=[TextPart(text="买入基金 021500 5000元")]
    )
    task.add_message(user_message)

    print("\n📊 测试输入: '买入基金 021500 5000元'")
    print("-" * 50)

    try:
        result = await agent.process_task(task)
        print(f"\n📋 任务状态: {result.status.state.value}")

        if result.artifacts:
            for artifact in result.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            print(f"\n📄 响应内容:\n{part.text[:800]}...")
                            break
        elif result.status.message:
            msg = result.status.message
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'text'):
                        print(f"\n📄 状态消息: {part.text}")
                        break

        return result.status.state.value == "completed"
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_preference_query():
    """测试偏好查询功能"""
    from src.agents.fund_agent import FundAgent
    from src.a2a import Task, Message, TextPart

    print("\n" + "=" * 70)
    print("  FundAgent 偏好查询测试")
    print("=" * 70)

    agent = FundAgent()

    task = Task(id="test_fund_004")
    user_message = Message(
        role="user",
        parts=[TextPart(text="我的投资偏好是什么")]
    )
    task.add_message(user_message)

    print("\n📊 测试输入: '我的投资偏好是什么'")
    print("-" * 50)

    try:
        result = await agent.process_task(task)
        print(f"\n📋 任务状态: {result.status.state.value}")

        if result.artifacts:
            for artifact in result.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            print(f"\n📄 响应内容:\n{part.text[:800]}...")
                            break
        elif result.status.message:
            msg = result.status.message
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'text'):
                        print(f"\n📄 状态消息: {part.text}")
                        break

        return result.status.state.value == "completed"
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("\n🚀 开始测试 FundAgent v3.1.0 (LLM 路由架构)...\n")

    results = {}

    # 测试 1: 基金分析
    results["fund_analysis"] = await test_fund_analysis()

    # 测试 2: 持仓查询
    results["fund_position"] = await test_fund_position()

    # 测试 3: 买入基金
    results["buy_fund"] = await test_buy_fund()

    # 测试 4: 偏好查询
    results["preference_query"] = await test_preference_query()

    # 打印测试总结
    print("\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  总计: {passed}/{total} 通过")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
