#!/usr/bin/env python3
"""测试 StockAgent 市场行情概览功能

测试用例:
1. "帮我看看今天A股行情怎么样" - 应该走 market_overview
2. "分析股票 600519" - 应该走 stock_analysis
3. "大盘走势如何" - 应该走 market_overview
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_task_classifier():
    """测试任务分类器"""
    from src.agents.stock_agent import StockAgent, StockTaskType

    print("=" * 70)
    print("  StockAgent 任务分类器测试")
    print("=" * 70)

    agent = StockAgent(enable_mcp_search=False)

    test_cases = [
        # (输入, 预期类型)
        ("帮我看看今天A股行情怎么样", StockTaskType.MARKET_OVERVIEW),
        ("大盘走势如何", StockTaskType.MARKET_OVERVIEW),
        ("今天股市什么情况", StockTaskType.MARKET_OVERVIEW),
        ("上证指数怎么样", StockTaskType.MARKET_OVERVIEW),
        ("分析股票 600519", StockTaskType.STOCK_ANALYSIS),
        ("帮我看看 000001 的技术指标", StockTaskType.STOCK_ANALYSIS),
        ("贵州茅台走势如何", StockTaskType.STOCK_ANALYSIS),  # 有股票名称
        ("帮我分析一下", StockTaskType.STOCK_ANALYSIS),  # 有分析关键词，无代码
    ]

    print("\n📝 分类测试:\n")
    passed = 0
    for text, expected in test_cases:
        result = agent._classify_task_type(text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"{status} '{text}'")
        print(f"   预期: {expected.value}, 实际: {result.value}")
        print()

    print(f"结果: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


async def test_market_overview():
    """测试市场行情概览功能"""
    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart

    print("\n" + "=" * 70)
    print("  StockAgent 市场行情概览测试")
    print("=" * 70)

    # 创建 Agent（禁用 MCP 搜索以加速测试）
    agent = StockAgent(enable_mcp_search=False)

    # 创建任务 - 需要将消息添加到 history 中
    user_message = Message(
        role="user",
        parts=[TextPart(type="text", text="帮我看看今天A股行情怎么样")],
    )
    task = Task(id="test_market_001")
    task.add_message(user_message)

    print("\n📊 测试输入: '帮我看看今天A股行情怎么样'")
    print("-" * 50)

    # 执行任务
    try:
        result = await agent.process_task(task)
        print(f"\n📋 任务状态: {result.status.state.value}")

        # 提取响应文本
        if result.artifacts:
            for artifact in result.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            print(f"\n📄 响应内容:\n{part.text[:1000]}...")
                            break

        return result.status.state.value == "completed"
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_stock_analysis():
    """测试个股分析功能（确保原有功能不受影响）"""
    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart

    print("\n" + "=" * 70)
    print("  StockAgent 个股分析测试")
    print("=" * 70)

    agent = StockAgent(enable_mcp_search=False)

    task = Task(
        id="test_stock_001",
        message=Message(
            role="user",
            parts=[TextPart(type="text", text="分析股票 000001")],
        ),
    )

    print("\n📊 测试输入: '分析股票 000001'")
    print("-" * 50)

    result = await agent.process_task(task)

    print(f"\n📋 任务状态: {result.state.value}")

    if result.artifacts:
        for artifact in result.artifacts:
            if hasattr(artifact, "parts"):
                for part in artifact.parts:
                    if hasattr(part, "text"):
                        # 只打印报告的前500字符
                        print(f"\n📄 响应内容:\n{part.text[:500]}...")
                        break

    return result.state.value == "completed"


async def main():
    """运行所有测试"""
    print("\n🚀 开始测试 StockAgent 市场行情功能...\n")

    # 测试1: 分类器
    classifier_ok = await test_task_classifier()

    # 测试2: 市场概览
    market_ok = await test_market_overview()

    # 测试3: 个股分析（回归测试）
    # stock_ok = await test_stock_analysis()  # 可选，需要较长时间

    print("\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    print(f"分类器测试: {'✅ 通过' if classifier_ok else '❌ 失败'}")
    print(f"市场概览测试: {'✅ 通过' if market_ok else '❌ 失败'}")
    # print(f"个股分析测试: {'✅ 通过' if stock_ok else '❌ 失败'}")


if __name__ == "__main__":
    asyncio.run(main())
