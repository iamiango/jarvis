#!/usr/bin/env python3
"""测试 Jarvis 处理市场行情查询"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_jarvis_market_query():
    """测试 Jarvis 处理 '帮我看看今天A股行情怎么样'"""
    from src.agents.jarvis import JarvisAgent
    from src.a2a import AgentCard, AgentSkill

    print("=" * 70)
    print("  Jarvis 市场行情查询测试")
    print("=" * 70)

    # 创建 Jarvis Agent
    print("\n📦 初始化 Jarvis Agent...")
    agent = JarvisAgent(
        enable_mcp_search=False,
        enable_llm_router=True,
    )

    # 注册模拟的 stock_agent（因为实际服务未启动）
    print("\n🔧 注册模拟 Stock Agent...")
    mock_stock_agent = AgentCard(
        name="stock_agent",
        description="股票分析 Agent - 提供 A 股数据获取、技术指标计算和专业投资分析报告，支持市场行情概览",
        url="http://localhost:8001",
        skills=[
            AgentSkill(
                id="stock_retriever",
                name="股票数据获取",
                description="获取 A 股历史数据并计算技术指标",
                tags=["股票", "数据", "技术指标"],
                examples=["获取股票 000001 的数据"],
            ),
            AgentSkill(
                id="market_overview",
                name="市场行情概览",
                description="获取 A 股大盘指数、市场整体行情和最新市场新闻",
                tags=["大盘", "指数", "市场", "行情", "A股"],
                examples=["今天A股行情怎么样", "大盘走势如何"],
            ),
        ],
    )
    agent._registry.register_local(mock_stock_agent, mock_stock_agent.url)

    # 测试输入
    test_input = "帮我看看今天A股行情怎么样"

    print(f"\n📝 测试输入: '{test_input}'")
    print("-" * 50)

    # 获取 Agent 列表
    agents = agent._registry.list_agents()
    print(f"\n📋 已注册 Agent: {[a.card.name for a in agents]}")

    # 使用规则解析器分析意图
    print("\n🔄 规则解析器分析...")
    parse_result = await agent._rule_parser.parse(test_input, agents=agents)

    print("\n📊 解析结果:")
    print(f"  - need_agent: {parse_result.need_agent}")
    print(f"  - intent_type: {parse_result.intent_type.value}")
    print(f"  - match_type: {parse_result.match_type}")
    print(f"  - agent_name: {parse_result.agent_name}")

    if parse_result.steps:
        print(f"\n📋 任务步骤 ({len(parse_result.steps)} 个):")
        for step in parse_result.steps:
            print(f"  步骤 {step.step}:")
            print(f"    - agent: {step.agent}")
            print(f"    - intent_type: {step.intent_type.value}")
            print(f"    - task_description: {step.task_description[:80]}...")

    if parse_result.llm_reasoning:
        print(f"\n🤖 LLM 推理: {parse_result.llm_reasoning[:200]}...")

    if parse_result.fallback_response:
        print(f"\n💬 直接回答: {parse_result.fallback_response[:300]}...")

    # 关闭资源
    await agent.close()

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)

    return parse_result


if __name__ == "__main__":
    asyncio.run(test_jarvis_market_query())
