#!/usr/bin/env python3
"""测试 LLM Fallback Router

测试输入: "帮我看看今天A股行情怎么样，顺便分析一下我持仓的基金008089"
预期: 规则未匹配，触发 LLM Router，识别为多步骤任务
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_llm_router():
    """测试 LLM Router 路由功能"""
    from src.agents.jarvis import JarvisAgent
    from src.config import qwen_config

    print("=" * 70)
    print("  LLM Fallback Router 测试")
    print("=" * 70)

    # 检查 Qwen API Key
    if not qwen_config.api_key:
        print("\n⚠️ QWEN_API_KEY 未配置，LLM Router 将不可用")
        print("请在 .env 文件中设置 QWEN_API_KEY")
    else:
        print(f"\n✅ Qwen API 已配置: {qwen_config.model}")

    # 创建 Jarvis Agent
    print("\n📦 初始化 Jarvis Agent...")
    agent = JarvisAgent(
        enable_mcp_search=False,  # 禁用 MCP 搜索以加速测试
        enable_llm_router=True,   # 启用 LLM Router
    )

    # 发现子 Agent（模拟，不实际连接）
    print("\n🔍 发现子 Agent...")
    try:
        await agent.discover_agents()
    except Exception as e:
        print(f"  ⚠️ Agent 发现失败（服务未启动）: {e}")
        print("  📝 将使用模拟 Agent 信息进行测试...")

        # 注册模拟 Agent 用于测试
        from src.a2a import AgentCard, AgentSkill
        from datetime import datetime

        mock_agents = [
            AgentCard(
                name="stock_agent",
                description="股票分析 Agent - 提供股票行情分析、技术指标计算、趋势预测",
                url="http://localhost:8001",
                skills=[
                    AgentSkill(
                        id="stock_analysis",
                        name="股票分析",
                        description="分析股票行情、计算技术指标（MACD/KDJ/RSI）、生成分析报告",
                        tags=["股票", "技术分析", "行情"],
                        examples=["分析股票600519", "贵州茅台走势如何"],
                    ),
                ],
            ),
            AgentCard(
                name="fund_agent",
                description="基金分析 Agent - 提供基金分析、持仓管理、投资建议",
                url="http://localhost:8003",
                skills=[
                    AgentSkill(
                        id="fund_analysis",
                        name="基金分析",
                        description="分析基金净值、收益率、风险指标，生成分析报告",
                        tags=["基金", "净值", "收益"],
                        examples=["分析基金008089", "基金008089怎么样"],
                    ),
                    AgentSkill(
                        id="fund_position",
                        name="持仓管理",
                        description="管理基金持仓，支持买入、卖出、查询持仓",
                        tags=["持仓", "买入", "卖出"],
                        examples=["买入基金008089 1000元", "查看我的持仓"],
                    ),
                ],
            ),
            AgentCard(
                name="email_agent",
                description="邮件发送 Agent - 发送邮件通知和报告",
                url="http://localhost:8002",
                skills=[
                    AgentSkill(
                        id="send_email",
                        name="发送邮件",
                        description="发送邮件到指定邮箱",
                        tags=["邮件", "通知"],
                        examples=["发送邮件到test@example.com"],
                    ),
                ],
            ),
        ]

        for card in mock_agents:
            agent._registry.register_local(card, card.url)
            print(f"  ✅ 注册模拟 Agent: {card.name}")

    # 测试输入 - 多个测试用例
    test_cases = [
        # 用例1: 规则可匹配（动态拆分）
        "帮我看看今天A股行情怎么样，顺便分析一下我持仓的基金008089",
        # 用例2: 规则无法匹配，需要 LLM 路由
        "最近有什么值得关注的投资机会吗",
        # 用例3: 规则无法匹配，但 LLM 可以识别多步骤
        "帮我研究一下最近表现好的科技股，整理成报告发到我邮箱 test@example.com",
        # 用例4: 简单问候，LLM 应该直接回答
        "你能帮我做什么",
    ]

    for i, test_input in enumerate(test_cases):
        print("\n" + "=" * 70)
        print(f"📝 测试用例 {i+1}: {test_input}")
        print("=" * 70)

        # 测试规则解析器
        print("\n🔄 规则解析器分析...")
        agents = agent._registry.list_agents()
        parse_result = await agent._rule_parser.parse(test_input, agents=agents)

        print("\n📊 解析结果:")
        print(f"  - need_agent: {parse_result.need_agent}")
        print(f"  - intent_type: {parse_result.intent_type.value}")
        print(f"  - match_type: {parse_result.match_type}")
        print(f"  - agent_name: {parse_result.agent_name}")

        if parse_result.match_type == "llm":
            print(f"  - llm_confidence: {parse_result.llm_confidence}")
            if parse_result.llm_reasoning:
                print(f"  - llm_reasoning: {parse_result.llm_reasoning[:200]}...")

        if parse_result.steps:
            print(f"\n📋 任务步骤 ({len(parse_result.steps)} 个):")
            for step in parse_result.steps:
                print(f"  步骤 {step.step}:")
                print(f"    - agent: {step.agent}")
                print(f"    - intent_type: {step.intent_type.value}")
                print(f"    - task_description: {step.task_description[:50]}...")
                print(f"    - depends_on_previous: {step.depends_on_previous}")
                if step.stock_code:
                    print(f"    - stock_code: {step.stock_code}")
                if step.fund_code:
                    print(f"    - fund_code: {step.fund_code}")

        if parse_result.fallback_response:
            print(f"\n💬 直接回答: {parse_result.fallback_response[:300]}...")

        # 输出完整 intent 字典
        print("\n📦 完整 Intent 字典:")
        intent = parse_result.to_dict()
        import json
        print(json.dumps(intent, ensure_ascii=False, indent=2))

    # 关闭资源
    await agent.close()

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_llm_router())
