#!/usr/bin/env python3
"""测试规则解析器"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.router import RuleParser, IntentType


def test_router():
    """测试路由解析器"""
    parser = RuleParser()

    print("=" * 60)
    print("🧪 规则解析器测试")
    print("=" * 60)

    test_cases = [
        # 1. 单一意图 - 股票分析
        ("分析股票002594", "single", IntentType.STOCK_ANALYSIS),
        ("帮我看看比亚迪的行情", "single", IntentType.STOCK_ANALYSIS),
        ("股票600519怎么样", "single", IntentType.STOCK_ANALYSIS),

        # 2. 单一意图 - 基金分析
        ("分析基金110011", "single", IntentType.FUND_ANALYSIS),
        ("基金519001表现如何", "single", IntentType.FUND_ANALYSIS),

        # 3. 单一意图 - 邮件
        ("发邮件给test@example.com", "single", IntentType.EMAIL),
        ("给admin@company.com发送报告", "single", IntentType.EMAIL),

        # 4. 动态拆分 - 复合任务
        ("分析股票002594，然后发邮件给test@example.com", "dynamic", IntentType.COMPOUND),
        ("先看看比亚迪股票，再发给我", "dynamic", IntentType.COMPOUND),
        ("分析基金110011并发送邮件", "dynamic", IntentType.COMPOUND),

        # 5. 固定模板匹配（如果配置了）
        ("分析股票002594并发送报告到test@example.com", "template", IntentType.COMPOUND),

        # 6. 兜底 - 直接回答
        ("你好", "single", IntentType.DIRECT_ANSWER),  # 匹配 direct_answer 关键词
        ("今天天气怎么样", "fallback", IntentType.DIRECT_ANSWER),
    ]

    passed = 0
    failed = 0

    for text, expected_match_type, expected_intent in test_cases:
        result = parser.parse(text)

        # 检查意图类型
        intent_ok = result.intent_type == expected_intent
        # 检查匹配类型（模板匹配可能会退化为动态或单一）
        match_ok = (
            result.match_type == expected_match_type or
            (expected_match_type == "template" and result.match_type in ["template", "dynamic", "single"])
        )

        status = "✅" if (intent_ok and match_ok) else "❌"
        if intent_ok and match_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} 输入: {text}")
        print(f"   期望: match_type={expected_match_type}, intent={expected_intent.value}")
        print(f"   实际: match_type={result.match_type}, intent={result.intent_type.value}")

        if result.steps:
            print(f"   步骤数: {len(result.steps)}")
            for step in result.steps:
                print(f"     - Step {step.step}: {step.agent} -> {step.task_description[:40]}...")

        if result.stock_code:
            print(f"   股票代码: {result.stock_code}")
        if result.fund_code:
            print(f"   基金代码: {result.fund_code}")
        if result.email_address:
            print(f"   邮箱地址: {result.email_address}")

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_router()
    sys.exit(0 if success else 1)
