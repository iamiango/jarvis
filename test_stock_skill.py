#!/usr/bin/env python3
"""测试股票分析 Skill"""
import asyncio
from src.skills import StockAnalysisSkill


async def test_stock_analysis():
    """测试股票分析功能"""
    skill = StockAnalysisSkill()

    print("=" * 60)
    print("🧪 测试股票分析 Skill")
    print("=" * 60)

    # 测试 1: 平安银行完整分析
    print("\n📊 测试 1: 平安银行 (000001) 完整分析")
    print("-" * 50)
    result = await skill.execute(
        symbol="000001",
        days=60,
        analysis_type="full"
    )
    if result.success:
        print(result.result)
    else:
        print(f"❌ 失败: {result.error}")

    # 测试 2: 贵州茅台 MACD 分析
    print("\n📊 测试 2: 贵州茅台 (600519) MACD 分析")
    print("-" * 50)
    result = await skill.execute(
        symbol="600519",
        days=30,
        analysis_type="macd"
    )
    if result.success:
        print(result.result)
    else:
        print(f"❌ 失败: {result.error}")

    # 测试 3: 比亚迪 K线形态
    print("\n📊 测试 3: 比亚迪 (002594) K线形态分析")
    print("-" * 50)
    result = await skill.execute(
        symbol="002594",
        days=20,
        analysis_type="pattern"
    )
    if result.success:
        print(result.result)
    else:
        print(f"❌ 失败: {result.error}")

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_stock_analysis())
