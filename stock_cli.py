#!/usr/bin/env python3
"""股票分析交互式入口"""
import asyncio
from src.skills import StockAnalysisSkill


def print_banner():
    print("\n" + "=" * 70)
    print("  📈 股票技术分析工具 - 专业版")
    print("=" * 70)
    print("  数据来源: AKShare (新浪财经/东方财富)")
    print("  分析功能: MACD / KDJ / RSI / 布林带 / K线形态 / 支撑压力位")
    print("-" * 70)
    print("  命令格式: <股票代码> [分析类型] [天数]")
    print("  示例: 000001          - 平安银行完整分析")
    print("        600519 macd    - 茅台MACD分析")
    print("        002594 kdj 30  - 比亚迪30日KDJ分析")
    print("-" * 70)
    print("  /help  查看帮助  |  /quit  退出程序")
    print("=" * 70 + "\n")


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                          使用说明                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  基本用法:                                                             ║
║    <股票代码>                    - 完整技术分析 (默认60日)                ║
║    <股票代码> <分析类型>          - 指定分析类型                          ║
║    <股票代码> <分析类型> <天数>    - 指定分析类型和数据周期                 ║
║                                                                        ║
║  分析类型:                                                             ║
║    full    - 完整分析报告 (默认)                                        ║
║    macd    - MACD 指标深度分析                                          ║
║    kdj     - KDJ 指标深度分析                                           ║
║    trend   - 趋势与均线分析                                             ║
║    pattern - K线形态识别                                                ║
║                                                                        ║
║  报告内容:                                                             ║
║    • 综合评分 (0-100分) 及投资评级                                       ║
║    • 基本行情数据 (价格/涨跌幅/成交量)                                    ║
║    • 趋势分析 (均线排列/多空判断)                                        ║
║    • MACD分析 (金叉死叉/背离检测/柱状图)                                  ║
║    • KDJ分析 (超买超卖/钝化判断)                                         ║
║    • RSI分析 (强弱指标)                                                 ║
║    • 布林带分析 (通道位置/带宽变化)                                       ║
║    • 量能分析 (量比/量价关系)                                            ║
║    • K线形态识别 (早晨之星/黄昏之星/红三兵等)                              ║
║    • 支撑压力位 (20日高低点/枢轴点)                                       ║
║    • 投资建议 (多维度综合建议)                                           ║
║                                                                        ║
║  常用股票:                                                             ║
║    000001 平安银行  |  600519 贵州茅台  |  000858 五粮液                  ║
║    002594 比亚迪    |  600036 招商银行  |  000002 万科A                   ║
║    601318 中国平安  |  600900 长江电力  |  002415 海康威视                ║
╚══════════════════════════════════════════════════════════════════════╝
""")


async def analyze_stock(skill: StockAnalysisSkill, symbol: str, analysis_type: str, days: int):
    """执行股票分析"""
    print(f"\n⏳ 正在获取 {symbol} 的数据并分析...")
    print(f"   分析类型: {analysis_type}  |  数据周期: {days}天")
    print("-" * 70)

    result = await skill.execute(
        symbol=symbol,
        days=days,
        analysis_type=analysis_type
    )

    if result.success:
        print(result.result)
    else:
        print(f"\n❌ 分析失败: {result.error}")
        print("\n可能的原因:")
        print("  1. 网络连接问题 - 请检查网络后重试")
        print("  2. 股票代码错误 - 请输入正确的6位股票代码")
        print("  3. 数据源限制 - 请稍后再试")

    print()


async def main():
    print_banner()
    skill = StockAnalysisSkill()

    while True:
        try:
            user_input = input("📊 请输入股票代码: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("\n👋 感谢使用，再见!")
                break

            if user_input.lower() == "/help":
                print_help()
                continue

            # 解析输入
            parts = user_input.split()
            symbol = parts[0]
            analysis_type = parts[1] if len(parts) > 1 else "full"
            days = int(parts[2]) if len(parts) > 2 else 60

            # 验证股票代码
            if not symbol.isdigit() or len(symbol) != 6:
                print("❌ 请输入6位数字的股票代码，如: 000001\n")
                continue

            # 验证分析类型
            valid_types = ["full", "macd", "kdj", "trend", "pattern", "ma"]
            if analysis_type not in valid_types:
                print(f"❌ 无效的分析类型，可选: {', '.join(valid_types)}\n")
                continue

            await analyze_stock(skill, symbol, analysis_type, days)

        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见!")
            break
        except ValueError as e:
            print(f"❌ 输入错误: {e}\n")
        except Exception as e:
            print(f"❌ 发生错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
