"""每日综合股票报告生成脚本

从 .env 读取 STOCK_CODES 配置，生成综合分析报告并发送邮件。
"""
import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agents.jarvis import JarvisAgent
from src.skills.stock_retriever import StockRetrieverSkill
from src.a2a import A2AClient, TaskState


async def get_index_summary() -> str:
    """获取大盘指数摘要"""
    skill = StockRetrieverSkill()
    index_data = await skill.get_index_data()

    lines = [
        "## 大盘指数行情",
        "",
        f"**日期**: {index_data['date']}",
        "",
        "| 指数 | 最新价 | 涨跌 | 涨跌幅 |",
        "|------|--------|------|--------|",
    ]

    for name, data in index_data.get("indices", {}).items():
        if "error" in data:
            lines.append(f"| {name} | 获取失败 | - | - |")
        else:
            change_symbol = "+" if data["change"] >= 0 else ""
            pct_symbol = "+" if data["pct_change"] >= 0 else ""
            color = "🟢" if data["pct_change"] >= 0 else "🔴"
            lines.append(
                f"| {color} {name} | {data['close']} | {change_symbol}{data['change']} | {pct_symbol}{data['pct_change']:.2f}% |"
            )

    lines.append("")
    return "\n".join(lines)


async def analyze_stocks(jarvis: JarvisAgent, stock_codes: List[str]) -> Dict[str, str]:
    """分析多只股票"""
    reports = {}

    for code in stock_codes:
        print(f"正在分析股票 {code}...")
        try:
            task = await jarvis.delegate_task("stock_agent", f"分析股票 {code}")

            if task.status.state == TaskState.COMPLETED:
                # 提取报告内容
                for msg in reversed(task.history):
                    if msg.role == "agent":
                        for part in msg.parts:
                            if hasattr(part, "text") and part.text:
                                reports[code] = part.text
                                print(f"  ✅ 股票 {code} 分析完成")
                                break
                        if code in reports:
                            break

                if code not in reports:
                    reports[code] = f"股票 {code} 分析完成，但未能获取报告内容"
            else:
                reports[code] = f"股票 {code} 分析失败: {task.status.message}"
                print(f"  ❌ 股票 {code} 分析失败")
        except Exception as e:
            reports[code] = f"股票 {code} 分析出错: {str(e)}"
            print(f"  ❌ 股票 {code} 分析出错: {e}")

    return reports


def generate_comprehensive_report(index_summary: str, stock_reports: Dict[str, str]) -> str:
    """生成综合报告"""
    report_date = datetime.now().strftime("%Y年%m月%d日")

    lines = [
        f"# 每日股票综合分析报告",
        "",
        f"**报告日期**: {report_date}",
        "",
        "---",
        "",
        index_summary,
        "---",
        "",
        "## 个股分析报告",
        "",
    ]

    for i, (code, report) in enumerate(stock_reports.items(), 1):
        lines.append(f"### {i}. 股票代码: {code}")
        lines.append("")
        lines.append(report)
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("")
    lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("*本报告由 Jarvis 智能助手自动生成，仅供参考，不构成投资建议。*")

    return "\n".join(lines)


async def send_report_email(jarvis: JarvisAgent, report: str, stock_codes: List[str]) -> bool:
    """发送报告邮件"""
    default_to = os.getenv("SMTP_DEFAULT_TO", "")
    if not default_to:
        print("❌ 未配置默认收件人邮箱 (SMTP_DEFAULT_TO)")
        return False

    stock_list = ", ".join(stock_codes)
    report_date = datetime.now().strftime("%Y-%m-%d")

    email_message = f"""发送邮件到 {default_to}
主题: '每日股票综合分析报告 - {report_date}'
内容: '{report}'"""

    try:
        task = await jarvis.delegate_task("email_agent", email_message)

        if task.status.state == TaskState.COMPLETED:
            print(f"✅ 报告已发送到 {default_to}")
            return True
        else:
            print(f"❌ 邮件发送失败: {task.status.message}")
            return False
    except Exception as e:
        print(f"❌ 邮件发送出错: {e}")
        return False


async def main():
    """主函数"""
    print("=" * 60)
    print("每日股票综合分析报告生成")
    print("=" * 60)
    print()

    # 1. 读取股票代码配置
    stock_codes_str = os.getenv("STOCK_CODES", "")
    if not stock_codes_str:
        print("❌ 未配置 STOCK_CODES 环境变量")
        return

    stock_codes = [code.strip() for code in stock_codes_str.split(",") if code.strip()]
    print(f"📋 待分析股票: {', '.join(stock_codes)}")
    print()

    # 2. 初始化 Jarvis
    jarvis = JarvisAgent()

    try:
        # 3. 发现子 Agent
        print("🔍 发现子 Agent...")
        await jarvis.discover_agents()
        print()

        # 4. 检查必要的 Agent 是否就绪
        if "stock_agent" not in jarvis._registry:
            print("❌ Stock Agent 未就绪，请先启动: python -m src.agents.stock_agent --port 8001")
            return

        if "email_agent" not in jarvis._registry:
            print("❌ Email Agent 未就绪，请先启动: python -m src.agents.email_agent --port 8002")
            return

        # 5. 获取大盘指数
        print("📊 获取大盘指数数据...")
        index_summary = await get_index_summary()
        print(index_summary)
        print()

        # 6. 分析各股票
        print("📈 开始分析个股...")
        stock_reports = await analyze_stocks(jarvis, stock_codes)
        print()

        # 7. 生成综合报告
        print("📝 生成综合报告...")
        comprehensive_report = generate_comprehensive_report(index_summary, stock_reports)
        print("✅ 综合报告生成完成")
        print()

        # 8. 发送邮件
        print("📧 发送报告邮件...")
        await send_report_email(jarvis, comprehensive_report, stock_codes)
        print()

        # 9. 打印报告摘要
        print("=" * 60)
        print("报告内容预览:")
        print("=" * 60)
        # 只打印前 2000 个字符
        print(comprehensive_report[:2000])
        if len(comprehensive_report) > 2000:
            print(f"\n... (共 {len(comprehensive_report)} 字符)")

    finally:
        await jarvis.close()


if __name__ == "__main__":
    asyncio.run(main())
