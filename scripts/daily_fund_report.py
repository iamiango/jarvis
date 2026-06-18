"""每日基金综合报告生成脚本

从 .env 读取 FUND_CODES 配置，生成综合分析报告并发送邮件。
支持 PostgreSQL checkpoint 持久化任务状态。
"""
import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agents.jarvis import JarvisAgent
from src.skills.stock_retriever import StockRetrieverSkill
from src.a2a import A2AClient, TaskState, Task, Message
from src.config import postgres_config
from src.storage import CheckpointManager


# 基金名称缓存
FUND_NAMES: Dict[str, str] = {}


async def get_fund_names(fund_codes: List[str]) -> Dict[str, str]:
    """预先获取所有基金名称"""
    import akshare as ak

    names = {}
    for code in fund_codes:
        try:
            # 使用 AKShare 获取基金名称
            df = ak.fund_individual_basic_info_xq(symbol=code)
            if df is not None and not df.empty:
                # 查找基金名称行
                for idx, row in df.iterrows():
                    item = str(row['item']) if 'item' in df.columns else ''
                    value = str(row['value']) if 'value' in df.columns else ''
                    # 优先使用基金名称（简称），其次是基金全称
                    if item == '基金名称':
                        names[code] = value
                        break
                    elif item == '基金全称' and code not in names:
                        names[code] = value
        except Exception as e:
            print(f"获取基金 {code} 名称失败: {e}")

        if code not in names:
            names[code] = code

    return names


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
            # 中国股市惯例：红色表示上涨，绿色表示下跌
            color = "🔴" if data["pct_change"] >= 0 else "🟢"
            lines.append(
                f"| {color} {name} | {data['close']} | {change_symbol}{data['change']} | {pct_symbol}{data['pct_change']:.2f}% |"
            )

    lines.append("")
    return "\n".join(lines)


async def analyze_single_fund(jarvis: JarvisAgent, code: str, fund_name: str, parent_task_id: str) -> Dict[str, str]:
    """分析单只基金

    Args:
        jarvis: JarvisAgent 实例
        code: 基金代码
        fund_name: 基金名称
        parent_task_id: 父任务 ID（用于 checkpoint 记录）
    """
    try:
        print(f"正在分析基金 {code} ({fund_name})...", flush=True)
        task = await jarvis.delegate_task("fund_agent", f"分析基金 {code}", task_id=parent_task_id)

        if task.status.state == TaskState.COMPLETED:
            # 提取报告内容
            for msg in reversed(task.history):
                if msg.role == "agent":
                    for part in msg.parts:
                        if hasattr(part, "text") and part.text:
                            print(f"  ✅ 基金 {code} ({fund_name}) 分析完成", flush=True)
                            return {
                                "name": fund_name,
                                "report": part.text
                            }
            return {
                "name": fund_name,
                "report": f"基金 {code} 分析完成，但未能获取报告内容"
            }
        else:
            print(f"  ❌ 基金 {code} ({fund_name}) 分析失败", flush=True)
            return {
                "name": fund_name,
                "report": f"基金 {code} 分析失败: {task.status.message}"
            }
    except Exception as e:
        print(f"  ❌ 基金 {code} ({fund_name}) 分析出错: {e}", flush=True)
        return {
            "name": fund_name,
            "report": f"基金 {code} 分析出错: {str(e)}"
        }


async def analyze_funds(jarvis: JarvisAgent, fund_codes: List[str], fund_names: Dict[str, str], parent_task_id: str) -> Dict[str, Dict[str, str]]:
    """串行分析多只基金（因为本地 LLM 不支持真正的并行处理）

    Args:
        jarvis: JarvisAgent 实例
        fund_codes: 基金代码列表
        fund_names: 预先获取的基金名称字典
        parent_task_id: 父任务 ID
    """
    reports = {}

    for code in fund_codes:
        fund_name = fund_names.get(code, code)
        result = await analyze_single_fund(jarvis, code, fund_name, parent_task_id)
        reports[code] = result

    return reports


def generate_comprehensive_report(index_summary: str, fund_reports: Dict[str, Dict[str, str]]) -> str:
    """生成综合报告"""
    report_date = datetime.now().strftime("%Y年%m月%d日")

    lines = [
        f"# 每日基金综合分析报告",
        "",
        f"**报告日期**: {report_date}",
        "",
        "---",
        "",
        index_summary,
        "---",
        "",
        "## 基金列表",
        "",
        "| 序号 | 基金代码 | 基金名称 |",
        "|------|----------|----------|",
    ]

    # 添加基金列表
    for i, (code, data) in enumerate(fund_reports.items(), 1):
        lines.append(f"| {i} | {code} | {data['name']} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 基金分析报告")
    lines.append("")

    # 添加每个基金的详细报告
    for i, (code, data) in enumerate(fund_reports.items(), 1):
        lines.append(f"### {i}. {data['name']} ({code})")
        lines.append("")
        lines.append(data['report'])
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("")
    lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("*本报告由 Jarvis 智能助手自动生成，仅供参考，不构成投资建议。*")

    return "\n".join(lines)


async def send_report_email(jarvis: JarvisAgent, report: str, fund_codes: List[str], parent_task_id: str) -> bool:
    """发送报告邮件

    Args:
        jarvis: JarvisAgent 实例
        report: 报告内容
        fund_codes: 基金代码列表
        parent_task_id: 父任务 ID
    """
    default_to = os.getenv("SMTP_DEFAULT_TO", "")
    if not default_to:
        print("❌ 未配置默认收件人邮箱 (SMTP_DEFAULT_TO)")
        return False

    report_date = datetime.now().strftime("%Y-%m-%d")

    email_message = f"""发送邮件到 {default_to}
主题: '每日基金综合分析报告 - {report_date}'
内容: '{report}'"""

    try:
        task = await jarvis.delegate_task("email_agent", email_message, task_id=parent_task_id)

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
    print("每日基金综合分析报告生成")
    print("=" * 60)
    print()

    # 1. 读取基金代码配置
    fund_codes_str = os.getenv("FUND_CODES", "")
    if not fund_codes_str:
        print("❌ 未配置 FUND_CODES 环境变量")
        return

    fund_codes = [code.strip() for code in fund_codes_str.split(",") if code.strip()]
    print(f"📋 待分析基金: {', '.join(fund_codes)}")
    print()

    # 2. 初始化 CheckpointManager（用于持久化任务状态）
    checkpoint_manager: Optional[CheckpointManager] = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
        await checkpoint_manager.initialize()
        print("✅ PostgreSQL checkpoint 已连接")
    except Exception as e:
        print(f"⚠️ PostgreSQL checkpoint 初始化失败: {e}")
        print("   任务状态将不会持久化到数据库")
    print()

    # 3. 初始化 Jarvis（传入 checkpoint_manager）
    jarvis = JarvisAgent(checkpoint_manager=checkpoint_manager)

    try:
        # 4. 发现子 Agent
        print("🔍 发现子 Agent...")
        await jarvis.discover_agents()
        print()

        # 5. 检查必要的 Agent 是否就绪
        if "fund_agent" not in jarvis._registry:
            print("❌ Fund Agent 未就绪，请先启动: python -m src.agents.fund_agent --port 8003")
            return

        if "email_agent" not in jarvis._registry:
            print("❌ Email Agent 未就绪，请先启动: python -m src.agents.email_agent --port 8002")
            return

        # 6. 创建顶层任务（用于 checkpoint 记录）
        task_id = str(uuid.uuid4())
        report_date = datetime.now().strftime("%Y-%m-%d")
        if checkpoint_manager:
            # 创建 session
            session_id = await checkpoint_manager.create_session(user_id="daily_report_scheduler")
            # 创建任务
            main_task = Task(id=task_id, session_id=session_id)
            main_task.add_message(Message.user_text(f"每日基金综合分析报告 - {report_date}"))
            main_task.set_state(TaskState.WORKING, "开始生成报告")
            await checkpoint_manager.save_task(main_task)
            print(f"📋 任务已创建: {task_id}")

        # 7. 预先获取基金名称
        print("📋 获取基金名称...")
        fund_names = await get_fund_names(fund_codes)
        for code, name in fund_names.items():
            print(f"  {code}: {name}")
        print()

        # 8. 获取大盘指数
        print("📊 获取大盘指数数据...")
        index_summary = await get_index_summary()
        print(index_summary)
        print()

        # 9. 分析各基金
        print("📈 开始分析基金...")
        fund_reports = await analyze_funds(jarvis, fund_codes, fund_names, task_id)
        print()

        # 10. 生成综合报告
        print("📝 生成综合报告...")
        comprehensive_report = generate_comprehensive_report(index_summary, fund_reports)
        print("✅ 综合报告生成完成")
        print()

        # 11. 发送邮件
        print("📧 发送报告邮件...")
        await send_report_email(jarvis, comprehensive_report, fund_codes, task_id)
        print()

        # 12. 更新任务状态为完成
        if checkpoint_manager:
            main_task.set_state(TaskState.COMPLETED, "报告生成并发送完成")
            main_task.add_message(Message.agent_text(f"报告已发送，包含 {len(fund_codes)} 只基金"))
            await checkpoint_manager.save_task(main_task)
            print(f"✅ 任务完成: {task_id}")

        # 13. 打印报告摘要
        print("=" * 60)
        print("报告内容预览:")
        print("=" * 60)
        # 只打印前 3000 个字符
        print(comprehensive_report[:3000])
        if len(comprehensive_report) > 3000:
            print(f"\n... (共 {len(comprehensive_report)} 字符)")

    finally:
        await jarvis.close()
        # 关闭 checkpoint_manager
        if checkpoint_manager:
            await checkpoint_manager.close()
            print("✅ PostgreSQL checkpoint 已关闭")


if __name__ == "__main__":
    asyncio.run(main())
