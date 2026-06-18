"""Jarvis 定时任务定义

定义所有可调度的任务函数。
"""
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Callable

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class JobRegistry:
    """任务注册表 - 管理所有可用的定时任务"""

    _jobs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, job_id: str, description: str = ""):
        """装饰器：注册任务函数"""
        def decorator(func: Callable):
            cls._jobs[job_id] = {
                "func": func,
                "description": description
            }
            return func
        return decorator

    @classmethod
    def get_job(cls, job_id: str) -> Callable:
        """获取任务函数"""
        job_info = cls._jobs.get(job_id)
        if job_info:
            return job_info["func"]
        raise ValueError(f"任务不存在: {job_id}")

    @classmethod
    def list_jobs(cls) -> List[Dict[str, str]]:
        """列出所有已注册的任务"""
        return [
            {"id": job_id, "description": info["description"]}
            for job_id, info in cls._jobs.items()
        ]


# ============================================================
# 定时任务定义
# ============================================================

@JobRegistry.register("daily_fund_report", "每日基金分析报告")
async def daily_fund_report_job():
    """每日基金分析报告任务"""
    from dotenv import load_dotenv
    load_dotenv()

    from src.agents.jarvis import JarvisAgent
    from src.skills.stock_retriever import StockRetrieverSkill
    from src.storage.checkpoint import CheckpointManager
    from src.config import postgres_config
    from src.a2a import TaskState

    logger.info("=" * 60)
    logger.info("📊 开始执行每日基金分析报告任务")
    logger.info(f"⏰ 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 读取基金代码配置
    fund_codes_str = os.getenv("FUND_CODES", "")
    if not fund_codes_str:
        logger.error("❌ 未配置 FUND_CODES 环境变量")
        return

    fund_codes = [code.strip() for code in fund_codes_str.split(",") if code.strip()]
    logger.info(f"📋 待分析基金: {', '.join(fund_codes)}")

    # 读取用户 ID 配置（用于获取持仓）
    user_id = os.getenv("REPORT_USER_ID", "default")
    logger.info(f"👤 用户 ID: {user_id}")

    # 初始化 CheckpointManager
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
        await checkpoint_manager.initialize()
        logger.info("✅ CheckpointManager 初始化成功")
    except Exception as e:
        logger.warning(f"⚠️ CheckpointManager 初始化失败: {e}，将不使用持仓信息")

    # 初始化 Jarvis
    jarvis = JarvisAgent(checkpoint_manager=checkpoint_manager)

    try:
        # 发现子 Agent
        logger.info("🔍 发现子 Agent...")
        await jarvis.discover_agents()

        # 检查必要的 Agent 是否就绪
        if "fund_agent" not in jarvis._registry:
            logger.error("❌ Fund Agent 未就绪")
            return

        if "email_agent" not in jarvis._registry:
            logger.error("❌ Email Agent 未就绪")
            return

        # 获取基金名称
        logger.info("📋 获取基金名称...")
        fund_names = await _get_fund_names(fund_codes)

        # 获取大盘指数
        logger.info("📊 获取大盘指数数据...")
        index_summary = await _get_index_summary()

        # 分析各基金
        logger.info("📈 开始分析基金...")
        fund_reports, report_generator = await _analyze_funds(jarvis, fund_codes, fund_names, user_id, checkpoint_manager)

        # 生成综合报告
        logger.info("📝 生成综合报告...")
        comprehensive_report = _generate_comprehensive_report(index_summary, fund_reports)

        # 发送邮件
        logger.info("📧 发送报告邮件...")
        await _send_report_email(jarvis, comprehensive_report, fund_codes)

        logger.info("✅ 每日基金分析报告任务执行完成")

        # 邮件发送后，关闭 MCP 资源
        logger.info("🔒 关闭 MCP 资源...")
        if report_generator:
            try:
                await report_generator.close()
            except Exception as e:
                logger.warning(f"关闭 MCP 资源时出错（可忽略）: {e}")

    except Exception as e:
        logger.error(f"❌ 任务执行失败: {e}")
        raise
    finally:
        await jarvis.close()
        if checkpoint_manager:
            await checkpoint_manager.close()


async def _get_fund_names(fund_codes: List[str]) -> Dict[str, str]:
    """预先获取所有基金名称"""
    import akshare as ak

    names = {}
    for code in fund_codes:
        try:
            df = ak.fund_individual_basic_info_xq(symbol=code)
            if df is not None and not df.empty:
                for idx, row in df.iterrows():
                    item = str(row['item']) if 'item' in df.columns else ''
                    value = str(row['value']) if 'value' in df.columns else ''
                    if item == '基金名称':
                        names[code] = value
                        break
                    elif item == '基金全称' and code not in names:
                        names[code] = value
        except Exception as e:
            logger.warning(f"获取基金 {code} 名称失败: {e}")

        if code not in names:
            names[code] = code

    return names


async def _get_index_summary() -> str:
    """获取大盘指数摘要"""
    from src.skills.stock_retriever import StockRetrieverSkill

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


async def _analyze_funds(
    jarvis,
    fund_codes: List[str],
    fund_names: Dict[str, str],
    user_id: str = "default",
    checkpoint_manager=None
) -> tuple[Dict[str, Dict[str, str]], any]:
    """分析多只基金

    直接复用 FundReportGeneratorSkill.execute() 方法，确保：
    - 获取用户偏好和持仓
    - 搜索最新新闻
    - 生成英文报告
    - 翻译成中文（含个性化持仓建议）

    所有逻辑与单独分析基金完全一致。

    Returns:
        (reports, report_generator) - 返回报告和 generator 以便调用方关闭 MCP
    """
    import json
    from src.skills.fund_retriever import FundRetrieverSkill
    from src.skills.fund_report import FundReportGeneratorSkill

    reports = {}
    retriever = FundRetrieverSkill()
    report_generator = FundReportGeneratorSkill(checkpoint_manager)

    for i, code in enumerate(fund_codes):
        fund_name = fund_names.get(code, code)
        logger.info(f"=" * 50)
        logger.info(f"[{i+1}/{len(fund_codes)}] 处理基金 {code} ({fund_name})")

        try:
            # 1. 获取基金数据
            logger.info(f"📊 [{code}] 获取基金数据...")
            data_result = await retriever.execute(fund_code=code)
            if not data_result.success:
                raise Exception(f"获取数据失败: {data_result.error}")

            # 2. 生成完整报告（复用 execute 方法）
            # execute() 内部会：获取偏好、获取持仓、搜索新闻、生成英文、翻译中文
            logger.info(f"📝 [{code}] 生成分析报告...")
            report_result = await report_generator.execute(
                fund_data=data_result.result,
                user_id=user_id,
                fund_code=code
            )

            if report_result.success:
                logger.info(f"✅ [{code}] ({fund_name}) 分析完成")
                reports[code] = {
                    "name": fund_name,
                    "report": report_result.result
                }
            else:
                raise Exception(report_result.error)

        except Exception as e:
            logger.error(f"❌ [{code}] 处理失败: {e}")
            reports[code] = {
                "name": fund_name,
                "report": f"基金 {code} 分析出错: {str(e)}"
            }

    # 返回 report_generator 以便调用方在邮件发送后关闭
    # 否则会导致 asyncio cancel scope 错误

    return reports, report_generator


def _generate_comprehensive_report(index_summary: str, fund_reports: Dict[str, Dict[str, str]]) -> str:
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

    for i, (code, data) in enumerate(fund_reports.items(), 1):
        lines.append(f"| {i} | {code} | {data['name']} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 基金分析报告")
    lines.append("")

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


async def _send_report_email(jarvis, report: str, fund_codes: List[str]) -> bool:
    """发送报告邮件"""
    from src.a2a import TaskState

    default_to = os.getenv("SMTP_DEFAULT_TO", "")
    if not default_to:
        logger.error("❌ 未配置默认收件人邮箱 (SMTP_DEFAULT_TO)")
        return False

    report_date = datetime.now().strftime("%Y-%m-%d")

    email_message = f"""发送邮件到 {default_to}
主题: '每日基金综合分析报告 - {report_date}'
内容: '{report}'"""

    try:
        task = await jarvis.delegate_task("email_agent", email_message)

        if task.status.state == TaskState.COMPLETED:
            logger.info(f"✅ 报告已发送到 {default_to}")
            return True
        else:
            logger.error(f"❌ 邮件发送失败: {task.status.message}")
            return False
    except Exception as e:
        logger.error(f"❌ 邮件发送出错: {e}")
        return False


# ============================================================
# 每日股票分析报告任务
# ============================================================

@JobRegistry.register("daily_stock_report", "每日股票分析报告")
async def daily_stock_report_job():
    """每日股票分析报告任务"""
    from dotenv import load_dotenv
    load_dotenv()

    from src.agents.jarvis import JarvisAgent
    from src.skills.stock_retriever import StockRetrieverSkill
    from src.a2a import TaskState

    logger.info("=" * 60)
    logger.info("📈 开始执行每日股票分析报告任务")
    logger.info(f"⏰ 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 读取股票代码配置
    stock_codes_str = os.getenv("STOCK_CODES", "")
    if not stock_codes_str:
        logger.error("❌ 未配置 STOCK_CODES 环境变量")
        return

    stock_codes = [code.strip() for code in stock_codes_str.split(",") if code.strip()]
    logger.info(f"📋 待分析股票: {', '.join(stock_codes)}")

    # 初始化 Jarvis
    jarvis = JarvisAgent()

    try:
        # 发现子 Agent
        logger.info("🔍 发现子 Agent...")
        await jarvis.discover_agents()

        # 检查必要的 Agent 是否就绪
        if "stock_agent" not in jarvis._registry:
            logger.error("❌ Stock Agent 未就绪")
            return

        if "email_agent" not in jarvis._registry:
            logger.error("❌ Email Agent 未就绪")
            return

        # 获取股票名称
        logger.info("📋 获取股票名称...")
        stock_names = await _get_stock_names(stock_codes)

        # 获取大盘指数
        logger.info("📊 获取大盘指数数据...")
        index_summary = await _get_index_summary()

        # 分析各股票
        logger.info("📈 开始分析股票...")
        stock_reports = await _analyze_stocks(stock_codes, stock_names)

        # 生成综合报告
        logger.info("📝 生成综合报告...")
        comprehensive_report = _generate_stock_comprehensive_report(index_summary, stock_reports)

        # 发送邮件
        logger.info("📧 发送报告邮件...")
        await _send_stock_report_email(jarvis, comprehensive_report, stock_codes)

        logger.info("✅ 每日股票分析报告任务执行完成")

    except Exception as e:
        logger.error(f"❌ 任务执行失败: {e}")
        raise
    finally:
        await jarvis.close()


async def _get_stock_names(stock_codes: List[str]) -> Dict[str, str]:
    """预先获取所有股票名称"""
    import akshare as ak

    names = {}

    # 使用 stock_info_a_code_name 批量获取（更稳定）
    try:
        stock_list = ak.stock_info_a_code_name()
        if stock_list is not None and not stock_list.empty:
            for code in stock_codes:
                name_row = stock_list[stock_list['code'] == code]
                if not name_row.empty:
                    names[code] = name_row.iloc[0]['name']
                    logger.info(f"[股票名称] {code} -> {names[code]}")
    except Exception as e:
        logger.warning(f"批量获取股票名称失败: {e}")

    # 对于未获取到名称的股票，尝试备用方法
    for code in stock_codes:
        if code not in names:
            try:
                # 备用方法: stock_individual_info_em
                df = ak.stock_individual_info_em(symbol=code)
                if df is not None and not df.empty:
                    for idx, row in df.iterrows():
                        item = str(row['item']) if 'item' in df.columns else ''
                        value = str(row['value']) if 'value' in df.columns else ''
                        if item == '股票简称':
                            names[code] = value
                            break
            except Exception as e:
                logger.warning(f"获取股票 {code} 名称失败: {e}")

            if code not in names:
                names[code] = code

    return names


async def _analyze_stocks(stock_codes: List[str], stock_names: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """并行分析多只股票

    优化策略：
    - 使用本地 Ollama 生成英文报告（占用本地 GPU）
    - 使用远程 Qwen API 翻译（不占用本地资源）
    - 当股票 N 的英文报告生成完成后，同时启动：
      1. 股票 N 的翻译任务（远程 Qwen）
      2. 股票 N+1 的英文报告生成（本地 Ollama）
    - 这样翻译和下一个报告生成可以并行执行
    """
    import json
    from src.skills.stock_retriever import StockRetrieverSkill
    from src.skills.stock_report_generator import StockReportGeneratorSkill

    reports = {}
    retriever = StockRetrieverSkill()
    report_generator = StockReportGeneratorSkill()

    # 存储中间结果
    english_reports: Dict[str, tuple] = {}  # code -> (english_report, data)
    translation_tasks: Dict[str, asyncio.Task] = {}  # code -> translation task

    async def get_stock_data_and_generate_english(code: str, stock_name: str) -> tuple:
        """获取股票数据并生成英文报告"""
        logger.info(f"📊 [{code}] 获取股票数据...")
        data_result = await retriever.execute(symbol=code)
        if not data_result.success:
            raise Exception(f"获取数据失败: {data_result.error}")

        data = json.loads(data_result.result)
        logger.info(f"🤖 [{code}] 生成英文分析报告 (本地 Ollama)...")
        english_report = await report_generator.generate_english_report(data)
        return english_report, data

    async def translate_report(code: str, stock_name: str, english_report: str, data: dict) -> str:
        """翻译报告为中文"""
        logger.info(f"🔄 [{code}] 翻译为中文 (远程 Qwen API)...")
        chinese_report = await report_generator.translate_to_chinese(english_report, data)
        return chinese_report

    # 流水线处理
    for i, code in enumerate(stock_codes):
        stock_name = stock_names.get(code, code)
        logger.info(f"=" * 50)
        logger.info(f"[{i+1}/{len(stock_codes)}] 处理股票 {code} ({stock_name})")

        try:
            # 1. 生成当前股票的英文报告
            english_report, data = await get_stock_data_and_generate_english(code, stock_name)
            english_reports[code] = (english_report, data)

            # 2. 启动当前股票的翻译任务（异步，不等待）
            translation_tasks[code] = asyncio.create_task(
                translate_report(code, stock_name, english_report, data)
            )
            logger.info(f"⏳ [{code}] 翻译任务已启动（后台运行）")

        except Exception as e:
            logger.error(f"❌ [{code}] 处理失败: {e}")
            reports[code] = {
                "name": stock_name,
                "report": f"股票 {code} 分析出错: {str(e)}"
            }

    # 等待所有翻译任务完成
    logger.info("=" * 50)
    logger.info("⏳ 等待所有翻译任务完成...")

    for code in stock_codes:
        stock_name = stock_names.get(code, code)
        if code in translation_tasks:
            try:
                chinese_report = await translation_tasks[code]
                logger.info(f"✅ [{code}] ({stock_name}) 分析完成")
                reports[code] = {
                    "name": stock_name,
                    "report": chinese_report
                }
            except Exception as e:
                logger.error(f"❌ [{code}] 翻译失败: {e}")
                # 翻译失败时使用英文原文
                if code in english_reports:
                    english_report, _ = english_reports[code]
                    reports[code] = {
                        "name": stock_name,
                        "report": f"[翻译失败，以下为英文原文]\n\n{english_report}"
                    }
                else:
                    reports[code] = {
                        "name": stock_name,
                        "report": f"股票 {code} 翻译失败: {str(e)}"
                    }

    return reports


def _generate_stock_comprehensive_report(index_summary: str, stock_reports: Dict[str, Dict[str, str]]) -> str:
    """生成股票综合报告"""
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
        "## 股票列表",
        "",
        "| 序号 | 股票代码 | 股票名称 |",
        "|------|----------|----------|",
    ]

    for i, (code, data) in enumerate(stock_reports.items(), 1):
        lines.append(f"| {i} | {code} | {data['name']} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 股票分析报告")
    lines.append("")

    for i, (code, data) in enumerate(stock_reports.items(), 1):
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


async def _send_stock_report_email(jarvis, report: str, stock_codes: List[str]) -> bool:
    """发送股票报告邮件"""
    from src.a2a import TaskState

    default_to = os.getenv("SMTP_DEFAULT_TO", "")
    if not default_to:
        logger.error("❌ 未配置默认收件人邮箱 (SMTP_DEFAULT_TO)")
        return False

    report_date = datetime.now().strftime("%Y-%m-%d")

    email_message = f"""发送邮件到 {default_to}
主题: '每日股票综合分析报告 - {report_date}'
内容: '{report}'"""

    try:
        task = await jarvis.delegate_task("email_agent", email_message)

        if task.status.state == TaskState.COMPLETED:
            logger.info(f"✅ 报告已发送到 {default_to}")
            return True
        else:
            logger.error(f"❌ 邮件发送失败: {task.status.message}")
            return False
    except Exception as e:
        logger.error(f"❌ 邮件发送出错: {e}")
        return False


# ============================================================
# 盘中股票监控作业（每小时执行）
# ============================================================

@JobRegistry.register("hourly_stock_monitor", "盘中股票监控提醒")
async def hourly_stock_monitor_job():
    """盘中股票监控任务

    工作日 9:00-15:00 每小时执行一次：
    1. 读取 STOCK_CODES 中的股票
    2. 获取数据并用本地 Finance LLM 分析
    3. 结合持仓和偏好，调用远端 Qwen 获取操作建议
    4. 若建议为"买入"或"卖出"，发送邮件提醒
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.agents.jarvis import JarvisAgent
    from src.skills.stock_monitor import StockMonitorSkill, TradeAction
    from src.storage.checkpoint import CheckpointManager
    from src.config import postgres_config
    from src.a2a import TaskState

    current_hour = datetime.now().hour
    logger.info("=" * 60)
    logger.info(f"🔔 盘中股票监控 (Hour: {current_hour}:00)")
    logger.info(f"⏰ 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 检查是否在交易时间内 (9:00 - 15:00)
    if current_hour < 9 or current_hour >= 15:
        logger.info("⏸️ 当前不在交易时间 (9:00-15:00)，跳过监控")
        return

    # 读取股票代码配置
    stock_codes_str = os.getenv("STOCK_CODES", "")
    if not stock_codes_str:
        logger.error("❌ 未配置 STOCK_CODES 环境变量")
        return

    stock_codes = [code.strip() for code in stock_codes_str.split(",") if code.strip()]
    logger.info(f"📋 监控股票: {', '.join(stock_codes)}")

    # 读取用户 ID
    user_id = os.getenv("REPORT_USER_ID", "default")
    logger.info(f"👤 用户 ID: {user_id}")

    # 初始化 CheckpointManager
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
        await checkpoint_manager.initialize()
        logger.info("✅ CheckpointManager 初始化成功")
    except Exception as e:
        logger.warning(f"⚠️ CheckpointManager 初始化失败: {e}")

    # 初始化监控技能
    monitor_skill = StockMonitorSkill(checkpoint_manager=checkpoint_manager)

    # 初始化 Jarvis（用于发送邮件）
    jarvis = None
    email_agent_ready = False

    try:
        jarvis = JarvisAgent(checkpoint_manager=checkpoint_manager)
        await jarvis.discover_agents()
        email_agent_ready = "email_agent" in jarvis._registry
        if not email_agent_ready:
            logger.warning("⚠️ Email Agent 未就绪，将无法发送邮件提醒")
    except Exception as e:
        logger.warning(f"⚠️ Jarvis 初始化失败: {e}")

    # 收集需要发送邮件的建议
    alerts_to_send: List[Dict[str, Any]] = []
    log_ids: Dict[str, int] = {}  # stock_code -> log_id

    # 逐个分析股票
    for i, code in enumerate(stock_codes):
        logger.info(f"-" * 40)
        logger.info(f"[{i+1}/{len(stock_codes)}] 分析股票 {code}")

        try:
            result = await monitor_skill.execute(stock_code=code, user_id=user_id)

            if result.success and result.result:
                advice_data = result.result
                action = advice_data.get("action", "持有")

                logger.info(f"📊 [{code}] {advice_data.get('stock_name', code)}: {action}")
                logger.info(f"   理由: {advice_data.get('reason', 'N/A')[:50]}...")

                # 保存监控日志
                if checkpoint_manager:
                    try:
                        position_info = advice_data.get("position_info") or {}
                        log_id = await checkpoint_manager.save_stock_monitor_log(
                            user_id=user_id,
                            stock_code=code,
                            stock_name=advice_data.get("stock_name"),
                            current_price=advice_data.get("current_price"),
                            action=action,
                            reason=advice_data.get("reason"),
                            analysis_summary=advice_data.get("analysis_summary"),
                            llm_request=advice_data.get("llm_request"),  # 发送给 LLM 的请求报文
                            llm_response=advice_data,  # 保存完整的 LLM 返回
                            position_shares=position_info.get("shares"),
                            cost_price=position_info.get("cost_price"),
                            profit_loss_pct=position_info.get("profit_loss_pct"),
                            email_sent=False,  # 稍后更新
                        )
                        log_ids[code] = log_id
                        logger.info(f"📝 [{code}] 监控日志已保存 (id={log_id})")
                    except Exception as e:
                        logger.warning(f"⚠️ [{code}] 保存监控日志失败: {e}")

                # 只有买入、卖出、建仓时才发送邮件
                if action in ["买入", "卖出", "建仓"]:
                    logger.info(f"⚡ [{code}] 检测到 {action} 信号，准备发送邮件提醒")
                    alerts_to_send.append(advice_data)
            else:
                logger.warning(f"⚠️ [{code}] 分析失败: {result.error}")

        except Exception as e:
            logger.error(f"❌ [{code}] 处理异常: {e}")

    # 发送邮件提醒
    if alerts_to_send and jarvis and email_agent_ready:
        logger.info("=" * 40)
        logger.info(f"📧 发送 {len(alerts_to_send)} 条操作提醒...")

        email_sent = await _send_stock_alerts_email(jarvis, monitor_skill, alerts_to_send)

        # 更新日志中的邮件发送状态
        if email_sent and checkpoint_manager:
            for alert in alerts_to_send:
                code = alert.get("stock_code")
                if code and code in log_ids:
                    try:
                        await checkpoint_manager.update_stock_monitor_log_email_status(
                            log_id=log_ids[code],
                            email_sent=True,
                        )
                        logger.info(f"📝 [{code}] 已更新邮件发送状态")
                    except Exception as e:
                        logger.warning(f"⚠️ [{code}] 更新邮件状态失败: {e}")
    elif alerts_to_send:
        logger.warning(f"⚠️ 有 {len(alerts_to_send)} 条提醒但无法发送邮件")
    else:
        logger.info("✅ 所有股票建议持有/观望，无需发送提醒")

    # 清理资源
    if jarvis:
        await jarvis.close()
    if checkpoint_manager:
        await checkpoint_manager.close()

    logger.info("=" * 60)
    logger.info("✅ 盘中监控任务完成")
    logger.info("=" * 60)


async def _send_stock_alerts_email(jarvis, monitor_skill, alerts: List[Dict[str, Any]]) -> bool:
    """发送股票操作提醒邮件"""
    from src.a2a import TaskState
    from src.skills.stock_monitor import StockAdvice, TradeAction

    default_to = os.getenv("SMTP_DEFAULT_TO", "")
    if not default_to:
        logger.error("❌ 未配置默认收件人邮箱 (SMTP_DEFAULT_TO)")
        return False

    # 构建邮件内容
    now = datetime.now()
    report_time = now.strftime("%Y-%m-%d %H:%M")

    # 汇总所有提醒
    content_parts = [
        f"# 📊 股票操作提醒",
        "",
        f"**提醒时间**: {report_time}",
        f"**提醒数量**: {len(alerts)} 条",
        "",
        "---",
        "",
    ]

    for i, alert in enumerate(alerts, 1):
        action = alert.get("action", "持有")
        emoji = "🟢" if action == "买入" else "🔴"

        content_parts.append(f"## {i}. {emoji} {alert.get('stock_name', 'N/A')} ({alert.get('stock_code', 'N/A')})")
        content_parts.append("")
        content_parts.append(f"| 项目 | 内容 |")
        content_parts.append(f"|------|------|")
        content_parts.append(f"| 当前价格 | ¥{alert.get('current_price', 0):.2f} |")
        content_parts.append(f"| **操作建议** | **{action}** |")
        content_parts.append("")
        content_parts.append(f"### 📋 建议理由")
        content_parts.append(alert.get("reason", "N/A"))
        content_parts.append("")

        # 持仓信息
        position_info = alert.get("position_info")
        if position_info and position_info.get("shares"):
            content_parts.append("### 💼 持仓情况")
            content_parts.append(f"| 项目 | 数值 |")
            content_parts.append(f"|------|------|")
            content_parts.append(f"| 持有股数 | {position_info.get('shares', 0):.0f} 股 |")
            content_parts.append(f"| 成本价 | ¥{position_info.get('cost_price', 0):.2f} |")
            pct = position_info.get("profit_loss_pct", 0)
            content_parts.append(f"| 当前盈亏 | {pct:+.2f}% |")
            content_parts.append("")

        content_parts.append("---")
        content_parts.append("")

    content_parts.append("")
    content_parts.append("*本提醒由 Jarvis 智能助手自动生成，仅供参考，不构成投资建议。*")

    email_content = "\n".join(content_parts)

    # 发送邮件
    subject = f"股票操作提醒 - {now.strftime('%Y-%m-%d %H:%M')}"
    email_message = f"""发送邮件到 {default_to}
主题: '{subject}'
内容: '{email_content}'"""

    try:
        task = await jarvis.delegate_task("email_agent", email_message)

        if task.status.state == TaskState.COMPLETED:
            logger.info(f"✅ 操作提醒已发送到 {default_to}")
            return True
        else:
            logger.error(f"❌ 邮件发送失败: {task.status.message}")
            return False
    except Exception as e:
        logger.error(f"❌ 邮件发送出错: {e}")
        return False
