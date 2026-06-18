#!/usr/bin/env python3
"""Jarvis 定时任务调度器启动脚本

使用方法:
    # 启动调度器（使用 .env 中的配置）
    python -m src.scheduler.run

    # 立即执行一次任务（测试用）
    python -m src.scheduler.run --run-now daily_fund_report
    python -m src.scheduler.run --run-now daily_stock_report
    python -m src.scheduler.run --run-now hourly_stock_monitor

    # 列出所有任务
    python -m src.scheduler.run --list

配置说明:
    在 .env 文件中配置以下环境变量：
    - SCHEDULER_ENABLED=true          # 是否启用调度器
    - FUND_REPORT_HOUR=14             # 基金报告发送小时
    - FUND_REPORT_MINUTE=20           # 基金报告发送分钟
    - STOCK_REPORT_HOUR=15            # 股票报告发送小时
    - STOCK_REPORT_MINUTE=30          # 股票报告发送分钟
    - STOCK_MONITOR_ENABLED=true      # 是否启用盘中股票监控
    - FUND_CODES=008089,021500,...    # 要分析的基金代码
    - STOCK_CODES=600588,002230,...   # 要分析的股票代码
    - SMTP_DEFAULT_TO=xxx@example.com # 报告接收邮箱

盘中股票监控说明:
    - 工作日 9:00, 10:00, 11:00, 13:00, 14:00 执行（跳过午休和收盘）
    - 读取 STOCK_CODES 中的股票，获取数据并分析
    - 结合用户持仓和偏好，使用 Qwen 模型给出操作建议
    - 仅在建议为"买入"或"卖出"时发送邮件提醒
"""
import argparse
import asyncio
import logging
import os
import signal
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.scheduler.scheduler import JarvisScheduler
from src.scheduler.jobs import JobRegistry, daily_fund_report_job, daily_stock_report_job, hourly_stock_monitor_job

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_scheduler() -> JarvisScheduler:
    """设置调度器和任务"""
    scheduler = JarvisScheduler(timezone="Asia/Shanghai")

    # 检查基金日报是否启用
    fund_report_enabled = os.getenv("FUND_REPORT_ENABLED", "true").lower() == "true"
    if fund_report_enabled:
        # 从环境变量读取基金报告配置
        fund_hour = int(os.getenv("FUND_REPORT_HOUR", "14"))
        fund_minute = int(os.getenv("FUND_REPORT_MINUTE", "20"))

        # 注册每日基金报告任务（工作日执行）
        scheduler.add_workday_job(
            job_id="daily_fund_report",
            func=daily_fund_report_job,
            hour=fund_hour,
            minute=fund_minute,
            description=f"每日基金分析报告 (工作日 {fund_hour:02d}:{fund_minute:02d})"
        )
    else:
        logger.info("⏭️ 基金日报作业已禁用 (FUND_REPORT_ENABLED=false)")

    # 检查股票日报是否启用
    stock_report_enabled = os.getenv("STOCK_REPORT_ENABLED", "true").lower() == "true"
    if stock_report_enabled:
        # 从环境变量读取股票报告配置
        stock_hour = int(os.getenv("STOCK_REPORT_HOUR", "15"))
        stock_minute = int(os.getenv("STOCK_REPORT_MINUTE", "30"))

        # 注册每日股票报告任务（工作日执行）
        scheduler.add_workday_job(
            job_id="daily_stock_report",
            func=daily_stock_report_job,
            hour=stock_hour,
            minute=stock_minute,
            description=f"每日股票分析报告 (工作日 {stock_hour:02d}:{stock_minute:02d})"
        )
    else:
        logger.info("⏭️ 股票日报作业已禁用 (STOCK_REPORT_ENABLED=false)")

    # 检查盘中股票监控是否启用
    stock_monitor_enabled = os.getenv("STOCK_MONITOR_ENABLED", "false").lower() == "true"
    if stock_monitor_enabled:
        # 盘中监控：工作日 9:00-14:00 每小时执行（整点）
        # 注意：15:00 不执行因为已收盘
        monitor_hours = [9, 10, 11, 13, 14]  # 跳过 12:00 午休
        for hour in monitor_hours:
            scheduler.add_workday_job(
                job_id=f"hourly_stock_monitor_{hour:02d}",
                func=hourly_stock_monitor_job,
                hour=hour,
                minute=0,
                description=f"盘中股票监控 (工作日 {hour:02d}:00)"
            )
        logger.info(f"📊 盘中股票监控已启用 (工作日 {monitor_hours})")
    else:
        logger.info("⏭️ 盘中股票监控已禁用 (STOCK_MONITOR_ENABLED=false)")

    return scheduler


async def run_scheduler():
    """运行调度器"""
    scheduler = setup_scheduler()

    # 注册信号处理
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(scheduler)))

    scheduler.start()

    # 打印任务列表
    print("\n" + "=" * 60)
    print("Jarvis 定时任务调度器")
    print("=" * 60)
    print("\n📅 已注册的定时任务:")
    for job in scheduler.list_jobs():
        print(f"  - [{job['id']}] {job['description']}")
        print(f"    下次执行: {job['next_run']}")
    print("\n按 Ctrl+C 停止调度器\n")
    print("=" * 60 + "\n")

    # 保持运行
    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass


async def shutdown(scheduler: JarvisScheduler):
    """关闭调度器"""
    logger.info("正在关闭调度器...")
    scheduler.stop()
    # 取消所有任务
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()


async def run_job_immediately(job_id: str):
    """立即执行指定任务"""
    logger.info(f"⚡ 立即执行任务: {job_id}")

    try:
        job_func = JobRegistry.get_job(job_id)
        await job_func()
        logger.info(f"✅ 任务 {job_id} 执行完成")
    except ValueError as e:
        logger.error(f"❌ {e}")
    except Exception as e:
        logger.error(f"❌ 任务执行失败: {e}")


def list_jobs():
    """列出所有可用任务"""
    print("\n📋 可用的定时任务:")
    print("-" * 40)
    for job in JobRegistry.list_jobs():
        print(f"  - {job['id']}: {job['description']}")
    print("-" * 40)
    print()


def main():
    parser = argparse.ArgumentParser(description="Jarvis 定时任务调度器")
    parser.add_argument(
        "--run-now",
        metavar="JOB_ID",
        help="立即执行指定任务（用于测试）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用任务"
    )

    args = parser.parse_args()

    if args.list:
        list_jobs()
        return

    if args.run_now:
        asyncio.run(run_job_immediately(args.run_now))
        return

    # 检查是否启用调度器
    if os.getenv("SCHEDULER_ENABLED", "true").lower() != "true":
        logger.warning("调度器已禁用 (SCHEDULER_ENABLED != true)")
        return

    # 启动调度器
    try:
        asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        logger.info("调度器已停止")


if __name__ == "__main__":
    main()
