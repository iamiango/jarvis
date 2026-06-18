"""Jarvis 定时任务调度器

使用 APScheduler 实现定时任务调度，支持：
- Cron 表达式调度
- 工作日过滤
- 任务注册和管理
"""
import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED, JobExecutionEvent

logger = logging.getLogger(__name__)


class JarvisScheduler:
    """Jarvis 定时任务调度器"""

    def __init__(self, timezone: str = "Asia/Shanghai"):
        """
        初始化调度器

        Args:
            timezone: 时区，默认上海时区
        """
        self.timezone = timezone
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._running = False

        # 添加事件监听
        self._scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self._scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        self._scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)

    def _on_job_executed(self, event: JobExecutionEvent) -> None:
        """任务执行成功回调"""
        job_id = event.job_id
        logger.info(f"✅ 定时任务 [{job_id}] 执行完成")

    def _on_job_error(self, event: JobExecutionEvent) -> None:
        """任务执行错误回调"""
        job_id = event.job_id
        exception = event.exception
        logger.error(f"❌ 定时任务 [{job_id}] 执行失败: {exception}")

    def _on_job_missed(self, event: JobExecutionEvent) -> None:
        """任务错过执行回调"""
        job_id = event.job_id
        scheduled_time = event.scheduled_run_time
        logger.warning(f"⚠️ 定时任务 [{job_id}] 错过执行时间: {scheduled_time}")

    def add_cron_job(
        self,
        job_id: str,
        func: Callable,
        hour: int,
        minute: int,
        day_of_week: str = "mon-fri",
        description: str = "",
        **kwargs
    ) -> None:
        """
        添加 Cron 定时任务

        Args:
            job_id: 任务唯一标识
            func: 要执行的函数（可以是异步函数）
            hour: 小时 (0-23)
            minute: 分钟 (0-59)
            day_of_week: 星期几执行，默认工作日 (mon-fri)
            description: 任务描述
            **kwargs: 传递给函数的参数
        """
        trigger = CronTrigger(
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=self.timezone
        )

        self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            name=description or job_id,
            kwargs=kwargs,
            replace_existing=True,
            misfire_grace_time=3600,  # 允许错过1小时内的任务仍然执行
            coalesce=True,  # 如果错过多次，只执行一次
        )

        self._jobs[job_id] = {
            "func": func.__name__,
            "schedule": f"{hour:02d}:{minute:02d} ({day_of_week})",
            "description": description,
            "kwargs": kwargs
        }

        logger.info(f"📅 添加定时任务: [{job_id}] {description} @ {hour:02d}:{minute:02d} ({day_of_week})")

    def add_workday_job(
        self,
        job_id: str,
        func: Callable,
        hour: int,
        minute: int,
        description: str = "",
        **kwargs
    ) -> None:
        """
        添加工作日定时任务（周一到周五）

        Args:
            job_id: 任务唯一标识
            func: 要执行的函数
            hour: 小时 (0-23)
            minute: 分钟 (0-59)
            description: 任务描述
            **kwargs: 传递给函数的参数
        """
        self.add_cron_job(
            job_id=job_id,
            func=func,
            hour=hour,
            minute=minute,
            day_of_week="mon-fri",
            description=description,
            **kwargs
        )

    def remove_job(self, job_id: str) -> bool:
        """
        移除定时任务

        Args:
            job_id: 任务唯一标识

        Returns:
            是否成功移除
        """
        try:
            self._scheduler.remove_job(job_id)
            self._jobs.pop(job_id, None)
            logger.info(f"🗑️ 移除定时任务: [{job_id}]")
            return True
        except Exception as e:
            logger.error(f"移除任务失败 [{job_id}]: {e}")
            return False

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        列出所有定时任务

        Returns:
            任务列表
        """
        jobs = []
        for job in self._scheduler.get_jobs():
            job_info = self._jobs.get(job.id, {})
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name,
                "schedule": job_info.get("schedule", "N/A"),
                "description": job_info.get("description", ""),
                "next_run": next_run.strftime("%Y-%m-%d %H:%M:%S") if next_run else "N/A"
            })
        return jobs

    def start(self) -> None:
        """启动调度器"""
        if not self._running:
            self._scheduler.start()
            self._running = True
            logger.info("🚀 Jarvis 调度器已启动")

    def stop(self) -> None:
        """停止调度器"""
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("🛑 Jarvis 调度器已停止")

    @property
    def is_running(self) -> bool:
        """调度器是否正在运行"""
        return self._running

    async def run_job_now(self, job_id: str) -> bool:
        """
        立即执行指定任务

        Args:
            job_id: 任务唯一标识

        Returns:
            是否成功触发
        """
        job = self._scheduler.get_job(job_id)
        if job:
            logger.info(f"⚡ 立即执行任务: [{job_id}]")
            job.modify(next_run_time=datetime.now(self._scheduler.timezone))
            return True
        else:
            logger.error(f"任务不存在: [{job_id}]")
            return False
