"""A2A Server - FastAPI 实现

提供 A2A 协议的 HTTP 端点:
- /.well-known/agent.json - Agent Card
- /tasks/send - 发送任务
- /tasks/get - 获取任务状态
- /tasks/cancel - 取消任务
"""
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from datetime import datetime

from .types import (
    AgentCard,
    Task,
    TaskState,
    TaskSendRequest,
    TaskSendResponse,
    TaskGetRequest,
    TaskGetResponse,
    TaskCancelRequest,
    TaskCancelResponse,
    Message,
    Artifact,
    TextPart,
)

if TYPE_CHECKING:
    from ..storage import CheckpointManager


class A2AServer:
    """A2A Protocol Server - FastAPI 实现"""

    def __init__(
        self,
        agent_card: AgentCard,
        task_handler: Callable[[Task], Task],
        checkpoint_manager: Optional["CheckpointManager"] = None,
    ):
        """
        初始化 A2A Server

        Args:
            agent_card: Agent Card 描述 Agent 能力
            task_handler: 任务处理函数，接收 Task 并返回处理后的 Task
            checkpoint_manager: 检查点管理器（可选，用于持久化任务状态）
        """
        self.agent_card = agent_card
        self.task_handler = task_handler
        self.checkpoint_manager = checkpoint_manager
        self._tasks: Dict[str, Task] = {}

        # 创建 FastAPI 应用
        self.app = FastAPI(
            title=f"{agent_card.name} A2A Server",
            description=agent_card.description,
            version=agent_card.version,
        )

        # 添加 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册路由
        self._register_routes()

    def _register_routes(self) -> None:
        """注册 A2A 协议路由"""

        @self.app.get("/.well-known/agent.json")
        async def get_agent_card() -> Dict[str, Any]:
            """返回 Agent Card"""
            return self.agent_card.to_json()

        @self.app.post("/tasks/send", response_model=TaskSendResponse)
        async def send_task(request: TaskSendRequest) -> TaskSendResponse:
            """发送任务"""
            import uuid
            # 创建新任务或获取现有任务
            task_id = request.id or str(uuid.uuid4())
            if task_id in self._tasks:
                task = self._tasks[task_id]
            else:
                # 尝试从 checkpoint 恢复任务
                if self.checkpoint_manager:
                    task = await self.checkpoint_manager.load_task(task_id)
                if task_id not in self._tasks and (not self.checkpoint_manager or not task):
                    task = Task(
                        id=task_id,
                        session_id=request.session_id,
                        metadata=request.metadata,
                    )
                self._tasks[task_id] = task

            # 添加用户消息到历史
            task.add_message(request.message)
            task.set_state(TaskState.WORKING, "正在处理任务...")

            # 保存任务状态到 checkpoint
            if self.checkpoint_manager:
                await self.checkpoint_manager.save_task(task, request.session_id)

            # 异步处理任务
            try:
                processed_task = await self._process_task(task)
                self._tasks[task_id] = processed_task

                # 保存完成后的任务状态
                if self.checkpoint_manager:
                    await self.checkpoint_manager.save_task(processed_task, request.session_id)

                return TaskSendResponse(task=processed_task)
            except Exception as e:
                task.set_state(TaskState.FAILED, f"任务处理失败: {str(e)}")
                self._tasks[task_id] = task

                # 保存失败状态
                if self.checkpoint_manager:
                    await self.checkpoint_manager.save_task(task, request.session_id)

                return TaskSendResponse(task=task)

        @self.app.post("/tasks/get", response_model=TaskGetResponse)
        async def get_task(request: TaskGetRequest) -> TaskGetResponse:
            """获取任务状态"""
            task_id = request.id

            # 先检查内存缓存
            if task_id not in self._tasks:
                # 尝试从 checkpoint 加载
                if self.checkpoint_manager:
                    task = await self.checkpoint_manager.load_task(task_id)
                    if task:
                        self._tasks[task_id] = task

            if task_id not in self._tasks:
                raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

            task = self._tasks[task_id]

            # 如果指定了历史长度限制，截取历史
            if request.history_length is not None and request.history_length > 0:
                task_copy = task.model_copy()
                task_copy.history = task_copy.history[-request.history_length:]
                return TaskGetResponse(task=task_copy)

            return TaskGetResponse(task=task)

        @self.app.post("/tasks/cancel", response_model=TaskCancelResponse)
        async def cancel_task(request: TaskCancelRequest) -> TaskCancelResponse:
            """取消任务"""
            task_id = request.id

            # 先检查内存缓存
            if task_id not in self._tasks:
                # 尝试从 checkpoint 加载
                if self.checkpoint_manager:
                    task = await self.checkpoint_manager.load_task(task_id)
                    if task:
                        self._tasks[task_id] = task

            if task_id not in self._tasks:
                raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

            task = self._tasks[task_id]

            # 只有特定状态的任务可以取消
            if task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                raise HTTPException(
                    status_code=400,
                    detail=f"任务 {task_id} 已经结束，无法取消"
                )

            task.set_state(TaskState.CANCELED, "任务已取消")
            self._tasks[task_id] = task

            # 保存取消状态到 checkpoint
            if self.checkpoint_manager:
                await self.checkpoint_manager.update_task_status(task_id, "canceled")

            return TaskCancelResponse(task=task)

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """健康检查"""
            return {
                "status": "healthy",
                "agent": self.agent_card.name,
                "version": self.agent_card.version,
                "timestamp": datetime.now().isoformat(),
            }

    async def _process_task(self, task: Task) -> Task:
        """处理任务

        调用注入的 task_handler 处理任务
        """
        # 如果 task_handler 是协程函数
        if asyncio.iscoroutinefunction(self.task_handler):
            return await self.task_handler(task)
        else:
            # 在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.task_handler, task)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """启动服务器"""
        uvicorn.run(self.app, host=host, port=port)

    async def arun(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """异步启动服务器"""
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()


def create_a2a_server(
    agent_card: AgentCard,
    task_handler: Callable[[Task], Task],
    checkpoint_manager: Optional["CheckpointManager"] = None,
) -> A2AServer:
    """创建 A2A Server 的工厂函数

    Args:
        agent_card: Agent Card
        task_handler: 任务处理函数
        checkpoint_manager: 检查点管理器（可选）

    Returns:
        A2AServer 实例
    """
    return A2AServer(
        agent_card=agent_card,
        task_handler=task_handler,
        checkpoint_manager=checkpoint_manager,
    )
