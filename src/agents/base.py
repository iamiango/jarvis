"""Base Agent - 所有 Agent 的基类

提供 A2A 协议支持和通用功能。
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import asyncio

from ..a2a import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    AgentProvider,
    A2AServer,
    Task,
    TaskState,
    Message,
    Artifact,
    TextPart,
)
from ..skills import BaseSkill, skill_manager

if TYPE_CHECKING:
    from ..storage import CheckpointManager


class BaseAgent(ABC):
    """所有 Agent 的基类，支持 A2A 协议"""

    # 子类必须定义的属性
    name: str = "base_agent"
    description: str = "Base Agent"
    version: str = "1.0.0"

    def __init__(self, checkpoint_manager: Optional["CheckpointManager"] = None):
        """初始化 Agent

        Args:
            checkpoint_manager: 检查点管理器（可选，用于持久化状态）
        """
        self._skills: Dict[str, BaseSkill] = {}
        self._server: Optional[A2AServer] = None
        self._url: Optional[str] = None
        self._checkpoint_manager = checkpoint_manager

    def register_skill(self, skill: BaseSkill) -> None:
        """注册技能到 Agent"""
        self._skills[skill.name] = skill
        # 同时注册到全局 skill_manager
        skill_manager.register(skill)

    def unregister_skill(self, name: str) -> None:
        """注销技能"""
        if name in self._skills:
            del self._skills[name]
            skill_manager.unregister(name)

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """获取技能"""
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有技能"""
        return list(self._skills.keys())

    @abstractmethod
    def get_skills(self) -> List[AgentSkill]:
        """获取 Agent 的技能列表（用于 Agent Card）

        子类必须实现此方法，返回 AgentSkill 列表
        """
        pass

    @abstractmethod
    async def process_task(self, task: Task) -> Task:
        """处理任务

        子类必须实现此方法，处理接收到的 A2A 任务

        Args:
            task: A2A Task 对象

        Returns:
            处理后的 Task 对象
        """
        pass

    def get_agent_card(self, url: Optional[str] = None) -> AgentCard:
        """获取 Agent Card

        Args:
            url: Agent 服务的 URL

        Returns:
            AgentCard 对象
        """
        return AgentCard(
            name=self.name,
            description=self.description,
            url=url or self._url or "http://localhost:8000",
            version=self.version,
            provider=AgentProvider(organization="Jarvis"),
            capabilities=AgentCapabilities(
                streaming=False,
                push_notifications=False,
                state_transition_history=True,
            ),
            skills=self.get_skills(),
        )

    def create_server(self, host: str = "0.0.0.0", port: int = 8000) -> A2AServer:
        """创建 A2A Server

        Args:
            host: 主机地址
            port: 端口号

        Returns:
            A2AServer 实例
        """
        self._url = f"http://{host}:{port}"
        agent_card = self.get_agent_card(self._url)

        self._server = A2AServer(
            agent_card=agent_card,
            task_handler=self.process_task,
            checkpoint_manager=self._checkpoint_manager,
        )
        return self._server

    def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """启动 A2A Server（阻塞）

        Args:
            host: 主机地址
            port: 端口号
        """
        if self._server is None:
            self.create_server(host, port)
        self._server.run(host=host, port=port)

    async def astart_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """异步启动 A2A Server

        Args:
            host: 主机地址
            port: 端口号
        """
        if self._server is None:
            self.create_server(host, port)
        await self._server.arun(host=host, port=port)

    # 辅助方法

    def _extract_text_from_task(self, task: Task) -> str:
        """从任务中提取用户文本消息"""
        for message in reversed(task.history):
            if message.role == "user":
                for part in message.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
        return ""

    def _create_text_response(self, task: Task, text: str, state: TaskState = TaskState.COMPLETED) -> Task:
        """创建文本响应

        Args:
            task: 原始任务
            text: 响应文本
            state: 任务状态

        Returns:
            更新后的 Task
        """
        task.add_message(Message.agent_text(text))
        task.set_state(state, text[:100] if len(text) > 100 else text)
        return task

    def _create_artifact_response(
        self,
        task: Task,
        artifact_name: str,
        artifact_text: str,
        state: TaskState = TaskState.COMPLETED,
    ) -> Task:
        """创建带产物的响应

        Args:
            task: 原始任务
            artifact_name: 产物名称
            artifact_text: 产物内容（文本）
            state: 任务状态

        Returns:
            更新后的 Task
        """
        artifact = Artifact(
            name=artifact_name,
            parts=[TextPart(text=artifact_text)],
        )
        task.add_artifact(artifact)
        task.add_message(Message.agent_text(f"已生成: {artifact_name}"))
        task.set_state(state)
        return task

    # ==================== Checkpoint 辅助方法 ====================

    @property
    def checkpoint_manager(self) -> Optional["CheckpointManager"]:
        """获取检查点管理器"""
        return self._checkpoint_manager

    @checkpoint_manager.setter
    def checkpoint_manager(self, manager: Optional["CheckpointManager"]) -> None:
        """设置检查点管理器"""
        self._checkpoint_manager = manager

    async def save_agent_state(
        self,
        task_id: str,
        state_type: str,
        state_data: Dict[str, Any],
        step_number: Optional[int] = None,
    ) -> None:
        """保存 Agent 中间状态

        Args:
            task_id: 任务 ID
            state_type: 状态类型（如 intent, reasoning, decision）
            state_data: 状态数据
            step_number: 步骤编号
        """
        if self._checkpoint_manager:
            await self._checkpoint_manager.save_agent_state(
                task_id=task_id,
                agent_name=self.name,
                state_type=state_type,
                state_data=state_data,
                step_number=step_number,
            )

    async def save_tool_call(
        self,
        task_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        error_message: Optional[str] = None,
        execution_ms: Optional[int] = None,
    ) -> None:
        """记录工具调用

        Args:
            task_id: 任务 ID
            tool_name: 工具名称
            tool_input: 输入参数
            tool_output: 输出结果
            success: 是否成功
            error_message: 错误信息
            execution_ms: 执行时间（毫秒）
        """
        if self._checkpoint_manager:
            await self._checkpoint_manager.save_tool_call(
                task_id=task_id,
                agent_name=self.name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                success=success,
                error_message=error_message,
                execution_ms=execution_ms,
            )

    async def save_workflow_checkpoint(
        self,
        task_id: str,
        current_node: str,
        current_step: int,
        graph_state: Optional[Dict[str, Any]] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        """保存工作流检查点

        Args:
            task_id: 任务 ID
            current_node: 当前节点名称
            current_step: 当前步骤编号
            graph_state: 图状态数据
            total_steps: 总步骤数
        """
        if self._checkpoint_manager:
            await self._checkpoint_manager.save_checkpoint(
                task_id=task_id,
                current_node=current_node,
                current_step=current_step,
                graph_state=graph_state,
                total_steps=total_steps,
            )
