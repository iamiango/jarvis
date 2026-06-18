"""A2A Client - 用于 Agent 发现和任务委派

使用 httpx 实现异步 HTTP 客户端，支持:
- Agent 发现（通过 Agent Card）
- 任务发送
- 任务状态查询
- 任务取消
"""
from typing import Any, Dict, Optional, List
import httpx
from datetime import datetime

from .types import (
    AgentCard,
    AgentInfo,
    Task,
    TaskState,
    TaskSendRequest,
    TaskSendResponse,
    TaskGetRequest,
    TaskGetResponse,
    TaskCancelRequest,
    TaskCancelResponse,
    Message,
    TextPart,
)


class A2AClientError(Exception):
    """A2A 客户端错误"""
    pass


class A2AClient:
    """A2A Protocol Client - 用于与其他 Agent 通信"""

    def __init__(self, timeout: float = 300.0):
        """
        初始化 A2A Client

        Args:
            timeout: 请求超时时间（秒），默认 300 秒以支持长时间运行的任务
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            # 使用 httpx.Timeout 正确配置所有超时
            timeout_config = httpx.Timeout(
                connect=30.0,       # 连接超时 30 秒
                read=self.timeout,  # 读取超时使用配置的超时时间
                write=30.0,         # 写入超时 30 秒
                pool=30.0           # 连接池超时 30 秒
            )
            # 设置 limits 以避免连接复用问题
            limits = httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0  # 30 秒后关闭空闲连接
            )
            self._client = httpx.AsyncClient(timeout=timeout_config, limits=limits)
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def discover(self, url: str) -> AgentCard:
        """发现 Agent - 获取 Agent Card

        Args:
            url: Agent 的基础 URL

        Returns:
            AgentCard 对象

        Raises:
            A2AClientError: 发现失败时抛出
        """
        client = await self._get_client()
        agent_card_url = f"{url.rstrip('/')}/.well-known/agent.json"

        try:
            response = await client.get(agent_card_url)
            response.raise_for_status()
            data = response.json()
            return AgentCard(**data)
        except httpx.HTTPStatusError as e:
            raise A2AClientError(f"无法获取 Agent Card: HTTP {e.response.status_code}") from e
        except httpx.ReadTimeout as e:
            raise A2AClientError(f"读取超时: 服务器响应时间过长 (>{self.timeout}s)") from e
        except httpx.ConnectTimeout as e:
            raise A2AClientError(f"连接超时: 无法连接到服务器") from e
        except httpx.RequestError as e:
            error_msg = str(e) or type(e).__name__
            raise A2AClientError(f"请求 Agent Card 失败: {error_msg}") from e
        except Exception as e:
            raise A2AClientError(f"解析 Agent Card 失败: {str(e)}") from e

    async def send_task(
        self,
        url: str,
        message: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """发送任务给 Agent

        Args:
            url: Agent 的基础 URL
            message: 用户消息文本
            task_id: 任务 ID（可选，用于继续现有任务）
            session_id: 会话 ID（可选）
            metadata: 元数据（可选）

        Returns:
            Task 对象

        Raises:
            A2AClientError: 发送失败时抛出
        """
        client = await self._get_client()
        task_url = f"{url.rstrip('/')}/tasks/send"

        # 构建请求
        request = TaskSendRequest(
            id=task_id,
            session_id=session_id,
            message=Message.user_text(message),
            metadata=metadata,
        )

        try:
            response = await client.post(
                task_url,
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            data = response.json()
            task_response = TaskSendResponse(**data)
            return task_response.task
        except httpx.HTTPStatusError as e:
            raise A2AClientError(f"发送任务失败: HTTP {e.response.status_code}") from e
        except httpx.ReadTimeout as e:
            raise A2AClientError(f"读取超时: 服务器响应时间过长 (>{self.timeout}s)") from e
        except httpx.ConnectTimeout as e:
            raise A2AClientError(f"连接超时: 无法连接到服务器") from e
        except httpx.RequestError as e:
            error_msg = str(e) or type(e).__name__
            raise A2AClientError(f"请求失败: {error_msg}") from e
        except Exception as e:
            raise A2AClientError(f"解析响应失败: {str(e)}") from e

    async def get_task(
        self,
        url: str,
        task_id: str,
        history_length: Optional[int] = None,
    ) -> Task:
        """获取任务状态

        Args:
            url: Agent 的基础 URL
            task_id: 任务 ID
            history_length: 返回的历史消息数量限制（可选）

        Returns:
            Task 对象

        Raises:
            A2AClientError: 获取失败时抛出
        """
        client = await self._get_client()
        task_url = f"{url.rstrip('/')}/tasks/get"

        request = TaskGetRequest(id=task_id, history_length=history_length)

        try:
            response = await client.post(
                task_url,
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            data = response.json()
            task_response = TaskGetResponse(**data)
            return task_response.task
        except httpx.HTTPStatusError as e:
            raise A2AClientError(f"获取任务失败: HTTP {e.response.status_code}") from e
        except httpx.ReadTimeout as e:
            raise A2AClientError(f"读取超时: 服务器响应时间过长 (>{self.timeout}s)") from e
        except httpx.ConnectTimeout as e:
            raise A2AClientError(f"连接超时: 无法连接到服务器") from e
        except httpx.RequestError as e:
            error_msg = str(e) or type(e).__name__
            raise A2AClientError(f"请求失败: {error_msg}") from e
        except Exception as e:
            raise A2AClientError(f"解析响应失败: {str(e)}") from e

    async def cancel_task(self, url: str, task_id: str) -> Task:
        """取消任务

        Args:
            url: Agent 的基础 URL
            task_id: 任务 ID

        Returns:
            Task 对象

        Raises:
            A2AClientError: 取消失败时抛出
        """
        client = await self._get_client()
        task_url = f"{url.rstrip('/')}/tasks/cancel"

        request = TaskCancelRequest(id=task_id)

        try:
            response = await client.post(
                task_url,
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            data = response.json()
            task_response = TaskCancelResponse(**data)
            return task_response.task
        except httpx.HTTPStatusError as e:
            raise A2AClientError(f"取消任务失败: HTTP {e.response.status_code}") from e
        except httpx.ReadTimeout as e:
            raise A2AClientError(f"读取超时: 服务器响应时间过长 (>{self.timeout}s)") from e
        except httpx.ConnectTimeout as e:
            raise A2AClientError(f"连接超时: 无法连接到服务器") from e
        except httpx.RequestError as e:
            error_msg = str(e) or type(e).__name__
            raise A2AClientError(f"请求失败: {error_msg}") from e
        except Exception as e:
            raise A2AClientError(f"解析响应失败: {str(e)}") from e

    async def health_check(self, url: str) -> bool:
        """检查 Agent 健康状态

        Args:
            url: Agent 的基础 URL

        Returns:
            True 如果 Agent 健康，否则 False
        """
        client = await self._get_client()
        health_url = f"{url.rstrip('/')}/health"

        try:
            response = await client.get(health_url)
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "healthy"
        except Exception:
            return False

    async def wait_for_completion(
        self,
        url: str,
        task_id: str,
        poll_interval: float = 1.0,
        max_wait: float = 300.0,
    ) -> Task:
        """等待任务完成

        Args:
            url: Agent 的基础 URL
            task_id: 任务 ID
            poll_interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）

        Returns:
            完成的 Task 对象

        Raises:
            A2AClientError: 超时或失败时抛出
        """
        import asyncio

        start_time = datetime.now()
        terminal_states = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.INPUT_REQUIRED}

        while True:
            task = await self.get_task(url, task_id)

            if task.status.state in terminal_states:
                return task

            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_wait:
                raise A2AClientError(f"等待任务完成超时 (>{max_wait}s)")

            await asyncio.sleep(poll_interval)


async def discover_agents(urls: List[str], timeout: float = 10.0) -> List[AgentInfo]:
    """发现多个 Agent

    Args:
        urls: Agent URL 列表
        timeout: 请求超时时间

    Returns:
        成功发现的 AgentInfo 列表
    """
    client = A2AClient(timeout=timeout)
    agents = []

    try:
        for url in urls:
            try:
                card = await client.discover(url)
                agents.append(AgentInfo(card=card, url=url))
            except A2AClientError as e:
                print(f"发现 Agent 失败 ({url}): {e}")
                continue
    finally:
        await client.close()

    return agents
