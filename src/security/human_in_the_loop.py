"""Human-in-the-Loop 模块 - 提供人工确认机制

在执行敏感操作前请求用户确认。
支持装饰器模式和回调模式。
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable
from functools import wraps


class ConfirmationResult(Enum):
    """确认结果"""
    APPROVED = "approved"      # 用户批准
    REJECTED = "rejected"      # 用户拒绝
    TIMEOUT = "timeout"        # 超时
    CANCELLED = "cancelled"    # 取消


@dataclass
class ConfirmationRequest:
    """确认请求"""
    action: str                          # 操作类型（如 "send_email"）
    description: str                     # 操作描述
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    timeout_seconds: Optional[float] = None  # 超时时间


@dataclass
class ConfirmationResponse:
    """确认响应"""
    result: ConfirmationResult
    message: Optional[str] = None
    modified_details: Optional[Dict[str, Any]] = None  # 用户修改后的详情


class ConfirmationHandler(ABC):
    """确认处理器抽象基类

    子类需要实现 request_confirmation 方法来处理确认请求
    """

    @abstractmethod
    async def request_confirmation(self, request: ConfirmationRequest) -> ConfirmationResponse:
        """请求用户确认

        Args:
            request: 确认请求

        Returns:
            ConfirmationResponse 确认响应
        """
        pass


class ConsoleConfirmationHandler(ConfirmationHandler):
    """控制台确认处理器

    在控制台打印确认请求，等待用户输入 y/n
    """

    async def request_confirmation(self, request: ConfirmationRequest) -> ConfirmationResponse:
        """在控制台请求用户确认"""
        print("\n" + "=" * 60)
        print(f"🛑 需要确认: {request.description}")
        print("=" * 60)

        # 显示详细信息
        if request.details:
            print("\n📋 操作详情:")
            for key, value in request.details.items():
                # 对于长文本，截断显示
                if isinstance(value, str) and len(value) > 200:
                    display_value = value[:200] + "...(已截断)"
                else:
                    display_value = value
                print(f"   {key}: {display_value}")

        print("\n" + "-" * 60)

        # 使用 asyncio 在线程池中运行阻塞的 input
        loop = asyncio.get_event_loop()
        try:
            user_input = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: input("确认执行? [y/N]: ")),
                timeout=request.timeout_seconds or 60.0
            )

            if user_input.lower() in ('y', 'yes', '是', '确认'):
                print("✅ 已确认，继续执行...")
                return ConfirmationResponse(result=ConfirmationResult.APPROVED)
            else:
                print("❌ 已取消")
                return ConfirmationResponse(
                    result=ConfirmationResult.REJECTED,
                    message="用户取消了操作"
                )

        except asyncio.TimeoutError:
            print("⏰ 确认超时，操作已取消")
            return ConfirmationResponse(
                result=ConfirmationResult.TIMEOUT,
                message="确认超时"
            )


class CallbackConfirmationHandler(ConfirmationHandler):
    """回调确认处理器

    通过回调函数请求确认，适用于 GUI 或 Web 应用
    """

    def __init__(
        self,
        callback: Callable[[ConfirmationRequest], Awaitable[ConfirmationResponse]]
    ):
        """初始化回调处理器

        Args:
            callback: 异步回调函数
        """
        self._callback = callback

    async def request_confirmation(self, request: ConfirmationRequest) -> ConfirmationResponse:
        """通过回调请求确认"""
        return await self._callback(request)


class HumanInTheLoop:
    """Human-in-the-Loop 管理器

    管理需要人工确认的操作。

    使用示例:
    ```python
    hitl = HumanInTheLoop()

    # 使用装饰器
    @hitl.before_action("send_email")
    async def send_email(to: str, content: str):
        ...

    # 手动检查
    response = await hitl.confirm(
        action="send_email",
        description="发送邮件",
        details={"to": "test@example.com", "subject": "Test"}
    )
    if response.result == ConfirmationResult.APPROVED:
        # 执行操作
        ...
    ```
    """

    def __init__(
        self,
        handler: Optional[ConfirmationHandler] = None,
        enabled: bool = True,
    ):
        """初始化 Human-in-the-Loop 管理器

        Args:
            handler: 确认处理器，默认使用控制台处理器
            enabled: 是否启用确认机制
        """
        self._handler = handler or ConsoleConfirmationHandler()
        self._enabled = enabled
        self._action_configs: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """设置是否启用"""
        self._enabled = value

    def set_handler(self, handler: ConfirmationHandler):
        """设置确认处理器"""
        self._handler = handler

    def configure_action(
        self,
        action: str,
        description: str,
        timeout_seconds: Optional[float] = None,
        detail_extractor: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        """配置操作的确认行为

        Args:
            action: 操作名称
            description: 操作描述模板
            timeout_seconds: 超时时间
            detail_extractor: 从参数中提取详情的函数
        """
        self._action_configs[action] = {
            "description": description,
            "timeout_seconds": timeout_seconds,
            "detail_extractor": detail_extractor,
        }

    async def confirm(
        self,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ConfirmationResponse:
        """请求确认

        Args:
            action: 操作类型
            description: 操作描述
            details: 详细信息
            timeout_seconds: 超时时间

        Returns:
            ConfirmationResponse
        """
        if not self._enabled:
            return ConfirmationResponse(result=ConfirmationResult.APPROVED)

        request = ConfirmationRequest(
            action=action,
            description=description,
            details=details or {},
            timeout_seconds=timeout_seconds,
        )

        return await self._handler.request_confirmation(request)

    def before_action(
        self,
        action: str,
        description: Optional[str] = None,
        detail_keys: Optional[List[str]] = None,
    ):
        """装饰器：在操作执行前请求确认

        Args:
            action: 操作名称
            description: 操作描述
            detail_keys: 要从函数参数中提取的键名列表

        Returns:
            装饰器函数
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self._enabled:
                    return await func(*args, **kwargs)

                # 构建详情
                details = {}
                if detail_keys:
                    for key in detail_keys:
                        if key in kwargs:
                            details[key] = kwargs[key]

                # 获取配置的描述或使用传入的描述
                config = self._action_configs.get(action, {})
                desc = description or config.get("description", f"执行操作: {action}")
                timeout = config.get("timeout_seconds")

                # 请求确认
                response = await self.confirm(
                    action=action,
                    description=desc,
                    details=details,
                    timeout_seconds=timeout,
                )

                if response.result != ConfirmationResult.APPROVED:
                    raise OperationCancelled(
                        f"操作 '{action}' 被取消: {response.message or response.result.value}"
                    )

                return await func(*args, **kwargs)

            return wrapper
        return decorator


class OperationCancelled(Exception):
    """操作被取消异常"""
    pass
