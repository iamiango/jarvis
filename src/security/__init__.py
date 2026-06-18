"""Security 模块 - 提供安全相关功能

包含:
- PII 防护: 检测和处理个人身份信息
- Human-in-the-Loop: 敏感操作人工确认机制
"""
from .pii_guard import PIIGuard, PIIType, PIIStrategy, PIIProcessResult, PIIBlockedError
from .human_in_the_loop import (
    HumanInTheLoop,
    ConfirmationHandler,
    ConsoleConfirmationHandler,
    CallbackConfirmationHandler,
    ConfirmationRequest,
    ConfirmationResponse,
    ConfirmationResult,
    OperationCancelled,
)

__all__ = [
    # PII Guard
    "PIIGuard",
    "PIIType",
    "PIIStrategy",
    "PIIProcessResult",
    "PIIBlockedError",
    # Human-in-the-Loop
    "HumanInTheLoop",
    "ConfirmationHandler",
    "ConsoleConfirmationHandler",
    "CallbackConfirmationHandler",
    "ConfirmationRequest",
    "ConfirmationResponse",
    "ConfirmationResult",
    "OperationCancelled",
]
