"""A2A Protocol 模块

Google A2A (Agent-to-Agent) 协议的 Python 实现。
提供 Agent 间通信的基础设施。
"""
from .types import (
    # 基础枚举
    TaskState,
    PartType,
    # 消息部分
    Part,
    TextPart,
    FilePart,
    DataPart,
    # 核心类型
    Message,
    Artifact,
    TaskStatus,
    Task,
    # 请求响应类型
    TaskSendRequest,
    TaskSendResponse,
    TaskGetRequest,
    TaskGetResponse,
    TaskCancelRequest,
    TaskCancelResponse,
    # Agent Card
    AgentAuthentication,
    AgentCapabilities,
    AgentSkill,
    AgentProvider,
    AgentCard,
    AgentInfo,
)
from .server import A2AServer, create_a2a_server
from .client import A2AClient, A2AClientError, discover_agents

__all__ = [
    # 枚举
    "TaskState",
    "PartType",
    # 消息部分
    "Part",
    "TextPart",
    "FilePart",
    "DataPart",
    # 核心类型
    "Message",
    "Artifact",
    "TaskStatus",
    "Task",
    # 请求响应
    "TaskSendRequest",
    "TaskSendResponse",
    "TaskGetRequest",
    "TaskGetResponse",
    "TaskCancelRequest",
    "TaskCancelResponse",
    # Agent Card
    "AgentAuthentication",
    "AgentCapabilities",
    "AgentSkill",
    "AgentProvider",
    "AgentCard",
    "AgentInfo",
    # Server
    "A2AServer",
    "create_a2a_server",
    # Client
    "A2AClient",
    "A2AClientError",
    "discover_agents",
]
