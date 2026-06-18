"""A2A Protocol 数据类型定义

Google A2A (Agent-to-Agent) 协议的核心数据类型。
参考: https://github.com/google/A2A
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class TaskState(str, Enum):
    """任务状态枚举"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class PartType(str, Enum):
    """消息部分类型"""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


class Part(BaseModel):
    """消息部分 - 可以是文本、文件或数据"""
    type: PartType
    text: Optional[str] = None
    file_uri: Optional[str] = None
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class TextPart(Part):
    """文本消息部分"""
    type: PartType = PartType.TEXT
    text: str


class FilePart(Part):
    """文件消息部分"""
    type: PartType = PartType.FILE
    file_uri: str
    file_name: Optional[str] = None
    mime_type: Optional[str] = None


class DataPart(Part):
    """数据消息部分"""
    type: PartType = PartType.DATA
    data: Dict[str, Any]


class Message(BaseModel):
    """A2A 消息"""
    role: str = Field(..., description="消息角色: user 或 agent")
    parts: List[Part] = Field(default_factory=list, description="消息内容部分列表")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

    @classmethod
    def user_text(cls, text: str) -> "Message":
        """创建用户文本消息"""
        return cls(role="user", parts=[TextPart(text=text)])

    @classmethod
    def agent_text(cls, text: str) -> "Message":
        """创建 Agent 文本消息"""
        return cls(role="agent", parts=[TextPart(text=text)])


class Artifact(BaseModel):
    """任务产物"""
    name: str = Field(..., description="产物名称")
    parts: List[Part] = Field(default_factory=list, description="产物内容")
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    """任务状态详情"""
    state: TaskState = TaskState.SUBMITTED
    message: Optional[Message] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class Task(BaseModel):
    """A2A 任务 - Agent 间工作交换的基本单位"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    status: TaskStatus = Field(default_factory=TaskStatus)
    artifacts: List[Artifact] = Field(default_factory=list)
    history: List[Message] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def add_message(self, message: Message) -> None:
        """添加消息到历史"""
        self.history.append(message)

    def add_artifact(self, artifact: Artifact) -> None:
        """添加产物"""
        self.artifacts.append(artifact)

    def set_state(self, state: TaskState, message: Optional[str] = None) -> None:
        """设置任务状态"""
        msg = Message.agent_text(message) if message else None
        self.status = TaskStatus(state=state, message=msg)


class TaskSendRequest(BaseModel):
    """发送任务请求"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    message: Message
    metadata: Optional[Dict[str, Any]] = None


class TaskSendResponse(BaseModel):
    """发送任务响应"""
    task: Task


class TaskGetRequest(BaseModel):
    """获取任务请求"""
    id: str
    history_length: Optional[int] = None


class TaskGetResponse(BaseModel):
    """获取任务响应"""
    task: Task


class TaskCancelRequest(BaseModel):
    """取消任务请求"""
    id: str


class TaskCancelResponse(BaseModel):
    """取消任务响应"""
    task: Task


class AgentAuthentication(BaseModel):
    """Agent 认证配置"""
    schemes: List[str] = Field(default_factory=list)
    credentials: Optional[str] = None


class AgentCapabilities(BaseModel):
    """Agent 能力描述"""
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True


class AgentSkill(BaseModel):
    """Agent 技能描述"""
    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    input_modes: List[str] = Field(default_factory=lambda: ["text"])
    output_modes: List[str] = Field(default_factory=lambda: ["text"])


class AgentProvider(BaseModel):
    """Agent 提供者信息"""
    organization: str
    url: Optional[str] = None


class AgentCard(BaseModel):
    """Agent Card - 描述 Agent 能力的 JSON 元数据

    发布在 /.well-known/agent.json
    """
    name: str = Field(..., description="Agent 名称")
    description: str = Field(..., description="Agent 描述")
    url: str = Field(..., description="Agent 服务 URL")
    version: str = Field(default="1.0.0", description="Agent 版本")
    documentation_url: Optional[str] = None
    provider: Optional[AgentProvider] = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    authentication: Optional[AgentAuthentication] = None
    default_input_modes: List[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: List[str] = Field(default_factory=lambda: ["text"])
    skills: List[AgentSkill] = Field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 可序列化的字典"""
        return self.model_dump(exclude_none=True)


class AgentInfo(BaseModel):
    """Agent 信息（用于注册表）"""
    card: AgentCard
    url: str
    last_seen: datetime = Field(default_factory=datetime.now)
    healthy: bool = True
