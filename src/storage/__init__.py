"""Storage module - 持久化存储支持

提供 PostgreSQL 检查点管理功能，支持:
- 会话管理
- 任务持久化
- 消息历史存储
- 工具调用记录
- Agent 状态保存
- 工作流检查点
- 多轮对话持久化 (LangGraph PostgresSaver)
"""
from .checkpoint import CheckpointManager
from .langgraph_checkpoint import LangGraphCheckpointer, generate_thread_id

__all__ = ["CheckpointManager", "LangGraphCheckpointer", "generate_thread_id"]
