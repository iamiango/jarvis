"""LangGraph PostgresSaver - 多轮对话持久化支持

使用 LangGraph 内置的 AsyncPostgresSaver 实现会话历史持久化，
支持多轮对话中的上下文引用（如"它的MACD怎么样"）。

与现有 CheckpointManager 的表独立，不冲突。
"""
import asyncio
import logging
from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ..config import postgres_config

logger = logging.getLogger(__name__)


class LangGraphCheckpointer:
    """LangGraph PostgresSaver 管理器（单例模式）

    提供异步 PostgreSQL 检查点存储，用于 LangGraph 的多轮对话持久化。

    使用方式:
        # 获取 checkpointer 并传递给 graph.compile()
        checkpointer = await LangGraphCheckpointer.get_checkpointer()
        graph = workflow.compile(checkpointer=checkpointer)

        # 调用时传入 thread_id
        config = {"configurable": {"thread_id": "user123:stock_agent:active"}}
        result = await graph.ainvoke(state, config=config)

    线程安全: AsyncPostgresSaver 内部有 asyncio.Lock 保护。

    自动创建的表:
        - checkpoint_migrations
        - checkpoints (按 thread_id 分区)
        - checkpoint_blobs
        - checkpoint_writes
    """

    _instance: Optional["LangGraphCheckpointer"] = None
    _checkpointer: Optional[AsyncPostgresSaver] = None
    _context_manager = None  # 保持 context manager 引用
    _lock: asyncio.Lock = asyncio.Lock()
    _initialized: bool = False

    def __new__(cls) -> "LangGraphCheckpointer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_checkpointer(cls) -> AsyncPostgresSaver:
        """获取 AsyncPostgresSaver 实例（懒加载，单例）

        首次调用时会自动初始化连接并创建必要的表。
        如果连接已关闭，会自动重新创建连接。

        Returns:
            AsyncPostgresSaver 实例
        """
        async with cls._lock:
            # 检查是否需要重新连接（连接不存在或已关闭）
            need_reconnect = cls._checkpointer is None

            if not need_reconnect and cls._checkpointer is not None:
                # 检查连接是否仍然有效
                try:
                    # 尝试检查连接状态
                    if hasattr(cls._checkpointer, 'conn') and cls._checkpointer.conn is not None:
                        if cls._checkpointer.conn.is_closed():
                            need_reconnect = True
                            logger.info("🔄 检测到连接已关闭，准备重新连接...")
                except Exception:
                    need_reconnect = True

            if need_reconnect:
                # 清理旧的 context manager
                if cls._context_manager is not None:
                    try:
                        await cls._context_manager.__aexit__(None, None, None)
                    except Exception:
                        pass
                    cls._context_manager = None
                    cls._checkpointer = None

                conn_string = postgres_config.connection_string
                logger.info(f"🔌 初始化 LangGraph PostgresSaver...")

                # from_conn_string 返回 async context manager
                # 需要通过 __aenter__ 获取实际的 saver
                cls._context_manager = AsyncPostgresSaver.from_conn_string(conn_string)
                cls._checkpointer = await cls._context_manager.__aenter__()
                await cls._checkpointer.setup()

                cls._initialized = True
                logger.info("✅ LangGraph PostgresSaver 已初始化")

            return cls._checkpointer

    @classmethod
    async def close(cls) -> None:
        """关闭 checkpointer 连接

        在应用关闭时调用以释放数据库连接。
        """
        async with cls._lock:
            if cls._context_manager is not None:
                try:
                    # 正确关闭 context manager
                    await cls._context_manager.__aexit__(None, None, None)
                    logger.info("✅ LangGraph PostgresSaver 连接已关闭")
                except Exception as e:
                    logger.warning(f"⚠️ 关闭 checkpointer 时出错: {e}")
                finally:
                    cls._checkpointer = None
                    cls._context_manager = None
                    cls._initialized = False

    @classmethod
    def is_initialized(cls) -> bool:
        """检查 checkpointer 是否已初始化"""
        return cls._initialized

    @classmethod
    def reset(cls) -> None:
        """重置单例状态（仅用于测试）"""
        cls._checkpointer = None
        cls._context_manager = None
        cls._initialized = False


def generate_thread_id(user_id: str, agent_name: str, session_id: str = "active") -> str:
    """生成 thread_id

    Args:
        user_id: 用户标识
        agent_name: Agent 名称（如 stock_agent, fund_agent）
        session_id: 会话 ID，用于区分同一用户的不同对话
                   - 默认 "active" 表示单一活跃会话（向后兼容）
                   - 传入 UUID 可支持多个并行会话

    Returns:
        格式为 "{user_id}:{agent_name}:{session_id}" 的 thread_id

    Example:
        >>> generate_thread_id("user123", "stock_agent")
        "user123:stock_agent:active"

        >>> generate_thread_id("user123", "stock_agent", "550e8400-e29b-41d4")
        "user123:stock_agent:550e8400-e29b-41d4"
    """
    return f"{user_id}:{agent_name}:{session_id}"
