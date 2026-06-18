"""CheckpointManager - PostgreSQL 检查点管理器

提供基于 PostgreSQL 的状态持久化功能，支持:
- Session 管理
- Task 持久化和恢复
- Message 历史存储
- 工具调用记录
- Agent 中间状态保存
- 工作流检查点
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg

from ..a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    Part,
    TextPart,
    FilePart,
    DataPart,
    PartType,
    Artifact,
)


# SQL 建表语句
CREATE_TABLES_SQL = """
-- 会话管理
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- 任务记录
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'submitted',
    input_text TEXT,
    output_text TEXT,
    artifacts JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

-- 对话历史
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT,
    parts JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(task_id);

-- 工具调用记录
CREATE TABLE IF NOT EXISTS tool_calls (
    id SERIAL PRIMARY KEY,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    agent_name VARCHAR(255) NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB,
    success BOOLEAN,
    error_message TEXT,
    execution_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id);

-- Agent 中间状态
CREATE TABLE IF NOT EXISTS agent_states (
    id SERIAL PRIMARY KEY,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    agent_name VARCHAR(255) NOT NULL,
    state_type VARCHAR(50) NOT NULL,
    state_data JSONB NOT NULL,
    step_number INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_states_task ON agent_states(task_id);

-- 工作流检查点
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    id SERIAL PRIMARY KEY,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE UNIQUE,
    current_node VARCHAR(255),
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER,
    graph_state JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_workflow_task ON workflow_checkpoints(task_id);

-- 用户偏好存储（长期记忆）
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,  -- 'stock', 'fund', 'email', 'general'
    preference_key VARCHAR(255) NOT NULL,
    preference_value JSONB NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(50) DEFAULT 'explicit',  -- 'explicit' | 'inferred'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, category, preference_key)
);
CREATE INDEX IF NOT EXISTS idx_pref_user ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_pref_category ON user_preferences(user_id, category);

-- 基金持仓存储（长期记忆）
CREATE TABLE IF NOT EXISTS fund_positions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    fund_code VARCHAR(20) NOT NULL,
    fund_name VARCHAR(255),
    shares DECIMAL(18,4) DEFAULT 0,         -- 持有份额
    cost_amount DECIMAL(18,2) DEFAULT 0,    -- 累计投入金额
    cost_nav DECIMAL(10,4),                  -- 持仓成本净值
    first_buy_date DATE,                     -- 首次买入日期
    last_trade_date DATE,                    -- 最后交易日期
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, fund_code)
);
CREATE INDEX IF NOT EXISTS idx_fund_positions_user ON fund_positions(user_id);

-- 股票持仓存储（长期记忆）
CREATE TABLE IF NOT EXISTS stock_positions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(255),
    shares DECIMAL(18,4) DEFAULT 0,         -- 持有股数
    cost_amount DECIMAL(18,2) DEFAULT 0,    -- 累计投入金额
    cost_price DECIMAL(10,4),                -- 持仓成本价
    first_buy_date DATE,                     -- 首次买入日期
    last_trade_date DATE,                    -- 最后交易日期
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_stock_positions_user ON stock_positions(user_id);

-- 盘中股票监控日志
CREATE TABLE IF NOT EXISTS stock_monitor_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL DEFAULT 'default',
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(255),
    job_time TIMESTAMP NOT NULL DEFAULT NOW(),
    current_price DECIMAL(10,4),
    action VARCHAR(20) NOT NULL,               -- '持有', '买入', '卖出', '建仓'
    reason TEXT,                               -- 操作建议理由
    analysis_summary TEXT,                     -- 技术分析摘要
    llm_request TEXT,                          -- 发送给远端 LLM 的请求报文
    llm_response JSONB,                        -- 远端 LLM 完整返回
    position_shares DECIMAL(18,4),             -- 当时持仓股数
    cost_price DECIMAL(10,4),                  -- 当时成本价
    profit_loss_pct DECIMAL(10,4),             -- 当时盈亏百分比
    email_sent BOOLEAN DEFAULT FALSE,          -- 是否发送了邮件
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_user ON stock_monitor_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_stock ON stock_monitor_logs(stock_code);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_time ON stock_monitor_logs(job_time);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_action ON stock_monitor_logs(action);
"""


class CheckpointManager:
    """PostgreSQL 检查点管理器

    提供完整的状态持久化功能，支持任务恢复和历史查询。
    """

    def __init__(self, connection_string: str):
        """初始化检查点管理器

        Args:
            connection_string: PostgreSQL 连接字符串
                格式: postgresql://user:password@host:port/database
        """
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化连接池和数据库表

        创建连接池并确保所有必需的表存在。
        """
        if self._initialized:
            return

        # 创建连接池
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )

        # 创建表
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)

        self._initialized = True

    async def close(self) -> None:
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """确保连接池已初始化（懒加载）"""
        if not self._initialized:
            await self.initialize()

    # ==================== Session 管理 ====================

    async def create_session(self, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """创建新会话

        Args:
            user_id: 用户标识（可选）
            metadata: 会话元数据

        Returns:
            会话 ID
        """
        await self._ensure_initialized()
        session_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, user_id, metadata)
                VALUES ($1, $2, $3)
                """,
                uuid.UUID(session_id),
                user_id,
                json.dumps(metadata or {}),
            )
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息

        Args:
            session_id: 会话 ID

        Returns:
            会话信息字典，不存在则返回 None
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1",
                uuid.UUID(session_id),
            )
            if row:
                return {
                    "id": str(row["id"]),
                    "user_id": row["user_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
        return None

    async def update_session(self, session_id: str, metadata: Dict) -> None:
        """更新会话元数据

        Args:
            session_id: 会话 ID
            metadata: 新的元数据（会合并到现有元数据）
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions
                SET metadata = metadata || $2::jsonb, updated_at = NOW()
                WHERE id = $1
                """,
                uuid.UUID(session_id),
                json.dumps(metadata),
            )

    # ==================== Task 管理 ====================

    async def save_task(self, task: Task, session_id: Optional[str] = None) -> None:
        """保存任务到数据库

        Args:
            task: A2A Task 对象
            session_id: 关联的会话 ID
        """
        await self._ensure_initialized()
        # 提取输入和输出文本
        input_text = self._extract_user_text(task)
        output_text = self._extract_agent_text(task)

        # 序列化 artifacts
        artifacts_json = json.dumps([
            {
                "name": a.name,
                "parts": [self._serialize_part(p) for p in a.parts],
                "metadata": a.metadata,
            }
            for a in task.artifacts
        ])

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tasks (id, session_id, status, input_text, output_text, artifacts, metadata, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    output_text = EXCLUDED.output_text,
                    artifacts = EXCLUDED.artifacts,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW(),
                    completed_at = EXCLUDED.completed_at
                """,
                uuid.UUID(task.id),
                uuid.UUID(session_id) if session_id else None,
                task.status.state.value,
                input_text,
                output_text,
                artifacts_json,
                json.dumps(task.metadata or {}),
                datetime.now() if task.status.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED} else None,
            )

        # 保存消息历史
        for msg in task.history:
            await self.save_message(task.id, msg)

    async def load_task(self, task_id: str) -> Optional[Task]:
        """从数据库加载任务

        Args:
            task_id: 任务 ID

        Returns:
            Task 对象，不存在则返回 None
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM tasks WHERE id = $1",
                uuid.UUID(task_id),
            )
            if not row:
                return None

            # 加载消息历史
            messages = await self.get_messages(task_id)

            # 反序列化 artifacts
            artifacts_data = json.loads(row["artifacts"]) if row["artifacts"] else []
            artifacts = [
                Artifact(
                    name=a["name"],
                    parts=[self._deserialize_part(p) for p in a.get("parts", [])],
                    metadata=a.get("metadata"),
                )
                for a in artifacts_data
            ]

            # 构建 Task 对象
            task = Task(
                id=str(row["id"]),
                session_id=str(row["session_id"]) if row["session_id"] else None,
                status=TaskStatus(
                    state=TaskState(row["status"]),
                    timestamp=row["updated_at"],
                ),
                artifacts=artifacts,
                history=messages,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            return task

    async def update_task_status(self, task_id: str, status: str, output_text: Optional[str] = None) -> None:
        """更新任务状态

        Args:
            task_id: 任务 ID
            status: 新状态
            output_text: 输出文本（可选）
        """
        await self._ensure_initialized()
        completed_at = datetime.now() if status in {"completed", "failed", "canceled"} else None

        async with self.pool.acquire() as conn:
            if output_text:
                await conn.execute(
                    """
                    UPDATE tasks SET status = $2, output_text = $3, updated_at = NOW(), completed_at = $4
                    WHERE id = $1
                    """,
                    uuid.UUID(task_id),
                    status,
                    output_text,
                    completed_at,
                )
            else:
                await conn.execute(
                    """
                    UPDATE tasks SET status = $2, updated_at = NOW(), completed_at = $3
                    WHERE id = $1
                    """,
                    uuid.UUID(task_id),
                    status,
                    completed_at,
                )

    async def list_tasks(
        self,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """列出任务

        Args:
            session_id: 会话 ID（过滤条件）
            status: 状态（过滤条件）
            limit: 返回数量限制

        Returns:
            任务信息列表
        """
        query = "SELECT id, session_id, status, input_text, created_at, updated_at FROM tasks WHERE 1=1"
        params = []
        param_idx = 1

        if session_id:
            query += f" AND session_id = ${param_idx}"
            params.append(uuid.UUID(session_id))
            param_idx += 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        query += f" ORDER BY created_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                {
                    "id": str(row["id"]),
                    "session_id": str(row["session_id"]) if row["session_id"] else None,
                    "status": row["status"],
                    "input_text": row["input_text"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    # ==================== Message 历史 ====================

    async def save_message(self, task_id: str, message: Message) -> int:
        """保存消息到历史

        Args:
            task_id: 任务 ID
            message: Message 对象

        Returns:
            消息 ID
        """
        # 提取文本内容
        content = ""
        for part in message.parts:
            if hasattr(part, "text") and part.text:
                content = part.text
                break

        # 序列化 parts
        parts_json = json.dumps([self._serialize_part(p) for p in message.parts])

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (task_id, role, content, parts, created_at)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING id
                """,
                uuid.UUID(task_id),
                message.role,
                content,
                parts_json,
                message.timestamp or datetime.now(),
            )
            return row["id"]

    async def get_messages(self, task_id: str) -> List[Message]:
        """获取任务的消息历史

        Args:
            task_id: 任务 ID

        Returns:
            Message 对象列表
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, parts, created_at
                FROM messages WHERE task_id = $1
                ORDER BY id ASC
                """,
                uuid.UUID(task_id),
            )

            messages = []
            for row in rows:
                parts_data = json.loads(row["parts"]) if row["parts"] else []
                parts = [self._deserialize_part(p) for p in parts_data]

                # 如果没有 parts，从 content 创建
                if not parts and row["content"]:
                    parts = [TextPart(text=row["content"])]

                messages.append(Message(
                    role=row["role"],
                    parts=parts,
                    timestamp=row["created_at"],
                ))

            return messages

    # ==================== 工具调用记录 ====================

    async def save_tool_call(
        self,
        task_id: str,
        agent_name: str,
        tool_name: str,
        tool_input: Dict,
        tool_output: Optional[Dict] = None,
        success: Optional[bool] = None,
        error_message: Optional[str] = None,
        execution_ms: Optional[int] = None,
    ) -> int:
        """记录工具调用

        Args:
            task_id: 任务 ID
            agent_name: Agent 名称
            tool_name: 工具名称
            tool_input: 输入参数
            tool_output: 输出结果
            success: 是否成功
            error_message: 错误信息
            execution_ms: 执行时间（毫秒）

        Returns:
            记录 ID
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tool_calls (task_id, agent_name, tool_name, tool_input, tool_output, success, error_message, execution_ms)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8)
                RETURNING id
                """,
                uuid.UUID(task_id),
                agent_name,
                tool_name,
                json.dumps(tool_input),
                json.dumps(tool_output) if tool_output else None,
                success,
                error_message,
                execution_ms,
            )
            return row["id"]

    async def get_tool_calls(self, task_id: str) -> List[Dict]:
        """获取任务的工具调用记录

        Args:
            task_id: 任务 ID

        Returns:
            工具调用记录列表
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM tool_calls WHERE task_id = $1
                ORDER BY created_at ASC
                """,
                uuid.UUID(task_id),
            )
            return [
                {
                    "id": row["id"],
                    "agent_name": row["agent_name"],
                    "tool_name": row["tool_name"],
                    "tool_input": json.loads(row["tool_input"]) if row["tool_input"] else {},
                    "tool_output": json.loads(row["tool_output"]) if row["tool_output"] else None,
                    "success": row["success"],
                    "error_message": row["error_message"],
                    "execution_ms": row["execution_ms"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # ==================== Agent 状态 ====================

    async def save_agent_state(
        self,
        task_id: str,
        agent_name: str,
        state_type: str,
        state_data: Dict,
        step_number: Optional[int] = None,
    ) -> int:
        """保存 Agent 中间状态

        Args:
            task_id: 任务 ID
            agent_name: Agent 名称
            state_type: 状态类型（如 intent, reasoning, decision）
            state_data: 状态数据
            step_number: 步骤编号

        Returns:
            记录 ID
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO agent_states (task_id, agent_name, state_type, state_data, step_number)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING id
                """,
                uuid.UUID(task_id),
                agent_name,
                state_type,
                json.dumps(state_data),
                step_number,
            )
            return row["id"]

    async def get_agent_states(self, task_id: str, agent_name: Optional[str] = None) -> List[Dict]:
        """获取 Agent 状态记录

        Args:
            task_id: 任务 ID
            agent_name: Agent 名称（可选过滤）

        Returns:
            状态记录列表
        """
        query = "SELECT * FROM agent_states WHERE task_id = $1"
        params = [uuid.UUID(task_id)]

        if agent_name:
            query += " AND agent_name = $2"
            params.append(agent_name)

        query += " ORDER BY created_at ASC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                {
                    "id": row["id"],
                    "agent_name": row["agent_name"],
                    "state_type": row["state_type"],
                    "state_data": json.loads(row["state_data"]) if row["state_data"] else {},
                    "step_number": row["step_number"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # ==================== 工作流检查点 ====================

    async def save_checkpoint(
        self,
        task_id: str,
        current_node: str,
        current_step: int,
        graph_state: Optional[Dict] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        """保存工作流检查点

        Args:
            task_id: 任务 ID
            current_node: 当前节点名称
            current_step: 当前步骤编号
            graph_state: LangGraph 状态数据
            total_steps: 总步骤数
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO workflow_checkpoints (task_id, current_node, current_step, total_steps, graph_state)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (task_id) DO UPDATE SET
                    current_node = EXCLUDED.current_node,
                    current_step = EXCLUDED.current_step,
                    total_steps = EXCLUDED.total_steps,
                    graph_state = EXCLUDED.graph_state,
                    updated_at = NOW()
                """,
                uuid.UUID(task_id),
                current_node,
                current_step,
                total_steps,
                json.dumps(graph_state) if graph_state else None,
            )

    async def load_checkpoint(self, task_id: str) -> Optional[Dict]:
        """加载工作流检查点

        Args:
            task_id: 任务 ID

        Returns:
            检查点数据，不存在则返回 None
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflow_checkpoints WHERE task_id = $1",
                uuid.UUID(task_id),
            )
            if row:
                return {
                    "task_id": str(row["task_id"]),
                    "current_node": row["current_node"],
                    "current_step": row["current_step"],
                    "total_steps": row["total_steps"],
                    "graph_state": json.loads(row["graph_state"]) if row["graph_state"] else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        return None

    async def delete_checkpoint(self, task_id: str) -> None:
        """删除工作流检查点

        Args:
            task_id: 任务 ID
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM workflow_checkpoints WHERE task_id = $1",
                uuid.UUID(task_id),
            )

    # ==================== 任务恢复 ====================

    async def recover_task(self, task_id: str) -> Optional[Task]:
        """恢复中断的任务

        从数据库加载任务及其完整状态，包括:
        - 任务基本信息
        - 消息历史
        - 工具调用记录
        - Agent 状态
        - 工作流检查点

        Args:
            task_id: 任务 ID

        Returns:
            恢复的 Task 对象，不存在则返回 None
        """
        task = await self.load_task(task_id)
        if not task:
            return None

        # 加载额外的恢复信息到 metadata
        task.metadata = task.metadata or {}

        # 工具调用历史
        task.metadata["tool_calls"] = await self.get_tool_calls(task_id)

        # Agent 状态历史
        task.metadata["agent_states"] = await self.get_agent_states(task_id)

        # 工作流检查点
        checkpoint = await self.load_checkpoint(task_id)
        if checkpoint:
            task.metadata["checkpoint"] = checkpoint

        return task

    async def get_incomplete_tasks(self, limit: int = 50) -> List[Dict]:
        """获取未完成的任务列表

        用于服务重启后恢复中断的任务。

        Args:
            limit: 返回数量限制

        Returns:
            未完成任务信息列表
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT t.id, t.session_id, t.status, t.input_text, t.created_at, t.updated_at,
                       w.current_node, w.current_step
                FROM tasks t
                LEFT JOIN workflow_checkpoints w ON t.id = w.task_id
                WHERE t.status NOT IN ('completed', 'failed', 'canceled')
                ORDER BY t.created_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [
                {
                    "id": str(row["id"]),
                    "session_id": str(row["session_id"]) if row["session_id"] else None,
                    "status": row["status"],
                    "input_text": row["input_text"],
                    "current_node": row["current_node"],
                    "current_step": row["current_step"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    # ==================== 辅助方法 ====================

    def _extract_user_text(self, task: Task) -> str:
        """从任务中提取用户输入文本"""
        for msg in task.history:
            if msg.role == "user":
                for part in msg.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
        return ""

    def _extract_agent_text(self, task: Task) -> str:
        """从任务中提取最新的 Agent 响应文本"""
        for msg in reversed(task.history):
            if msg.role == "agent":
                for part in msg.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
        return ""

    def _serialize_part(self, part: Part) -> Dict:
        """序列化 Part 对象为字典"""
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        elif isinstance(part, FilePart):
            return {
                "type": "file",
                "file_uri": part.file_uri,
                "file_name": part.file_name,
                "mime_type": part.mime_type,
            }
        elif isinstance(part, DataPart):
            return {"type": "data", "data": part.data}
        else:
            return {
                "type": part.type.value if hasattr(part.type, "value") else str(part.type),
                "text": getattr(part, "text", None),
                "data": getattr(part, "data", None),
            }

    def _deserialize_part(self, data: Dict) -> Part:
        """从字典反序列化 Part 对象"""
        part_type = data.get("type", "text")

        if part_type == "text":
            return TextPart(text=data.get("text", ""))
        elif part_type == "file":
            return FilePart(
                file_uri=data.get("file_uri", ""),
                file_name=data.get("file_name"),
                mime_type=data.get("mime_type"),
            )
        elif part_type == "data":
            return DataPart(data=data.get("data", {}))
        else:
            # 默认作为文本处理
            return TextPart(text=data.get("text", ""))

    # ==================== User Preferences ====================

    async def save_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> None:
        """保存用户偏好 (upsert)

        Args:
            user_id: 用户标识
            category: 偏好类别 ('stock', 'fund', 'email', 'general')
            key: 偏好键名
            value: 偏好值（任意类型，将序列化为 JSON）
            confidence: 置信度 (0.0-1.0)
            source: 来源 ('explicit' 用户明确设置, 'inferred' 系统推断)
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_preferences (user_id, category, preference_key, preference_value, confidence, source)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                ON CONFLICT (user_id, category, preference_key) DO UPDATE SET
                    preference_value = EXCLUDED.preference_value,
                    confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source,
                    updated_at = NOW()
                """,
                user_id,
                category,
                key,
                json.dumps(value),
                confidence,
                source,
            )

    async def get_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取用户偏好列表

        Args:
            user_id: 用户标识
            category: 偏好类别（可选，不指定则返回所有）

        Returns:
            偏好列表，每项包含 category, key, value, confidence, source
        """
        await self._ensure_initialized()
        if category:
            query = """
                SELECT category, preference_key, preference_value, confidence, source, updated_at
                FROM user_preferences
                WHERE user_id = $1 AND category = $2
                ORDER BY updated_at DESC
            """
            params = [user_id, category]
        else:
            query = """
                SELECT category, preference_key, preference_value, confidence, source, updated_at
                FROM user_preferences
                WHERE user_id = $1
                ORDER BY category, updated_at DESC
            """
            params = [user_id]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                {
                    "category": row["category"],
                    "key": row["preference_key"],
                    "value": json.loads(row["preference_value"]) if row["preference_value"] else None,
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    async def delete_preference(
        self,
        user_id: str,
        category: str,
        key: str,
    ) -> bool:
        """删除用户偏好

        Args:
            user_id: 用户标识
            category: 偏好类别
            key: 偏好键名

        Returns:
            是否删除成功（True 表示有记录被删除）
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM user_preferences
                WHERE user_id = $1 AND category = $2 AND preference_key = $3
                """,
                user_id,
                category,
                key,
            )
            return result == "DELETE 1"

    async def get_preferences_as_context(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> str:
        """获取格式化的偏好上下文 (用于 LLM prompt 注入)

        Args:
            user_id: 用户标识
            category: 偏好类别（可选）

        Returns:
            格式化的偏好上下文字符串，可直接注入 LLM prompt
        """
        preferences = await self.get_preferences(user_id, category)

        if not preferences:
            return ""

        # 按类别分组
        grouped: Dict[str, List[Dict]] = {}
        for pref in preferences:
            cat = pref["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(pref)

        # 格式化输出
        lines = ["## User Preferences"]
        for cat, prefs in grouped.items():
            lines.append(f"\n### {cat.title()} Preferences")
            for pref in prefs:
                value = pref["value"]
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = str(value)
                lines.append(f"- **{pref['key']}**: {value_str}")

        return "\n".join(lines)

    # ==================== Fund Positions ====================

    async def save_fund_position(
        self,
        user_id: str,
        fund_code: str,
        fund_name: Optional[str] = None,
        shares_delta: float = 0,
        amount_delta: float = 0,
        nav: Optional[float] = None,
        operation: str = "buy",
    ) -> Dict[str, Any]:
        """保存或更新基金持仓

        Args:
            user_id: 用户标识
            fund_code: 基金代码
            fund_name: 基金名称（可选）
            shares_delta: 份额变化量（买入为正，卖出为负；set 操作时为绝对值）
            amount_delta: 金额变化量（买入为正，卖出为负；set 操作时为绝对值）
            nav: 当前净值（用于计算份额或成本）
            operation: 操作类型 'set' | 'buy' | 'sell' | 'clear'

        Returns:
            更新后的持仓信息
        """
        await self._ensure_initialized()
        from datetime import date

        today = date.today()

        async with self.pool.acquire() as conn:
            if operation == "clear":
                # 清仓操作：将份额和金额清零
                await conn.execute(
                    """
                    UPDATE fund_positions
                    SET shares = 0, cost_amount = 0, cost_nav = NULL,
                        last_trade_date = $3, updated_at = NOW()
                    WHERE user_id = $1 AND fund_code = $2
                    """,
                    user_id,
                    fund_code,
                    today,
                )
                return {
                    "user_id": user_id,
                    "fund_code": fund_code,
                    "operation": "clear",
                    "message": "持仓已清空",
                }

            # 获取当前持仓
            existing = await conn.fetchrow(
                """
                SELECT shares, cost_amount, cost_nav, first_buy_date
                FROM fund_positions
                WHERE user_id = $1 AND fund_code = $2
                """,
                user_id,
                fund_code,
            )

            if existing:
                old_shares = float(existing["shares"] or 0)
                old_cost_amount = float(existing["cost_amount"] or 0)
                old_cost_nav = float(existing["cost_nav"]) if existing["cost_nav"] else None
                first_buy_date = existing["first_buy_date"]
            else:
                old_shares = 0
                old_cost_amount = 0
                old_cost_nav = None
                first_buy_date = None

            # 对于 "set" 操作，直接设置绝对值而不是增量
            if operation == "set":
                # Set 操作：直接设置持仓值
                new_shares = shares_delta if shares_delta > 0 else 0
                new_cost_amount = amount_delta if amount_delta > 0 else 0

                # 如果只提供了金额，尝试用净值计算份额
                if new_shares == 0 and new_cost_amount > 0 and nav and nav > 0:
                    new_shares = new_cost_amount / nav

                # 计算成本净值
                if new_shares > 0 and new_cost_amount > 0:
                    new_cost_nav = new_cost_amount / new_shares
                else:
                    new_cost_nav = nav  # 如果无法计算，使用提供的净值

                # 设置首次买入日期（如果是新记录）
                if first_buy_date is None:
                    first_buy_date = today
            else:
                # 买入/卖出操作：使用增量计算
                # 计算份额变化
                if shares_delta == 0 and amount_delta != 0 and nav and nav > 0:
                    # 根据金额和净值计算份额
                    shares_delta = amount_delta / nav

                # 计算新持仓
                new_shares = max(0, old_shares + shares_delta)
                new_cost_amount = max(0, old_cost_amount + amount_delta)

                # 计算新成本净值
                if new_shares > 0 and new_cost_amount > 0:
                    new_cost_nav = new_cost_amount / new_shares
                else:
                    new_cost_nav = None

                # 首次买入日期
                if first_buy_date is None and operation == "buy" and new_shares > 0:
                    first_buy_date = today

            # Upsert 持仓
            await conn.execute(
                """
                INSERT INTO fund_positions (
                    user_id, fund_code, fund_name, shares, cost_amount,
                    cost_nav, first_buy_date, last_trade_date
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, fund_code) DO UPDATE SET
                    fund_name = COALESCE(EXCLUDED.fund_name, fund_positions.fund_name),
                    shares = EXCLUDED.shares,
                    cost_amount = EXCLUDED.cost_amount,
                    cost_nav = EXCLUDED.cost_nav,
                    first_buy_date = COALESCE(fund_positions.first_buy_date, EXCLUDED.first_buy_date),
                    last_trade_date = EXCLUDED.last_trade_date,
                    updated_at = NOW()
                """,
                user_id,
                fund_code,
                fund_name,
                new_shares,
                new_cost_amount,
                new_cost_nav,
                first_buy_date,
                today,
            )

            return {
                "user_id": user_id,
                "fund_code": fund_code,
                "fund_name": fund_name,
                "operation": operation,
                "shares_delta": shares_delta,
                "amount_delta": amount_delta,
                "new_shares": new_shares,
                "new_cost_amount": new_cost_amount,
                "new_cost_nav": new_cost_nav,
                "first_buy_date": str(first_buy_date) if first_buy_date else None,
                "last_trade_date": str(today),
            }

    async def get_fund_position(
        self,
        user_id: str,
        fund_code: str,
    ) -> Optional[Dict[str, Any]]:
        """获取单个基金持仓

        Args:
            user_id: 用户标识
            fund_code: 基金代码

        Returns:
            持仓信息字典，不存在则返回 None
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT fund_code, fund_name, shares, cost_amount, cost_nav,
                       first_buy_date, last_trade_date, created_at, updated_at
                FROM fund_positions
                WHERE user_id = $1 AND fund_code = $2
                """,
                user_id,
                fund_code,
            )
            if row:
                return {
                    "fund_code": row["fund_code"],
                    "fund_name": row["fund_name"],
                    "shares": float(row["shares"]) if row["shares"] else 0,
                    "cost_amount": float(row["cost_amount"]) if row["cost_amount"] else 0,
                    "cost_nav": float(row["cost_nav"]) if row["cost_nav"] else None,
                    "first_buy_date": str(row["first_buy_date"]) if row["first_buy_date"] else None,
                    "last_trade_date": str(row["last_trade_date"]) if row["last_trade_date"] else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        return None

    async def get_all_fund_positions(
        self,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """获取用户所有基金持仓

        Args:
            user_id: 用户标识

        Returns:
            持仓信息列表
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT fund_code, fund_name, shares, cost_amount, cost_nav,
                       first_buy_date, last_trade_date, created_at, updated_at
                FROM fund_positions
                WHERE user_id = $1 AND shares > 0
                ORDER BY cost_amount DESC
                """,
                user_id,
            )
            return [
                {
                    "fund_code": row["fund_code"],
                    "fund_name": row["fund_name"],
                    "shares": float(row["shares"]) if row["shares"] else 0,
                    "cost_amount": float(row["cost_amount"]) if row["cost_amount"] else 0,
                    "cost_nav": float(row["cost_nav"]) if row["cost_nav"] else None,
                    "first_buy_date": str(row["first_buy_date"]) if row["first_buy_date"] else None,
                    "last_trade_date": str(row["last_trade_date"]) if row["last_trade_date"] else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    async def clear_fund_position(
        self,
        user_id: str,
        fund_code: str,
    ) -> bool:
        """清仓指定基金

        Args:
            user_id: 用户标识
            fund_code: 基金代码

        Returns:
            是否成功清仓
        """
        await self._ensure_initialized()
        result = await self.save_fund_position(
            user_id=user_id,
            fund_code=fund_code,
            operation="clear",
        )
        return result.get("operation") == "clear"

    async def get_positions_as_context(
        self,
        user_id: str,
        fund_code: Optional[str] = None,
    ) -> str:
        """获取格式化的持仓上下文（用于 LLM prompt 注入）

        Args:
            user_id: 用户标识
            fund_code: 基金代码（可选，指定则只返回该基金持仓）

        Returns:
            格式化的持仓上下文字符串，可直接注入 LLM prompt
        """
        if fund_code:
            position = await self.get_fund_position(user_id, fund_code)
            if not position or position.get("shares", 0) <= 0:
                return ""
            positions = [position]
        else:
            positions = await self.get_all_fund_positions(user_id)

        if not positions:
            return ""

        lines = ["## User's Fund Positions"]

        for pos in positions:
            fund_name = pos.get("fund_name") or pos["fund_code"]
            shares = pos.get("shares", 0)
            cost_amount = pos.get("cost_amount", 0)
            cost_nav = pos.get("cost_nav")
            first_buy_date = pos.get("first_buy_date")
            last_trade_date = pos.get("last_trade_date")

            lines.append(f"\n### {fund_name} ({pos['fund_code']})")
            lines.append(f"- **Shares held**: {shares:.4f} units")
            lines.append(f"- **Total cost**: ¥{cost_amount:.2f}")
            if cost_nav:
                lines.append(f"- **Average cost NAV**: {cost_nav:.4f}")
            if first_buy_date:
                lines.append(f"- **First purchase date**: {first_buy_date}")
            if last_trade_date:
                lines.append(f"- **Last trade date**: {last_trade_date}")

        return "\n".join(lines)

    # ==================== Stock Positions ====================

    async def save_stock_position(
        self,
        user_id: str,
        stock_code: str,
        stock_name: Optional[str] = None,
        shares_delta: float = 0,
        amount_delta: float = 0,
        price: Optional[float] = None,
        operation: str = "buy",
    ) -> Dict[str, Any]:
        """保存或更新股票持仓

        Args:
            user_id: 用户标识
            stock_code: 股票代码
            stock_name: 股票名称（可选）
            shares_delta: 股数变化量（买入为正，卖出为负；set 操作时为绝对值）
            amount_delta: 金额变化量（买入为正，卖出为负；set 操作时为绝对值）
            price: 当前价格（用于计算股数或成本）
            operation: 操作类型 'set' | 'buy' | 'sell' | 'clear'

        Returns:
            更新后的持仓信息
        """
        await self._ensure_initialized()
        from datetime import date

        today = date.today()

        async with self.pool.acquire() as conn:
            if operation == "clear":
                # 清仓操作：将股数和金额清零
                await conn.execute(
                    """
                    UPDATE stock_positions
                    SET shares = 0, cost_amount = 0, cost_price = NULL,
                        last_trade_date = $3, updated_at = NOW()
                    WHERE user_id = $1 AND stock_code = $2
                    """,
                    user_id,
                    stock_code,
                    today,
                )
                return {
                    "user_id": user_id,
                    "stock_code": stock_code,
                    "operation": "clear",
                    "message": "持仓已清空",
                }

            # 获取当前持仓
            existing = await conn.fetchrow(
                """
                SELECT shares, cost_amount, cost_price, first_buy_date
                FROM stock_positions
                WHERE user_id = $1 AND stock_code = $2
                """,
                user_id,
                stock_code,
            )

            if existing:
                old_shares = float(existing["shares"] or 0)
                old_cost_amount = float(existing["cost_amount"] or 0)
                old_cost_price = float(existing["cost_price"]) if existing["cost_price"] else None
                first_buy_date = existing["first_buy_date"]
            else:
                old_shares = 0
                old_cost_amount = 0
                old_cost_price = None
                first_buy_date = None

            # 对于 "set" 操作，直接设置绝对值而不是增量
            if operation == "set":
                # Set 操作：直接设置持仓值
                new_shares = shares_delta if shares_delta > 0 else 0
                new_cost_amount = amount_delta if amount_delta > 0 else 0

                # 如果只提供了金额，尝试用价格计算股数
                if new_shares == 0 and new_cost_amount > 0 and price and price > 0:
                    new_shares = int(new_cost_amount / price / 100) * 100  # 股票通常100股为一手

                # 计算成本价
                if new_shares > 0 and new_cost_amount > 0:
                    new_cost_price = new_cost_amount / new_shares
                else:
                    new_cost_price = price  # 如果无法计算，使用提供的价格

                # 设置首次买入日期（如果是新记录）
                if first_buy_date is None:
                    first_buy_date = today
            else:
                # 买入/卖出操作：使用增量计算
                # 计算股数变化
                if shares_delta == 0 and amount_delta != 0 and price and price > 0:
                    # 根据金额和价格计算股数
                    shares_delta = amount_delta / price

                # 计算新持仓
                new_shares = max(0, old_shares + shares_delta)
                new_cost_amount = max(0, old_cost_amount + amount_delta)

                # 计算新成本价
                if new_shares > 0 and new_cost_amount > 0:
                    new_cost_price = new_cost_amount / new_shares
                else:
                    new_cost_price = None

                # 首次买入日期
                if first_buy_date is None and operation == "buy" and new_shares > 0:
                    first_buy_date = today

            # Upsert 持仓
            await conn.execute(
                """
                INSERT INTO stock_positions (
                    user_id, stock_code, stock_name, shares, cost_amount,
                    cost_price, first_buy_date, last_trade_date
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, stock_code) DO UPDATE SET
                    stock_name = COALESCE(EXCLUDED.stock_name, stock_positions.stock_name),
                    shares = EXCLUDED.shares,
                    cost_amount = EXCLUDED.cost_amount,
                    cost_price = EXCLUDED.cost_price,
                    first_buy_date = COALESCE(stock_positions.first_buy_date, EXCLUDED.first_buy_date),
                    last_trade_date = EXCLUDED.last_trade_date,
                    updated_at = NOW()
                """,
                user_id,
                stock_code,
                stock_name,
                new_shares,
                new_cost_amount,
                new_cost_price,
                first_buy_date,
                today,
            )

            return {
                "user_id": user_id,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "operation": operation,
                "shares_delta": shares_delta,
                "amount_delta": amount_delta,
                "new_shares": new_shares,
                "new_cost_amount": new_cost_amount,
                "new_cost_price": new_cost_price,
                "first_buy_date": str(first_buy_date) if first_buy_date else None,
                "last_trade_date": str(today),
            }

    async def get_stock_position(
        self,
        user_id: str,
        stock_code: str,
    ) -> Optional[Dict[str, Any]]:
        """获取单个股票持仓

        Args:
            user_id: 用户标识
            stock_code: 股票代码

        Returns:
            持仓信息字典，不存在则返回 None
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT stock_code, stock_name, shares, cost_amount, cost_price,
                       first_buy_date, last_trade_date, created_at, updated_at
                FROM stock_positions
                WHERE user_id = $1 AND stock_code = $2
                """,
                user_id,
                stock_code,
            )
            if row:
                return {
                    "stock_code": row["stock_code"],
                    "stock_name": row["stock_name"],
                    "shares": float(row["shares"]) if row["shares"] else 0,
                    "cost_amount": float(row["cost_amount"]) if row["cost_amount"] else 0,
                    "cost_price": float(row["cost_price"]) if row["cost_price"] else None,
                    "first_buy_date": str(row["first_buy_date"]) if row["first_buy_date"] else None,
                    "last_trade_date": str(row["last_trade_date"]) if row["last_trade_date"] else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        return None

    async def get_all_stock_positions(
        self,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """获取用户所有股票持仓

        Args:
            user_id: 用户标识

        Returns:
            持仓信息列表
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT stock_code, stock_name, shares, cost_amount, cost_price,
                       first_buy_date, last_trade_date, created_at, updated_at
                FROM stock_positions
                WHERE user_id = $1 AND shares > 0
                ORDER BY cost_amount DESC
                """,
                user_id,
            )
            return [
                {
                    "stock_code": row["stock_code"],
                    "stock_name": row["stock_name"],
                    "shares": float(row["shares"]) if row["shares"] else 0,
                    "cost_amount": float(row["cost_amount"]) if row["cost_amount"] else 0,
                    "cost_price": float(row["cost_price"]) if row["cost_price"] else None,
                    "first_buy_date": str(row["first_buy_date"]) if row["first_buy_date"] else None,
                    "last_trade_date": str(row["last_trade_date"]) if row["last_trade_date"] else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    async def clear_stock_position(
        self,
        user_id: str,
        stock_code: str,
    ) -> bool:
        """清仓指定股票

        Args:
            user_id: 用户标识
            stock_code: 股票代码

        Returns:
            是否成功清仓
        """
        await self._ensure_initialized()
        result = await self.save_stock_position(
            user_id=user_id,
            stock_code=stock_code,
            operation="clear",
        )
        return result.get("operation") == "clear"

    async def get_stock_positions_as_context(
        self,
        user_id: str,
        stock_code: Optional[str] = None,
    ) -> str:
        """获取格式化的股票持仓上下文（用于 LLM prompt 注入）

        Args:
            user_id: 用户标识
            stock_code: 股票代码（可选，指定则只返回该股票持仓）

        Returns:
            格式化的持仓上下文字符串，可直接注入 LLM prompt
        """
        if stock_code:
            position = await self.get_stock_position(user_id, stock_code)
            if not position or position.get("shares", 0) <= 0:
                return ""
            positions = [position]
        else:
            positions = await self.get_all_stock_positions(user_id)

        if not positions:
            return ""

        lines = ["## User's Stock Positions"]

        for pos in positions:
            stock_name = pos.get("stock_name") or pos["stock_code"]
            shares = pos.get("shares", 0)
            cost_amount = pos.get("cost_amount", 0)
            cost_price = pos.get("cost_price")
            first_buy_date = pos.get("first_buy_date")
            last_trade_date = pos.get("last_trade_date")

            lines.append(f"\n### {stock_name} ({pos['stock_code']})")
            lines.append(f"- **Shares held**: {shares:.0f} shares")
            lines.append(f"- **Total cost**: ¥{cost_amount:.2f}")
            if cost_price:
                lines.append(f"- **Average cost price**: ¥{cost_price:.2f}")
            if first_buy_date:
                lines.append(f"- **First purchase date**: {first_buy_date}")
            if last_trade_date:
                lines.append(f"- **Last trade date**: {last_trade_date}")

        return "\n".join(lines)

    # ==================== Stock Monitor Logs ====================

    async def save_stock_monitor_log(
        self,
        user_id: str,
        stock_code: str,
        stock_name: Optional[str],
        current_price: Optional[float],
        action: str,
        reason: Optional[str] = None,
        analysis_summary: Optional[str] = None,
        llm_request: Optional[str] = None,
        llm_response: Optional[Dict[str, Any]] = None,
        position_shares: Optional[float] = None,
        cost_price: Optional[float] = None,
        profit_loss_pct: Optional[float] = None,
        email_sent: bool = False,
    ) -> int:
        """保存股票监控日志

        Args:
            user_id: 用户标识
            stock_code: 股票代码
            stock_name: 股票名称
            current_price: 当前价格
            action: 操作建议（持有/买入/卖出/建仓）
            reason: 操作理由
            analysis_summary: 技术分析摘要
            llm_request: 发送给远端 LLM 的请求报文
            llm_response: 远端 LLM 的完整返回
            position_shares: 当时持仓股数
            cost_price: 当时成本价
            profit_loss_pct: 当时盈亏百分比
            email_sent: 是否发送了邮件

        Returns:
            日志记录 ID
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO stock_monitor_logs (
                    user_id, stock_code, stock_name, job_time, current_price,
                    action, reason, analysis_summary, llm_request, llm_response,
                    position_shares, cost_price, profit_loss_pct, email_sent
                )
                VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12, $13)
                RETURNING id
                """,
                user_id,
                stock_code,
                stock_name,
                current_price,
                action,
                reason,
                analysis_summary,
                llm_request,
                json.dumps(llm_response) if llm_response else None,
                position_shares,
                cost_price,
                profit_loss_pct,
                email_sent,
            )
            return row["id"]

    async def get_stock_monitor_logs(
        self,
        user_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取股票监控日志

        Args:
            user_id: 用户标识（可选过滤）
            stock_code: 股票代码（可选过滤）
            action: 操作类型（可选过滤）
            start_time: 开始时间（可选过滤）
            end_time: 结束时间（可选过滤）
            limit: 返回数量限制

        Returns:
            监控日志列表
        """
        await self._ensure_initialized()

        query = "SELECT * FROM stock_monitor_logs WHERE 1=1"
        params = []
        param_idx = 1

        if user_id:
            query += f" AND user_id = ${param_idx}"
            params.append(user_id)
            param_idx += 1

        if stock_code:
            query += f" AND stock_code = ${param_idx}"
            params.append(stock_code)
            param_idx += 1

        if action:
            query += f" AND action = ${param_idx}"
            params.append(action)
            param_idx += 1

        if start_time:
            query += f" AND job_time >= ${param_idx}"
            params.append(start_time)
            param_idx += 1

        if end_time:
            query += f" AND job_time <= ${param_idx}"
            params.append(end_time)
            param_idx += 1

        query += f" ORDER BY job_time DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "stock_code": row["stock_code"],
                    "stock_name": row["stock_name"],
                    "job_time": row["job_time"],
                    "current_price": float(row["current_price"]) if row["current_price"] else None,
                    "action": row["action"],
                    "reason": row["reason"],
                    "analysis_summary": row["analysis_summary"],
                    "llm_request": row["llm_request"],
                    "llm_response": json.loads(row["llm_response"]) if row["llm_response"] else None,
                    "position_shares": float(row["position_shares"]) if row["position_shares"] else None,
                    "cost_price": float(row["cost_price"]) if row["cost_price"] else None,
                    "profit_loss_pct": float(row["profit_loss_pct"]) if row["profit_loss_pct"] else None,
                    "email_sent": row["email_sent"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    async def update_stock_monitor_log_email_status(
        self,
        log_id: int,
        email_sent: bool = True,
    ) -> None:
        """更新监控日志的邮件发送状态

        Args:
            log_id: 日志记录 ID
            email_sent: 邮件是否已发送
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE stock_monitor_logs SET email_sent = $2 WHERE id = $1",
                log_id,
                email_sent,
            )

    async def get_stock_monitor_summary(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """获取股票监控统计摘要

        Args:
            user_id: 用户标识
            days: 统计天数

        Returns:
            统计摘要
        """
        await self._ensure_initialized()
        async with self.pool.acquire() as conn:
            # 统计各操作类型数量
            action_stats = await conn.fetch(
                """
                SELECT action, COUNT(*) as count
                FROM stock_monitor_logs
                WHERE user_id = $1 AND job_time >= NOW() - INTERVAL '%s days'
                GROUP BY action
                ORDER BY count DESC
                """ % days,
                user_id,
            )

            # 统计各股票监控次数
            stock_stats = await conn.fetch(
                """
                SELECT stock_code, stock_name, COUNT(*) as monitor_count,
                       SUM(CASE WHEN action IN ('买入', '卖出') THEN 1 ELSE 0 END) as alert_count
                FROM stock_monitor_logs
                WHERE user_id = $1 AND job_time >= NOW() - INTERVAL '%s days'
                GROUP BY stock_code, stock_name
                ORDER BY monitor_count DESC
                """ % days,
                user_id,
            )

            # 总计
            total = await conn.fetchrow(
                """
                SELECT COUNT(*) as total_monitors,
                       SUM(CASE WHEN email_sent THEN 1 ELSE 0 END) as emails_sent
                FROM stock_monitor_logs
                WHERE user_id = $1 AND job_time >= NOW() - INTERVAL '%s days'
                """ % days,
                user_id,
            )

            return {
                "days": days,
                "total_monitors": total["total_monitors"] if total else 0,
                "emails_sent": total["emails_sent"] if total else 0,
                "action_stats": {row["action"]: row["count"] for row in action_stats},
                "stock_stats": [
                    {
                        "stock_code": row["stock_code"],
                        "stock_name": row["stock_name"],
                        "monitor_count": row["monitor_count"],
                        "alert_count": row["alert_count"],
                    }
                    for row in stock_stats
                ],
            }
