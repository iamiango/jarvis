"""LangGraph API 适配层 - 让 Agent Chat UI 能与 Jarvis 通信

提供 LangGraph 兼容的 API 接口:
- POST /threads - 创建会话线程
- POST /threads/{thread_id}/runs/stream - 流式执行
- GET  /threads/{thread_id}/state - 获取线程状态
- GET  /assistants - 列出可用的 assistants
- GET  /assistants/{assistant_id} - 获取 assistant 详情

Agent Chat UI 会调用这些端点与后端 Agent 交互
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agents.jarvis import JarvisAgent
from ..config import agent_config
from ..streaming import StreamEvent, StreamEventType


# =============================================================================
# 数据模型
# =============================================================================

class ThreadCreateRequest(BaseModel):
    """创建线程请求"""
    metadata: Optional[Dict[str, Any]] = None


class ThreadCreateResponse(BaseModel):
    """创建线程响应"""
    thread_id: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class MessageContent(BaseModel):
    """消息内容"""
    type: str = "text"
    text: Optional[str] = None


class Message(BaseModel):
    """消息"""
    type: str  # "human" or "ai"
    content: str
    id: Optional[str] = None
    name: Optional[str] = None


def normalize_content(content: Any) -> str:
    """将消息内容标准化为字符串

    Agent Chat UI 可能发送以下格式:
    - 字符串: "hello"
    - 内容块列表: [{"type": "text", "text": "hello"}]

    Args:
        content: 消息内容，可能是字符串或列表

    Returns:
        标准化后的字符串内容
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 提取所有 text 类型内容块的文本
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    texts.append(block["text"])
                elif "text" in block:
                    texts.append(block["text"])
        return " ".join(texts) if texts else ""
    return str(content) if content else ""


class RunInput(BaseModel):
    """运行输入"""
    messages: List[Message]


class RunCreateRequest(BaseModel):
    """创建运行请求"""
    assistant_id: str
    input: RunInput
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    stream_mode: Optional[List[str]] = None


class AssistantInfo(BaseModel):
    """Assistant 信息"""
    assistant_id: str
    graph_id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class ThreadState(BaseModel):
    """线程状态"""
    values: Dict[str, Any]
    next: List[str]
    tasks: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


# =============================================================================
# 线程存储
# =============================================================================

@dataclass
class ThreadData:
    """线程数据"""
    thread_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


class ThreadStore:
    """线程存储管理"""

    def __init__(self):
        self._threads: Dict[str, ThreadData] = {}

    def create(self, metadata: Optional[Dict[str, Any]] = None) -> ThreadData:
        """创建新线程"""
        thread_id = str(uuid.uuid4())
        now = datetime.now()
        thread = ThreadData(
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._threads[thread_id] = thread
        return thread

    def get(self, thread_id: str) -> Optional[ThreadData]:
        """获取线程"""
        return self._threads.get(thread_id)

    def update(self, thread_id: str, **kwargs) -> Optional[ThreadData]:
        """更新线程"""
        thread = self._threads.get(thread_id)
        if thread:
            thread.updated_at = datetime.now()
            for key, value in kwargs.items():
                if hasattr(thread, key):
                    setattr(thread, key, value)
        return thread

    def add_message(self, thread_id: str, message: Dict[str, Any]) -> None:
        """添加消息到线程"""
        thread = self._threads.get(thread_id)
        if thread:
            thread.messages.append(message)
            thread.updated_at = datetime.now()

    def delete(self, thread_id: str) -> bool:
        """删除线程"""
        if thread_id in self._threads:
            del self._threads[thread_id]
            return True
        return False

    def list_all(self) -> List[ThreadData]:
        """列出所有线程"""
        return list(self._threads.values())


# =============================================================================
# LangGraph 适配器
# =============================================================================

class LangGraphAdapter:
    """LangGraph API 适配器

    将 Jarvis Agent 的功能适配为 LangGraph API 格式，
    使 Agent Chat UI 能够与 Jarvis 交互。
    """

    def __init__(
        self,
        jarvis: Optional[JarvisAgent] = None,
        sub_agent_urls: Optional[Dict[str, str]] = None,
    ):
        """初始化适配器

        Args:
            jarvis: 已初始化的 JarvisAgent 实例（可选）
            sub_agent_urls: 子 Agent URL 配置
        """
        self._jarvis = jarvis
        self._sub_agent_urls = sub_agent_urls or agent_config.sub_agents
        self._initialized = False
        self._thread_store = ThreadStore()

    async def initialize(self) -> None:
        """初始化 Jarvis Agent"""
        if self._initialized:
            return

        if not self._jarvis:
            self._jarvis = JarvisAgent(sub_agent_urls=self._sub_agent_urls)

        # 发现子 Agent
        await self._jarvis.discover_agents()
        self._initialized = True
        print("✅ LangGraph Adapter 初始化完成")

    async def close(self) -> None:
        """关闭资源"""
        if self._jarvis:
            await self._jarvis.close()

    def get_assistants(self) -> List[AssistantInfo]:
        """获取可用的 assistants 列表"""
        now = datetime.now().isoformat()
        return [
            AssistantInfo(
                assistant_id="jarvis",
                graph_id="jarvis",
                name="Jarvis",
                description=self._jarvis.description if self._jarvis else "Jarvis 智能助手",
                metadata={
                    "version": self._jarvis.version if self._jarvis else "2.0.0",
                    "capabilities": ["stock_analysis", "fund_analysis", "email"],
                },
                created_at=now,
                updated_at=now,
            )
        ]

    def get_assistant(self, assistant_id: str) -> Optional[AssistantInfo]:
        """获取指定 assistant"""
        assistants = self.get_assistants()
        for assistant in assistants:
            if assistant.assistant_id == assistant_id:
                return assistant
        return None

    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> ThreadData:
        """创建新的会话线程"""
        return self._thread_store.create(metadata)

    def get_thread(self, thread_id: str) -> Optional[ThreadData]:
        """获取线程"""
        return self._thread_store.get(thread_id)

    def get_thread_state(self, thread_id: str) -> Optional[ThreadState]:
        """获取线程状态"""
        thread = self._thread_store.get(thread_id)
        if not thread:
            return None

        return ThreadState(
            values={
                "messages": thread.messages,
            },
            next=[],
            tasks=[],
            metadata=thread.metadata,
            created_at=thread.created_at.isoformat(),
            updated_at=thread.updated_at.isoformat(),
        )

    async def stream_run(
        self,
        thread_id: str,
        assistant_id: str,
        input_messages: List[Message],
        config: Optional[Dict[str, Any]] = None,
        stream_mode: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """流式执行 Agent

        支持两种流式输出模式:
        1. "updates" - 输出每个步骤的执行进度
        2. "messages" - 输出 LLM 的 token 流

        Args:
            thread_id: 线程 ID
            assistant_id: Assistant ID
            input_messages: 输入消息列表
            config: 配置选项
            stream_mode: 流式模式列表 ["updates", "messages"]

        Yields:
            SSE 格式的事件数据
        """
        if not self._initialized:
            await self.initialize()

        thread = self._thread_store.get(thread_id)
        if not thread:
            yield self._format_sse_event("error", {"message": f"Thread {thread_id} not found"})
            return

        if not input_messages:
            yield self._format_sse_event("error", {"message": "No input messages provided"})
            return

        # 获取用户消息
        user_message = input_messages[-1].content
        run_id = str(uuid.uuid4())

        # 解析流式模式
        include_progress = True
        include_tokens = True
        if stream_mode:
            include_progress = "updates" in stream_mode
            include_tokens = "messages" in stream_mode

        # 保存用户消息到线程
        self._thread_store.add_message(thread_id, {
            "type": "human",
            "content": user_message,
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        })

        # 发送元数据事件
        yield self._format_sse_event("metadata", {"run_id": run_id})

        # 获取 user_id 和 session_id
        user_id = thread.metadata.get("user_id", "default")
        session_id = thread.metadata.get("session_id") or thread_id

        try:
            # 发送开始处理事件
            yield self._format_sse_event("messages/partial", [{
                "type": "ai",
                "content": "",
                "id": run_id,
            }])

            # 使用流式执行
            start_time = time.time()
            final_result = None
            accumulated_tokens = ""

            async for event in self._jarvis.arun_stream(
                user_message,
                user_id=user_id,
                session_id=session_id,
                include_progress=include_progress,
                include_tokens=include_tokens,
            ):
                # 将 StreamEvent 转换为 SSE 事件
                if event.type == StreamEventType.NODE_START and include_progress:
                    yield self._format_sse_event("progress/start", {
                        "node": event.progress.node_name if event.progress else "unknown",
                        "step": event.progress.step if event.progress else 0,
                        "message": event.progress.message if event.progress else "",
                    })

                elif event.type == StreamEventType.NODE_END and include_progress:
                    yield self._format_sse_event("progress/end", {
                        "node": event.progress.node_name if event.progress else "unknown",
                        "step": event.progress.step if event.progress else 0,
                        "duration_ms": event.progress.duration_ms if event.progress else None,
                        "message": event.progress.message if event.progress else "",
                        "output": event.progress.output if event.progress else None,
                    })

                elif event.type == StreamEventType.TOKEN and include_tokens:
                    accumulated_tokens += event.token.content if event.token else ""
                    yield self._format_sse_event("messages/partial", [{
                        "type": "ai",
                        "content": accumulated_tokens,
                        "id": run_id,
                    }])

                elif event.type == StreamEventType.COMPLETE:
                    final_result = event.result

                elif event.type == StreamEventType.ERROR:
                    yield self._format_sse_event("error", {"message": event.error})
                    self._thread_store.add_message(thread_id, {
                        "type": "ai",
                        "content": event.error or "未知错误",
                        "id": run_id,
                        "timestamp": datetime.now().isoformat(),
                        "is_error": True,
                    })
                    yield self._format_sse_event("end", None)
                    return

            elapsed = time.time() - start_time

            # 使用最终结果，如果没有则使用累积的 tokens
            result = final_result or accumulated_tokens or "处理完成"

            # 保存 AI 响应到线程
            ai_message = {
                "type": "ai",
                "content": result,
                "id": run_id,
                "timestamp": datetime.now().isoformat(),
                "response_metadata": {
                    "elapsed_time": elapsed,
                },
            }
            self._thread_store.add_message(thread_id, ai_message)

            # 发送完整消息事件
            yield self._format_sse_event("messages/complete", [ai_message])

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            yield self._format_sse_event("error", {"message": error_msg})

            # 保存错误消息
            self._thread_store.add_message(thread_id, {
                "type": "ai",
                "content": error_msg,
                "id": run_id,
                "timestamp": datetime.now().isoformat(),
                "is_error": True,
            })

        # 发送结束事件
        yield self._format_sse_event("end", None)

    def _format_sse_event(self, event: str, data: Any) -> str:
        """格式化 SSE 事件

        Args:
            event: 事件类型
            data: 事件数据

        Returns:
            SSE 格式的字符串
        """
        if data is None:
            return f"event: {event}\ndata: \n\n"
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# =============================================================================
# FastAPI 应用
# =============================================================================

def create_langgraph_app(
    jarvis: Optional[JarvisAgent] = None,
    sub_agent_urls: Optional[Dict[str, str]] = None,
) -> FastAPI:
    """创建 LangGraph API FastAPI 应用

    Args:
        jarvis: 已初始化的 JarvisAgent 实例（可选）
        sub_agent_urls: 子 Agent URL 配置

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title="Jarvis LangGraph API",
        description="LangGraph 兼容的 API 接口，供 Agent Chat UI 使用",
        version="1.0.0",
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 创建适配器
    adapter = LangGraphAdapter(jarvis=jarvis, sub_agent_urls=sub_agent_urls)

    # =========================================================================
    # 生命周期事件
    # =========================================================================

    @app.on_event("startup")
    async def startup():
        """应用启动时初始化"""
        await adapter.initialize()

    @app.on_event("shutdown")
    async def shutdown():
        """应用关闭时清理资源"""
        await adapter.close()

    # =========================================================================
    # API 路由
    # =========================================================================

    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "service": "jarvis-langgraph-api",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/info")
    async def get_info():
        """获取 API 信息 - Agent Chat UI 需要此端点来检测 API 可用性"""
        return {
            "version": "1.0.0",
            "name": "Jarvis LangGraph API",
            "description": "LangGraph 兼容的 API 接口，供 Agent Chat UI 使用",
        }

    # -------------------------------------------------------------------------
    # Assistants API
    # -------------------------------------------------------------------------

    @app.get("/assistants")
    async def list_assistants():
        """列出所有可用的 assistants"""
        assistants = adapter.get_assistants()
        return [a.model_dump() for a in assistants]

    @app.get("/assistants/{assistant_id}")
    async def get_assistant(assistant_id: str):
        """获取指定 assistant 的详情"""
        assistant = adapter.get_assistant(assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")
        return assistant.model_dump()

    # -------------------------------------------------------------------------
    # Threads API
    # -------------------------------------------------------------------------

    @app.post("/threads")
    async def create_thread(request: Request):
        """创建新的会话线程"""
        try:
            body = await request.json()
            metadata = body.get("metadata") if body else None
        except Exception:
            metadata = None
        thread = adapter.create_thread(metadata)
        return ThreadCreateResponse(
            thread_id=thread.thread_id,
            created_at=thread.created_at.isoformat(),
            metadata=thread.metadata,
        ).model_dump()

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str):
        """获取线程信息"""
        thread = adapter.get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
        return {
            "thread_id": thread.thread_id,
            "created_at": thread.created_at.isoformat(),
            "updated_at": thread.updated_at.isoformat(),
            "metadata": thread.metadata,
        }

    @app.get("/threads/{thread_id}/state")
    async def get_thread_state(thread_id: str):
        """获取线程状态"""
        state = adapter.get_thread_state(thread_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
        return state.model_dump()

    @app.get("/threads/{thread_id}/history")
    async def get_thread_history(thread_id: str):
        """获取线程历史消息"""
        thread = adapter.get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
        return {"messages": thread.messages}

    # -------------------------------------------------------------------------
    # Runs API (流式执行)
    # -------------------------------------------------------------------------

    @app.post("/threads/{thread_id}/runs/stream")
    async def stream_run(thread_id: str, request: Request):
        """流式执行 Agent

        这是 Agent Chat UI 的核心接口，使用 Server-Sent Events (SSE) 返回响应

        支持的 stream_mode:
        - "updates": 输出每个步骤的执行进度
        - "messages": 输出 LLM 的 token 流

        请求体示例:
        {
            "assistant_id": "jarvis",
            "input": {"messages": [{"type": "human", "content": "分析600588"}]},
            "stream_mode": ["updates", "messages"]
        }
        """
        body = await request.json()

        # 解析请求
        assistant_id = body.get("assistant_id", "jarvis")
        input_data = body.get("input", {})
        messages_data = input_data.get("messages", [])
        config = body.get("config")
        stream_mode = body.get("stream_mode", ["updates", "messages"])

        # 转换消息格式
        messages = [
            Message(
                type=m.get("type", "human"),
                content=normalize_content(m.get("content", "")),
                id=m.get("id"),
                name=m.get("name"),
            )
            for m in messages_data
        ]

        # 如果线程不存在，自动创建
        thread = adapter.get_thread(thread_id)
        if not thread:
            adapter.create_thread()
            # 手动设置 thread_id
            adapter._thread_store._threads[thread_id] = ThreadData(
                thread_id=thread_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        # 返回 SSE 流
        return StreamingResponse(
            adapter.stream_run(thread_id, assistant_id, messages, config, stream_mode),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/threads/{thread_id}/runs")
    async def create_run(thread_id: str, request: RunCreateRequest):
        """创建运行（非流式）

        同步执行 Agent 并返回结果
        """
        if not adapter._initialized:
            await adapter.initialize()

        thread = adapter.get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

        if not request.input.messages:
            raise HTTPException(status_code=400, detail="No input messages provided")

        # 获取用户消息
        user_message = request.input.messages[-1].content
        run_id = str(uuid.uuid4())

        # 保存用户消息
        adapter._thread_store.add_message(thread_id, {
            "type": "human",
            "content": user_message,
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        })

        try:
            # 调用 Jarvis 处理
            result = await adapter._jarvis.arun(user_message)

            # 保存 AI 响应
            ai_message = {
                "type": "ai",
                "content": result,
                "id": run_id,
                "timestamp": datetime.now().isoformat(),
            }
            adapter._thread_store.add_message(thread_id, ai_message)

            return {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": "completed",
                "output": {
                    "messages": [ai_message],
                },
            }

        except Exception as e:
            return {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": "failed",
                "error": str(e),
            }

    return app


# =============================================================================
# 独立运行入口
# =============================================================================

def main():
    """独立运行入口"""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Jarvis LangGraph API Server")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=2024, help="服务端口 (默认 2024)")
    parser.add_argument(
        "--stock-agent-url",
        default="http://localhost:8001",
        help="Stock Agent URL"
    )
    parser.add_argument(
        "--email-agent-url",
        default="http://localhost:8002",
        help="Email Agent URL"
    )
    parser.add_argument(
        "--fund-agent-url",
        default="http://localhost:8003",
        help="Fund Agent URL"
    )
    args = parser.parse_args()

    sub_agent_urls = {
        "stock_agent": args.stock_agent_url,
        "email_agent": args.email_agent_url,
        "fund_agent": args.fund_agent_url,
    }

    print("=" * 60)
    print("  Jarvis LangGraph API Server")
    print("=" * 60)
    print(f"  端口: {args.port}")
    print(f"  API 文档: http://{args.host}:{args.port}/docs")
    print(f"  健康检查: http://{args.host}:{args.port}/health")
    print("-" * 60)
    print("  Agent Chat UI 配置:")
    print(f"    NEXT_PUBLIC_API_URL=http://localhost:{args.port}")
    print(f"    NEXT_PUBLIC_ASSISTANT_ID=jarvis")
    print("=" * 60)

    app = create_langgraph_app(sub_agent_urls=sub_agent_urls)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
