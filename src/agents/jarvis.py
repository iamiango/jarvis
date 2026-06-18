"""Jarvis Orchestrator Agent - 调度 Agent

Jarvis 是一个调度 Agent，负责:
1. 发现并注册子 Agent
2. 基于 LLM 路由理解用户意图
3. 将任务路由到合适的子 Agent
4. 协调多 Agent 协作完成复杂任务
5. PII 防护（自动检测和处理敏感信息）
6. Human-in-the-Loop（敏感操作人工确认）
7. MCP 搜索支持（Tavily 网络搜索）

路由方式：
- 纯 LLM 路由 - 使用本地 Ollama 分析用户意图
- 子 Agent 内部 LLM 负责语义理解和实体提取（如股票代码）
"""
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence, TypedDict, TYPE_CHECKING
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
import argparse
import asyncio
import hashlib
import json
import time
from collections import OrderedDict

from .base import BaseAgent
from .registry import AgentRegistry
from ..a2a import (
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    A2AClient,
)
from ..streaming import StreamEvent, StreamEventType, NodeProgress, TokenChunk
from ..config import ollama_config, agent_config, email_config, qwen_config
from ..router import LLMRouter, LLMRouteResult
from ..security import (
    PIIGuard,
    PIIStrategy,
    HumanInTheLoop,
    ConfirmationHandler,
    ConfirmationResult,
    OperationCancelled,
)
from ..mcp import MultiServerMCPClient
from ..prompt_loader import load_prompt, PromptNames

if TYPE_CHECKING:
    from ..storage import CheckpointManager


class TavilySearchCache:
    """Tavily 搜索结果缓存

    - LRU 缓存，最多保存 100 条
    - TTL 过期时间，默认 5 分钟
    - 基于 query 的 hash 作为 key
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _hash_query(self, query: str) -> str:
        """生成 query 的 hash key"""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """获取缓存结果"""
        key = self._hash_query(query)
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            # 过期，删除
            del self._cache[key]
            return None

        # 移到最后（LRU）
        self._cache.move_to_end(key)
        return result

    def set(self, query: str, result: str) -> None:
        """设置缓存结果"""
        key = self._hash_query(query)

        # 如果已存在，更新并移到最后
        if key in self._cache:
            self._cache.move_to_end(key)

        self._cache[key] = (result, time.time())

        # 超过最大容量，删除最旧的
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def has_similar(self, query: str) -> bool:
        """检查是否有相似的缓存（精确匹配）"""
        return self.get(query) is not None

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }


class OrchestratorState(TypedDict):
    """Orchestrator 状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: int
    max_steps: int
    sub_agent_results: Dict[str, Any]


class JarvisAgent(BaseAgent):
    """Jarvis Orchestrator Agent - 智能调度中心

    基于 LLM 路由实现任务分发：
    1. 使用本地 Ollama 分析用户意图
    2. 路由到合适的子 Agent
    3. 子 Agent 内部 LLM 负责语义理解和实体提取
    """

    name = "jarvis"
    description = "Jarvis 智能助手 - 调度和协调多个专业 Agent 完成复杂任务"
    version = "4.0.0"  # 升级版本号，标记纯 LLM 路由架构

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        sub_agent_urls: Optional[Dict[str, str]] = None,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        pii_strategy: PIIStrategy = PIIStrategy.REDACT,
        email_confirmation: bool = True,
        confirmation_handler: Optional[ConfirmationHandler] = None,
        enable_mcp_search: bool = True,
    ):
        """初始化 Jarvis Orchestrator

        Args:
            model_name: LLM 模型名称（用于直接回答）
            base_url: Ollama 服务地址
            temperature: LLM 温度参数
            sub_agent_urls: 子 Agent URL 配置，如 {"stock_agent": "http://localhost:8001"}
            checkpoint_manager: 检查点管理器（可选，用于持久化状态）
            pii_strategy: PII 处理策略，默认为 REDACT
            email_confirmation: 是否在发送邮件前要求用户确认，默认为 True
            confirmation_handler: 自定义确认处理器（可选）
            enable_mcp_search: 是否启用 MCP 搜索功能，默认为 True
        """
        super().__init__(checkpoint_manager=checkpoint_manager)

        # LLM 配置（用于直接回答）
        self.model_name = model_name or ollama_config.model
        self.base_url = base_url or ollama_config.base_url
        self.temperature = temperature

        # 子 Agent 配置
        self._sub_agent_urls = sub_agent_urls or agent_config.sub_agents
        self._registry = AgentRegistry()
        self._a2a_client = A2AClient(timeout=600.0)

        # 初始化 LLM Router（用于路由决策）
        self._llm_router: Optional[LLMRouter] = None
        if qwen_config.api_key:
            self._llm_router = LLMRouter()
            print("  ✅ LLM Router 已启用 (本地 Ollama)")
        else:
            print("  ⚠️ LLM Router 未配置，将使用直接回答模式")

        # 初始化 LLM（用于直接回答）
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        # 初始化 PII 防护
        self._pii_guard = PIIGuard(strategy=pii_strategy)

        # 初始化 Human-in-the-Loop（邮件确认）
        self._hitl = HumanInTheLoop(
            handler=confirmation_handler,
            enabled=email_confirmation,
        )

        # MCP 配置
        self._enable_mcp_search = enable_mcp_search
        self._mcp_client: Optional[MultiServerMCPClient] = None
        self._mcp_initialized = False
        self._llm_with_tools: Optional[ChatOllama] = None

        # Tavily 搜索缓存 (TTL 5分钟，最多100条)
        self._search_cache = TavilySearchCache(max_size=100, ttl_seconds=300)

        # 单次对话搜索计数器（防止无限搜索）
        self._search_count_per_turn: int = 0
        self._max_searches_per_turn: int = 2  # 单次对话最多搜索2次

    def get_skills(self) -> List[AgentSkill]:
        """获取 Orchestrator 的元技能"""
        return [
            AgentSkill(
                id="orchestrate",
                name="任务调度",
                description="分析用户请求并调度合适的 Agent 执行任务",
                tags=["调度", "协调", "路由"],
                examples=[
                    "分析股票 000001 并把报告发送到 test@example.com",
                    "帮我查询贵州茅台的走势",
                    "发送邮件给 user@company.com",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ]

    async def discover_agents(self, urls: Optional[List[str]] = None) -> List[AgentCard]:
        """发现并注册子 Agent"""
        target_urls = urls or list(self._sub_agent_urls.values())
        discovered = []

        for url in target_urls:
            try:
                card = await self._registry.discover(url)
                discovered.append(card)
                print(f"✅ 发现 Agent: {card.name} @ {url}")
                print(f"   技能: {[s.name for s in card.skills]}")
            except Exception as e:
                print(f"⚠️ 发现 Agent 失败 ({url}): {e}")

        return discovered

    async def delegate_task(
        self,
        agent_name: str,
        message: str,
        wait_for_completion: bool = True,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Task:
        """将任务委派给子 Agent

        Args:
            agent_name: 子 Agent 名称
            message: 消息内容
            wait_for_completion: 是否等待完成
            task_id: 任务 ID（用于检查点）
            user_id: 用户 ID（用于多轮对话，传递给子 Agent）
            session_id: 会话 ID（用于区分同一用户的不同对话）
        """
        agent_url = self._registry.get_agent_url(agent_name)
        if not agent_url:
            raise ValueError(f"Agent '{agent_name}' 未注册")

        start_time = time.time()
        tool_input = {"agent_name": agent_name, "message": message}
        error_message = None
        success = True
        task = None

        try:
            # 传递 user_id 和 session_id 到 metadata，支持子 Agent 多轮对话
            metadata = {}
            if user_id:
                metadata["user_id"] = user_id
            if session_id:
                metadata["session_id"] = session_id
            metadata = metadata if metadata else None

            task = await self._a2a_client.send_task(
                url=agent_url,
                message=message,
                metadata=metadata,
            )

            if wait_for_completion and task.status.state not in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELED,
                TaskState.INPUT_REQUIRED,
            }:
                task = await self._a2a_client.wait_for_completion(
                    url=agent_url,
                    task_id=task.id,
                )

            success = task.status.state == TaskState.COMPLETED
            return task

        except Exception as e:
            success = False
            error_message = str(e)
            raise

        finally:
            execution_ms = int((time.time() - start_time) * 1000)
            if task_id and self._checkpoint_manager:
                tool_output = None
                if task:
                    tool_output = {
                        "task_id": task.id,
                        "status": task.status.state.value,
                    }
                await self.save_tool_call(
                    task_id=task_id,
                    tool_name=f"delegate_to_{agent_name}",
                    tool_input=tool_input,
                    tool_output=tool_output,
                    success=success,
                    error_message=error_message,
                    execution_ms=execution_ms,
                )

    async def process_task(self, task: Task) -> Task:
        """处理用户任务 - A2A 接口"""
        user_text = self._extract_text_from_task(task)

        if not user_text:
            return self._create_text_response(
                task,
                "请告诉我您需要什么帮助？",
                TaskState.INPUT_REQUIRED,
            )

        # 提取 user_id 和 session_id 用于多轮对话
        user_id = task.metadata.get("user_id") if task.metadata else None
        session_id = task.metadata.get("session_id") if task.metadata else None

        try:
            # PII 防护
            pii_result = self._pii_guard.process(user_text)
            if pii_result.has_pii:
                print(f"🛡️ PII 防护: 检测到 {len(pii_result.matches)} 个敏感信息")
                user_text = pii_result.processed_text

            await self.save_workflow_checkpoint(
                task_id=task.id,
                current_node="process_task_start",
                current_step=0,
                total_steps=3,
            )

            result = await self._arun_with_task_id(user_text, task.id, user_id=user_id, session_id=session_id)
            return self._create_text_response(task, result)

        except Exception as e:
            return self._create_text_response(
                task,
                f"处理请求失败: {str(e)}",
                TaskState.FAILED,
            )

    async def arun(self, user_input: str, max_steps: int = None, user_id: str = None, session_id: str = None) -> str:
        """异步运行 Orchestrator"""
        return await self._arun_with_task_id(user_input, task_id=None, max_steps=max_steps, user_id=user_id, session_id=session_id)

    async def _arun_with_task_id(
        self,
        user_input: str,
        task_id: Optional[str] = None,
        max_steps: int = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """异步运行 Orchestrator（带任务 ID 用于检查点）

        使用 LLM Router 分析用户意图，路由到合适的子 Agent。
        子 Agent 内部的 LLM 负责语义理解和实体提取。

        Args:
            user_input: 用户输入
            task_id: 任务 ID（用于检查点）
            max_steps: 最大步数（暂未使用）
            user_id: 用户 ID（用于多轮对话，传递给子 Agent）
            session_id: 会话 ID（用于区分同一用户的不同对话）
        """
        # 确保已发现子 Agent
        if len(self._registry) == 0:
            await self.discover_agents()

        # 获取已注册的 Agent 列表
        agents = self._registry.list_agents()

        # 使用 LLM Router 分析意图
        if not self._llm_router:
            # LLM Router 不可用，直接回答
            return await self._handle_direct_answer(user_input, {})

        print(f"🤖 LLM 路由器分析用户意图...")
        route_result = await self._llm_router.route(user_input, agents=agents)

        # 构建意图字典（用于检查点和后续处理）
        intent = {
            "need_agent": route_result.need_agent,
            "direct_answer": route_result.direct_answer,
            "confidence": route_result.confidence,
            "reasoning": route_result.reasoning,
            "steps": [
                {
                    "step": s.step,
                    "agent_name": s.agent_name,
                    "task_description": s.task_description,
                    "depends_on_previous": s.depends_on_previous,
                }
                for s in route_result.steps
            ],
            "match_type": "llm_router",
        }

        # 确定意图类型和目标 Agent
        if route_result.steps:
            first_step = route_result.steps[0]
            intent["type"] = self._infer_intent_type(first_step.agent_name, first_step.task_description)
            intent["agent_name"] = first_step.agent_name
        else:
            intent["type"] = "direct_answer"
            intent["agent_name"] = None

        print(f"📍 路由决策: {intent.get('type')} -> {intent.get('agent_name', 'N/A')}")
        print(f"   置信度: {route_result.confidence:.2f}")
        if route_result.reasoning:
            print(f"   推理: {route_result.reasoning[:80]}...")

        # 保存意图分析结果
        if task_id:
            await self.save_agent_state(
                task_id=task_id,
                state_type="intent",
                state_data=intent,
                step_number=1,
            )
            await self.save_workflow_checkpoint(
                task_id=task_id,
                current_node="intent_analyzed",
                current_step=1,
                graph_state={"intent": intent},
                total_steps=3,
            )

        # 如果不需要调用 Agent，直接回答
        if not route_result.need_agent:
            result = await self._handle_direct_answer(user_input, intent)
            if task_id:
                await self.save_workflow_checkpoint(
                    task_id=task_id,
                    current_node="completed",
                    current_step=3,
                    total_steps=3,
                )
            return result

        # 根据意图路由
        intent_type = intent.get("type")
        agent_name = intent.get("agent_name")

        if task_id:
            await self.save_agent_state(
                task_id=task_id,
                state_type="routing",
                state_data={"intent_type": intent_type, "agent_name": agent_name},
                step_number=2,
            )
            await self.save_workflow_checkpoint(
                task_id=task_id,
                current_node="routing",
                current_step=2,
                total_steps=3,
            )

        # 根据意图类型分发处理
        steps = intent.get("steps", [])
        if len(steps) > 1:
            # 多步任务
            result = await self._handle_compound_task(user_input, intent, task_id, user_id, session_id)
        elif intent_type == "email":
            result = await self._handle_email(user_input, intent, task_id, user_id, session_id)
        elif agent_name:
            # 单步 Agent 任务 - 直接转发用户原始输入
            result = await self._handle_generic_agent_call(agent_name, user_input, intent, task_id, user_id, session_id)
        else:
            result = await self._handle_direct_answer(user_input, intent)

        if task_id:
            await self.save_workflow_checkpoint(
                task_id=task_id,
                current_node="completed",
                current_step=3,
                total_steps=3,
            )

        return result

    def _infer_intent_type(self, agent_name: str, task_description: str) -> str:
        """从 Agent 名称和任务描述推断意图类型

        Args:
            agent_name: Agent 名称
            task_description: 任务描述

        Returns:
            意图类型字符串
        """
        if agent_name == "stock_agent":
            return "stock_analysis"
        elif agent_name == "fund_agent":
            return "fund_analysis"
        elif agent_name == "email_agent":
            return "email"
        else:
            return "unknown"

    def run(self, user_input: str, max_steps: int = None) -> str:
        """同步运行 Orchestrator"""
        return asyncio.get_event_loop().run_until_complete(
            self.arun(user_input, max_steps)
        )

    async def arun_stream(
        self,
        user_input: str,
        max_steps: int = None,
        user_id: str = None,
        session_id: str = None,
        include_progress: bool = True,
        include_tokens: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式运行 Orchestrator

        支持两种流式输出模式:
        1. include_progress=True: 输出每个步骤的执行进度
        2. include_tokens=True: 输出 LLM 的 token 流

        Args:
            user_input: 用户输入
            max_steps: 最大步数
            user_id: 用户 ID
            session_id: 会话 ID
            include_progress: 是否输出进度
            include_tokens: 是否输出 tokens

        Yields:
            StreamEvent: 流式事件
        """
        import time

        step = 0

        # 发送元数据
        yield StreamEvent(
            type=StreamEventType.METADATA,
            metadata={
                "user_id": user_id or "default",
                "session_id": session_id or "active",
                "agent": self.name,
            }
        )

        try:
            # 确保已发现子 Agent
            if len(self._registry) == 0:
                step += 1
                yield StreamEvent(
                    type=StreamEventType.NODE_START,
                    progress=NodeProgress(
                        node_name="discover_agents",
                        step=step,
                        status="running",
                        message="正在发现子 Agent...",
                    )
                )
                start_time = time.time()
                await self.discover_agents()
                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    progress=NodeProgress(
                        node_name="discover_agents",
                        step=step,
                        status="completed",
                        duration_ms=int((time.time() - start_time) * 1000),
                        message=f"发现 {len(self._registry)} 个子 Agent",
                    )
                )

            # 意图分析
            agents = self._registry.list_agents()
            step += 1
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                progress=NodeProgress(
                    node_name="llm_route",
                    step=step,
                    status="running",
                    message="正在使用 LLM 分析用户意图...",
                )
            )

            start_time = time.time()

            # 使用 LLM Router 分析意图
            if not self._llm_router:
                # LLM Router 不可用，直接回答
                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    progress=NodeProgress(
                        node_name="llm_route",
                        step=step,
                        status="completed",
                        duration_ms=int((time.time() - start_time) * 1000),
                        message="LLM Router 不可用，使用直接回答",
                    )
                )
                result = await self._handle_direct_answer(user_input, {})
                yield StreamEvent(
                    type=StreamEventType.COMPLETE,
                    result=result,
                    metadata={"total_steps": step},
                )
                return

            route_result = await self._llm_router.route(user_input, agents=agents)

            # 构建意图字典
            intent = {
                "need_agent": route_result.need_agent,
                "direct_answer": route_result.direct_answer,
                "confidence": route_result.confidence,
                "steps": [
                    {
                        "step": s.step,
                        "agent_name": s.agent_name,
                        "task_description": s.task_description,
                        "depends_on_previous": s.depends_on_previous,
                    }
                    for s in route_result.steps
                ],
                "match_type": "llm_router",
            }

            if route_result.steps:
                first_step = route_result.steps[0]
                intent["type"] = self._infer_intent_type(first_step.agent_name, first_step.task_description)
                intent["agent_name"] = first_step.agent_name
            else:
                intent["type"] = "direct_answer"
                intent["agent_name"] = None

            yield StreamEvent(
                type=StreamEventType.NODE_END,
                progress=NodeProgress(
                    node_name="llm_route",
                    step=step,
                    status="completed",
                    duration_ms=int((time.time() - start_time) * 1000),
                    output={
                        "type": intent.get("type"),
                        "agent_name": intent.get("agent_name"),
                        "confidence": route_result.confidence,
                    },
                    message=f"意图类型: {intent.get('type')} -> {intent.get('agent_name', 'N/A')}",
                )
            )

            # 如果不需要调用 Agent，直接回答
            if not route_result.need_agent:
                step += 1
                yield StreamEvent(
                    type=StreamEventType.NODE_START,
                    progress=NodeProgress(
                        node_name="direct_answer",
                        step=step,
                        status="running",
                        message="正在生成回答...",
                    )
                )

                start_time = time.time()
                result = await self._handle_direct_answer(user_input, intent)

                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    progress=NodeProgress(
                        node_name="direct_answer",
                        step=step,
                        status="completed",
                        duration_ms=int((time.time() - start_time) * 1000),
                        message="回答生成完成",
                    )
                )

                yield StreamEvent(
                    type=StreamEventType.COMPLETE,
                    result=result,
                    metadata={"total_steps": step},
                )
                return

            # 路由分发
            intent_type = intent.get("type")
            agent_name = intent.get("agent_name")
            steps = intent.get("steps", [])

            step += 1
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                progress=NodeProgress(
                    node_name="route_task",
                    step=step,
                    status="running",
                    message=f"正在路由到 {agent_name or intent_type}...",
                )
            )

            # 根据意图类型分发处理
            start_time = time.time()
            if len(steps) > 1:
                # 多步任务
                result = await self._handle_compound_task(user_input, intent, None, user_id, session_id)
            elif intent_type == "email":
                result = await self._handle_email(user_input, intent, None, user_id, session_id)
            elif agent_name:
                # 单步 Agent 任务 - 直接转发用户原始输入
                result = await self._handle_generic_agent_call(agent_name, user_input, intent, None, user_id, session_id)
            else:
                result = await self._handle_direct_answer(user_input, intent)

            yield StreamEvent(
                type=StreamEventType.NODE_END,
                progress=NodeProgress(
                    node_name="route_task",
                    step=step,
                    status="completed",
                    duration_ms=int((time.time() - start_time) * 1000),
                    message=f"任务执行完成",
                )
            )

            # 发送完成事件
            yield StreamEvent(
                type=StreamEventType.COMPLETE,
                result=result,
                metadata={"total_steps": step},
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error=str(e),
            )

    # =========================================================================
    # MCP 工具集成（带缓存和调用限制）
    # =========================================================================

    def _get_tavily_tool_description(self) -> str:
        """获取 Tavily 工具描述（从 md 文件加载）"""
        return load_prompt(PromptNames.TAVILY_TOOL_DESCRIPTION, max_calls=self._max_searches_per_turn)

    def _get_direct_answer_prompt_with_mcp(self) -> str:
        """获取带 MCP 工具的直接回答提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.JARVIS_DIRECT_ANSWER_MCP, max_search_calls=self._max_searches_per_turn)

    def _get_direct_answer_prompt(self) -> str:
        """获取普通直接回答提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.JARVIS_DIRECT_ANSWER)

    async def _ensure_mcp_initialized(self) -> bool:
        """确保 MCP 客户端已初始化并绑定工具到 LLM

        Returns:
            是否初始化成功
        """
        if self._mcp_initialized:
            return True

        if not self._enable_mcp_search:
            return False

        try:
            import os
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                print("⚠️ TAVILY_API_KEY 未配置，MCP 搜索不可用")
                return False

            # 创建 MCP 客户端（使用 async with 管理）
            self._mcp_client = MultiServerMCPClient()
            # 初始化 exit_stack
            if self._mcp_client._exit_stack is None:
                from contextlib import AsyncExitStack
                self._mcp_client._exit_stack = AsyncExitStack()
                await self._mcp_client._exit_stack.__aenter__()

            success = await self._mcp_client.add_stdio_server(
                name="tavily",
                command="npx",
                args=["-y", "tavily-mcp"],
                env={"TAVILY_API_KEY": api_key},
                timeout=90,
            )

            if not success:
                print("⚠️ Tavily MCP Server 连接失败，将使用普通 LLM 回答")
                self._mcp_client = None
                return False

            # 获取工具列表并绑定到 LLM（使用自定义描述）
            tools = self._mcp_client.get_tools()
            if tools:
                langchain_tools = self._convert_mcp_tools_to_langchain_with_limits(tools)
                self._llm_with_tools = self._llm.bind_tools(langchain_tools)
                print(f"✅ MCP 工具已绑定到 LLM（带调用限制）: {[t['name'] for t in tools]}")

            self._mcp_initialized = True
            return True

        except Exception as e:
            print(f"⚠️ MCP 初始化失败: {e}，将使用普通 LLM 回答")
            self._mcp_client = None
            return False

    def _convert_mcp_tools_to_langchain_with_limits(
        self,
        mcp_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """将 MCP 工具转换为 LangChain 工具格式（带使用限制描述）

        Args:
            mcp_tools: MCP 工具列表

        Returns:
            LangChain 格式的工具定义
        """
        langchain_tools = []

        for tool in mcp_tools:
            tool_name = tool.get("full_name", tool["name"])

            # 对 Tavily search 工具使用自定义描述
            if "tavily" in tool_name.lower() and "search" in tool["name"].lower():
                description = self._get_tavily_tool_description()
            else:
                description = f"[MCP] {tool.get('description', '')}"

            langchain_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                }
            }
            langchain_tools.append(langchain_tool)

        return langchain_tools

    async def _execute_mcp_tool_with_cache(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> tuple[str, bool]:
        """执行 MCP 工具（带缓存）

        Args:
            tool_name: 工具名称 (格式: mcp_{server}_{tool})
            arguments: 工具参数

        Returns:
            (工具执行结果, 是否来自缓存)
        """
        if not self._mcp_client:
            return "MCP 客户端未初始化", False

        # 检查是否是搜索工具
        is_search = "search" in tool_name.lower()
        query = arguments.get("query", "")

        if is_search and query:
            # 检查缓存
            cached_result = self._search_cache.get(query)
            if cached_result:
                print(f"   📦 缓存命中: {query[:30]}...")
                return cached_result, True

            # 检查调用次数限制
            if self._search_count_per_turn >= self._max_searches_per_turn:
                return f"⚠️ 已达到单次对话搜索上限（{self._max_searches_per_turn}次），请基于现有信息回答", False

        # 执行实际调用
        result = await self._mcp_client.call_tool_by_full_name(tool_name, arguments)

        if result["success"]:
            result_text = result["result"]

            # 缓存搜索结果
            if is_search and query:
                self._search_cache.set(query, result_text)
                self._search_count_per_turn += 1
                print(f"   💾 结果已缓存 (本轮搜索: {self._search_count_per_turn}/{self._max_searches_per_turn})")

            return result_text, False
        else:
            return f"工具调用失败: {result.get('error', '未知错误')}", False

    def _reset_search_counter(self) -> None:
        """重置单次对话的搜索计数器"""
        self._search_count_per_turn = 0

    # =========================================================================
    # 任务处理方法
    # =========================================================================

    async def _handle_direct_answer(self, text: str, intent: Dict[str, Any]) -> str:
        """直接回答用户问题（使用 LLM + MCP 工具，带缓存和限制）

        LLM 自动决定是否调用 MCP 工具（如 Tavily 搜索）
        如果 MCP 不可用，则使用普通 LLM 回答
        """
        # 如果有兜底响应，直接返回
        if intent.get("fallback_response"):
            return intent["fallback_response"]

        # 重置搜索计数器（新一轮对话）
        self._reset_search_counter()

        # 尝试初始化 MCP（失败也没关系）
        mcp_available = await self._ensure_mcp_initialized()

        # 选择 LLM（带工具或普通）
        if mcp_available and self._llm_with_tools:
            llm = self._llm_with_tools
            # 构建系统提示（强调搜索限制）
            system_prompt = self._get_direct_answer_prompt_with_mcp()
        else:
            llm = self._llm
            # 普通 LLM 提示
            system_prompt = self._get_direct_answer_prompt()

        messages = [
            HumanMessage(content=f"{system_prompt}\n\n用户问题: {text}")
        ]

        # 调用 LLM
        response = await llm.ainvoke(messages)

        # 检查是否有工具调用（仅当 MCP 可用时）
        if mcp_available and hasattr(response, 'tool_calls') and response.tool_calls:
            # 添加 AI 响应到消息历史
            messages.append(response)

            # 执行工具调用（最多执行 max_searches_per_turn 次搜索）
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", tool_name)

                print(f"🔧 LLM 调用工具: {tool_name}")
                print(f"   参数: {tool_args}")

                tool_result, from_cache = await self._execute_mcp_tool_with_cache(tool_name, tool_args)

                cache_indicator = "（缓存）" if from_cache else ""
                print(f"   ✅ 结果获取完成{cache_indicator}")

                # 添加工具结果到消息历史
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_id,
                ))

            # 再次调用 LLM 生成最终回答
            final_response = await self._llm.ainvoke(messages)
            return final_response.content

        # 没有工具调用，直接返回响应
        return response.content

    async def _handle_generic_agent_call(
        self,
        agent_name: str,
        text: str,
        intent: Dict[str, Any],
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """通用 Agent 调用

        直接转发用户原始输入给目标 Agent。
        Agent 内部的 LLM 负责语义理解和实体提取。
        """
        if agent_name not in self._registry:
            return f"Agent '{agent_name}' 未就绪，请检查服务状态"

        try:
            # 直接使用用户原始输入，让 Agent 内部 LLM 理解
            task = await self.delegate_task(agent_name, text, task_id=task_id, user_id=user_id, session_id=session_id)

            if task.status.state == TaskState.COMPLETED:
                return self._extract_agent_response(task)
            elif task.status.state == TaskState.INPUT_REQUIRED:
                return self._extract_agent_response(task) or "需要更多信息"
            else:
                return f"任务失败: {task.status.message}"

        except Exception as e:
            return f"调用 {agent_name} 失败: {str(e)}"

    async def _handle_email(
        self,
        text: str,
        intent: Dict[str, Any],
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """处理邮件发送请求（包含 Human-in-the-Loop 确认）"""
        email_addr = intent.get("email_address")
        if not email_addr:
            return "请提供收件人邮箱地址"

        if "email_agent" not in self._registry:
            return "邮件 Agent 未就绪，请检查服务状态"

        task_message = intent.get("task_message") or text
        email_subject = "（未指定主题）"
        email_content = task_message

        try:
            # Human-in-the-Loop 确认
            confirmation = await self._hitl.confirm(
                action="send_email",
                description="即将发送邮件，请确认以下信息",
                details={
                    "收件人": email_addr,
                    "主题": email_subject,
                    "内容预览": email_content[:500] if len(email_content) > 500 else email_content,
                },
                timeout_seconds=120.0,
            )

            if confirmation.result != ConfirmationResult.APPROVED:
                if confirmation.result == ConfirmationResult.TIMEOUT:
                    return "⏰ 确认超时，邮件发送已取消。"
                else:
                    return f"⛔ 邮件发送已取消: {confirmation.message or '用户取消'}"

            task = await self.delegate_task("email_agent", task_message, task_id=task_id, user_id=user_id, session_id=session_id)

            if task.status.state == TaskState.COMPLETED:
                return f"✅ 邮件已发送到 {email_addr}"
            elif task.status.state == TaskState.INPUT_REQUIRED:
                return self._extract_agent_response(task) or "需要更多邮件信息"
            else:
                return f"邮件发送失败: {task.status.message}"

        except OperationCancelled as e:
            return f"邮件发送已取消: {str(e)}"
        except Exception as e:
            return f"邮件发送请求失败: {str(e)}"

    async def _handle_compound_task(
        self,
        text: str,
        intent: Dict[str, Any],
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """处理复合任务 - 分步执行规则解析的任务队列"""
        steps = intent.get("steps", [])
        email_addr = intent.get("email_address") or email_config.default_to

        if not steps:
            return "无法解析复合任务步骤"

        results = []
        step_results = {}

        print(f"📋 复合任务: {len(steps)} 个步骤")

        for step_info in steps:
            step_num = step_info.get("step", 0)
            agent_name = step_info.get("agent_name")
            task_desc = step_info.get("task_description", "")
            use_previous_result = step_info.get("use_previous_result", False)

            print(f"\n🔹 步骤 {step_num}: {task_desc[:50]}...")
            print(f"   Agent: {agent_name}")

            # 检查 Agent 可用性
            if agent_name not in self._registry:
                error_msg = f"步骤 {step_num} 失败: Agent '{agent_name}' 未就绪"
                print(f"   ❌ {error_msg}")
                results.append(error_msg)
                continue

            # 构建任务消息
            task_message = task_desc
            if use_previous_result and step_num > 1:
                previous_result = step_results.get(step_num - 1)
                if previous_result:
                    if agent_name == "email_agent":
                        # 邮件任务：将前一步结果作为邮件内容
                        # 格式匹配 EmailAgent 的解析规则：主题是'xxx'，内容是'xxx'
                        task_message = f"""发送邮件到 {email_addr}，主题是'分析报告'，内容是'{previous_result[:2000]}'"""
                    else:
                        task_message = f"{task_desc}\n\n参考信息:\n{previous_result[:1000]}"

            # Human-in-the-Loop: 邮件发送前确认
            if agent_name == "email_agent":
                email_content = step_results.get(step_num - 1, task_desc) if use_previous_result else task_desc

                confirmation = await self._hitl.confirm(
                    action="send_email",
                    description=f"步骤 {step_num}: 即将发送邮件，请确认",
                    details={
                        "收件人": email_addr,
                        "主题": "分析报告",
                        "内容预览": str(email_content)[:500] + "..." if len(str(email_content)) > 500 else email_content,
                    },
                    timeout_seconds=120.0,
                )

                if confirmation.result != ConfirmationResult.APPROVED:
                    if confirmation.result == ConfirmationResult.TIMEOUT:
                        cancel_msg = "⏰ 确认超时，任务已取消"
                    else:
                        cancel_msg = f"⛔ 邮件发送已取消: {confirmation.message or '用户取消'}"
                    print(f"   {cancel_msg}")
                    results.append(cancel_msg)
                    return "\n".join(results)

            # 执行任务
            try:
                results.append(f"🔄 步骤 {step_num}: 正在执行...")

                task = await self.delegate_task(agent_name, task_message, task_id=task_id, user_id=user_id, session_id=session_id)

                if task.status.state == TaskState.COMPLETED:
                    step_result = self._extract_agent_response(task)
                    if step_result:
                        step_results[step_num] = step_result
                        print(f"   ✅ 步骤 {step_num} 完成")
                        results.append(f"✅ 步骤 {step_num} 完成")
                    else:
                        results.append(f"⚠️ 步骤 {step_num} 完成但未获取到结果")
                else:
                    error_msg = f"❌ 步骤 {step_num} 失败: {task.status.message}"
                    print(f"   {error_msg}")
                    results.append(error_msg)

                    # 如果后续步骤依赖此步骤，则中断
                    if any(s.get("depends_on_previous") and s.get("step", 0) > step_num for s in steps):
                        results.append("⛔ 由于步骤失败，后续依赖步骤已跳过")
                        break

            except Exception as e:
                error_msg = f"❌ 步骤 {step_num} 执行异常: {str(e)}"
                print(f"   {error_msg}")
                results.append(error_msg)

        # 汇总结果
        results.append("\n" + "=" * 40)
        results.append("📊 任务执行摘要")
        results.append("=" * 40)

        # 显示最后一个分析类任务的详细结果
        for step_num in sorted(step_results.keys(), reverse=True):
            result = step_results[step_num]
            if len(result) > 100:
                results.append(f"\n--- 步骤 {step_num} 详细结果 ---\n")
                results.append(result)
                break

        return "\n".join(results)

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _extract_agent_response(self, task: Task) -> Optional[str]:
        """从任务中提取 Agent 响应"""
        for msg in reversed(task.history):
            if msg.role == "agent":
                for part in msg.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
        return None

    async def close(self) -> None:
        """关闭资源"""
        await self._registry.close()
        await self._a2a_client.close()
        if self._checkpoint_manager:
            await self._checkpoint_manager.close()
        if self._mcp_client:
            try:
                await self._mcp_client.close_all()
                if self._mcp_client._exit_stack:
                    await self._mcp_client._exit_stack.aclose()
            except Exception:
                # 忽略 anyio 关闭时的已知问题
                pass
            self._mcp_client = None


# 保持向后兼容的别名
AgentState = OrchestratorState


def main():
    """Jarvis Orchestrator 独立启动入口"""
    parser = argparse.ArgumentParser(description="Jarvis Orchestrator - 智能调度中心")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
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
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启动交互式命令行模式"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="直接执行单次查询"
    )
    args = parser.parse_args()

    sub_agent_urls = {
        "stock_agent": args.stock_agent_url,
        "email_agent": args.email_agent_url,
        "fund_agent": args.fund_agent_url,
    }

    agent = JarvisAgent(sub_agent_urls=sub_agent_urls)

    if args.query:
        asyncio.run(run_single_query(agent, args.query))
        return

    if args.interactive:
        asyncio.run(run_interactive(agent))
        return

    print(f"启动 Jarvis Orchestrator (v{agent.version})...")
    print(f"  - 路由方式: 纯 LLM 路由 (本地 Ollama)")
    print(f"  - 地址: http://{args.host}:{args.port}")
    print(f"  - Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
    print(f"  - Stock Agent: {args.stock_agent_url}")
    print(f"  - Email Agent: {args.email_agent_url}")
    print(f"  - Fund Agent: {args.fund_agent_url}")

    asyncio.run(agent.discover_agents())
    agent.start_server(host=args.host, port=args.port)


async def run_single_query(agent: JarvisAgent, query: str):
    """执行单次查询"""
    try:
        await agent.discover_agents()
        result = await agent.arun(query)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60)
    finally:
        await agent.close()


async def run_interactive(agent: JarvisAgent):
    """交互式命令行模式"""
    print("\n" + "=" * 60)
    print(f"  Jarvis 智能助手 v{agent.version} - 交互模式")
    print("  路由方式: 纯 LLM 路由 (本地 Ollama)")
    print("=" * 60)
    print("\n正在发现子 Agent...")

    await agent.discover_agents()

    print("\n" + "-" * 60)
    print("可用命令:")
    print("  - 输入问题与 Jarvis 对话")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'agents' 查看可用 Agent")
    print("-" * 60 + "\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            if user_input.lower() == "agents":
                print("\n可用 Agent:")
                for agent_info in agent._registry.list_agents():
                    card = agent_info.card
                    print(f"  - {card.name}: {card.description}")
                    for skill in card.skills:
                        print(f"      • {skill.name}: {skill.description}")
                print()
                continue

            print("\nJarvis: ", end="", flush=True)
            try:
                result = await agent.arun(user_input)
                print(f"\n{result}\n")
            except Exception as e:
                print(f"\n错误: {e}\n")

    except KeyboardInterrupt:
        print("\n\n再见！")
    finally:
        await agent.close()


if __name__ == "__main__":
    main()
