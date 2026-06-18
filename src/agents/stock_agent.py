"""Stock Agent - 股票分析 Agent (LangGraph 架构)

专注于股票数据获取和分析报告生成。
包含 StockRetrieverSkill 和 StockReportGeneratorSkill。
支持用户偏好检测和保存。
支持用户持仓管理（设置/买入/卖出/清仓/查询）- 使用 LLM Tool Calling 自主判断。
支持通用市场行情查询（无需具体股票代码）。

v2.0.0 架构改进:
- 核心逻辑使用 LangGraph StateGraph 实现
- 一份代码同时支持 A2A 服务和 Studio 可视化
- 支持 LLM Tool Calling 模式
"""
import re
import logging
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence, TypedDict
from enum import Enum
import argparse
import asyncio

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .base import BaseAgent
from ..a2a import AgentSkill, Task, TaskState
from ..streaming import StreamEvent, StreamEventType, NodeProgress, TokenChunk, StreamHandler
from ..skills import StockRetrieverSkill, StockReportGeneratorSkill, skill_manager
from ..skills.user_preference import SaveUserPreferenceSkill, GetUserPreferencesSkill
from ..skills.stock_position import UpdateStockPositionSkill, GetStockPositionsSkill
from ..storage.checkpoint import CheckpointManager
from ..storage.langgraph_checkpoint import LangGraphCheckpointer, generate_thread_id
from ..config import postgres_config, ollama_config
from ..prompt_loader import load_prompt, PromptNames


logger = logging.getLogger(__name__)


# =============================================================================
# 状态定义
# =============================================================================

class StockAgentState(TypedDict):
    """StockAgent 图状态"""
    # 消息历史
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 用户输入
    user_input: str
    user_id: str

    # 解析结果
    stock_code: Optional[str]
    task_type: str  # stock_analysis / market_overview / position_update / position_query / preference_save / preference_query

    # 持仓操作
    position_operation: Optional[str]  # buy / sell / clear / query / set
    position_amount: Optional[float]
    position_shares: Optional[int]
    position_price: Optional[float]

    # 偏好设置
    preference_key: Optional[str]
    preference_value: Optional[str]
    preference_category: Optional[str]  # 偏好查询类别: stock / fund / general

    # 执行状态
    stock_data: Optional[str]  # 股票数据 JSON
    stock_name: Optional[str]
    index_data: Optional[Dict[str, Any]]  # 指数数据
    market_news: Optional[List[Dict[str, Any]]]  # 市场新闻

    # 最终响应
    final_response: Optional[str]

    # 状态
    status: str  # pending / parsing / llm_routing / fetching_data / generating_report / position_op / preference_op / completed / failed

    # 错误信息
    error: Optional[str]


class StockTaskType(Enum):
    """股票任务类型"""
    STOCK_ANALYSIS = "stock_analysis"      # 个股分析（需要股票代码）
    MARKET_OVERVIEW = "market_overview"    # 市场行情概览（无需代码）
    UNKNOWN = "unknown"                    # 无法识别


# 市场行情关键词（不需要股票代码）
MARKET_OVERVIEW_KEYWORDS = [
    # 大盘/指数相关
    "大盘", "指数", "A股", "行情", "市场", "整体",
    "上证", "深证", "创业板", "沪深", "两市",
    # 通用行情查询
    "今天怎么样", "今日行情", "市场走势", "大盘走势",
    "股市", "市况", "盘面", "总体", "整体情况",
    # 趋势询问
    "涨了吗", "跌了吗", "怎么走", "什么情况",
]

# 个股分析关键词（需要股票代码）
STOCK_ANALYSIS_KEYWORDS = [
    "分析", "股票", "个股", "技术分析", "K线",
    "MACD", "KDJ", "RSI", "均线", "支撑", "压力",
]

# 股票名称到代码映射（用于识别不带代码的个股请求）
STOCK_NAME_MAPPING = {
    "贵州茅台": "600519",
    "比亚迪": "002594",
    "宁德时代": "300750",
    "中国平安": "601318",
    "招商银行": "600036",
    "腾讯控股": "00700",
    "五粮液": "000858",
    "美的集团": "000333",
    "格力电器": "000651",
    "中国中免": "601888",
    "隆基绿能": "601012",
    "药明康德": "603259",
    "长江电力": "600900",
    "恒瑞医药": "600276",
    "海天味业": "603288",
    "茅台": "600519",  # 简称
    "平安": "601318",
    "招行": "600036",
    "腾讯": "00700",
}


class StockAgent(BaseAgent):
    """Stock Agent - 股票分析专家 (LangGraph 架构)

    技能:
    - stock_retriever: 获取股票数据和技术指标
    - stock_report_generator: 生成专业分析报告
    - save_user_preference: 保存用户偏好
    - update_stock_position: 更新股票持仓
    - get_stock_positions: 查询股票持仓

    支持两种任务类型:
    - 个股分析: 需要具体股票代码，进行技术分析和报告生成
    - 市场概览: 无需代码，获取大盘指数和市场新闻，生成市场分析
    - 持仓管理: 使用 LLM Tool Calling 自主判断操作类型

    架构特点:
    - 核心逻辑使用 LangGraph StateGraph 实现
    - 通过 self.graph 属性导出编译后的图供 Studio 使用
    - A2A process_task() 方法内部调用 graph.ainvoke()
    """

    name = "stock_agent"
    description = "股票分析 Agent - 提供 A 股数据获取、技术指标计算、专业投资分析报告和持仓管理，支持市场行情概览"
    version = "2.0.0"  # LangGraph 架构版本

    # 摘要配置
    SUMMARIZATION_TRIGGER = 10  # 消息数超过此值时触发摘要
    SUMMARIZATION_KEEP = 5      # 摘要后保留最近 N 条消息

    def _get_tool_calling_prompt(self) -> str:
        """获取 Tool Calling 系统提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.STOCK_AGENT_TOOL_CALLING)

    def _get_summarization_prompt(self, messages: str) -> str:
        """获取摘要提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.SUMMARIZATION, asset_type="股票", messages=messages)

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        enable_mcp_search: bool = True,
    ):
        """初始化 Stock Agent

        Args:
            checkpoint_manager: 检查点管理器，用于持久化用户偏好和持仓
            enable_mcp_search: 是否启用 MCP 搜索
        """
        super().__init__()
        self._checkpoint_manager = checkpoint_manager
        self._enable_mcp_search = enable_mcp_search
        self._ollama_llm: Optional[ChatOllama] = None
        self._langgraph_checkpointer = None  # 延迟初始化

        # 注册股票分析相关技能
        self._stock_retriever = StockRetrieverSkill()
        self._report_generator = StockReportGeneratorSkill()
        self._report_generator.set_checkpoint_manager(checkpoint_manager)
        self._save_preference_skill = SaveUserPreferenceSkill(checkpoint_manager)
        self._get_preferences_skill = GetUserPreferencesSkill(checkpoint_manager)

        # 注册持仓管理技能
        self._update_position_skill = UpdateStockPositionSkill(checkpoint_manager)
        self._get_positions_skill = GetStockPositionsSkill(checkpoint_manager)

        self.register_skill(self._stock_retriever)
        self.register_skill(self._report_generator)
        self.register_skill(self._save_preference_skill)
        self.register_skill(self._get_preferences_skill)
        self.register_skill(self._update_position_skill)
        self.register_skill(self._get_positions_skill)

        # MCP 搜索技能（用于获取市场新闻）
        self._mcp_search = None
        if enable_mcp_search:
            try:
                from ..skills.mcp_search import MCPSearchSkill
                self._mcp_search = MCPSearchSkill()
                logger.info("✅ MCP Search 已启用 (用于市场新闻)")
            except Exception as e:
                logger.warning(f"⚠️ MCP Search 不可用: {e}")

        # 构建 LLM 工具定义
        self._tools = self._build_tools()

        # 构建 LangGraph（不带 checkpointer，需要异步初始化后重建）
        self._graph: Optional[CompiledStateGraph] = None
        self._build_graph()

        logger.info(f"✅ StockAgent v{self.version} 初始化完成 (LangGraph 架构)")

    async def initialize(self) -> None:
        """异步初始化（首次使用前调用以启用多轮对话）

        初始化 LangGraph PostgresSaver 并重新编译图。
        如果不调用此方法，Agent 仍可正常工作，但不支持多轮对话历史。
        """
        if self._langgraph_checkpointer is not None:
            return  # 已初始化

        try:
            self._langgraph_checkpointer = await LangGraphCheckpointer.get_checkpointer()
            self._build_graph(checkpointer=self._langgraph_checkpointer)
            logger.info("✅ StockAgent 多轮对话支持已启用")
        except Exception as e:
            logger.warning(f"⚠️ 无法启用多轮对话支持: {e}")
            self._langgraph_checkpointer = None

    async def _ensure_checkpointer_connection(self) -> None:
        """确保 checkpointer 连接有效，必要时重新初始化

        在每次 process_task 调用前检查连接状态，如果连接已关闭则重新获取并重建图。
        """
        if self._langgraph_checkpointer is None:
            return  # 未启用多轮对话

        try:
            # 检查连接是否仍然有效
            # psycopg AsyncConnection 使用 .closed 属性 (bool)，不是 is_closed() 方法
            if hasattr(self._langgraph_checkpointer, 'conn') and self._langgraph_checkpointer.conn is not None:
                conn = self._langgraph_checkpointer.conn
                # 检查 closed 属性（psycopg3 的 AsyncConnection）
                is_closed = getattr(conn, 'closed', False)
                if is_closed:
                    logger.info("🔄 检测到 checkpointer 连接已关闭，重新初始化...")
                    # 获取新的 checkpointer（LangGraphCheckpointer 会处理重连）
                    self._langgraph_checkpointer = await LangGraphCheckpointer.get_checkpointer()
                    self._build_graph(checkpointer=self._langgraph_checkpointer)
                    logger.info("✅ StockAgent checkpointer 已重新初始化")
        except Exception as e:
            logger.warning(f"⚠️ 检查 checkpointer 连接状态失败: {e}")
            # 尝试重新初始化
            try:
                self._langgraph_checkpointer = await LangGraphCheckpointer.get_checkpointer()
                self._build_graph(checkpointer=self._langgraph_checkpointer)
                logger.info("✅ StockAgent checkpointer 已重新初始化")
            except Exception as e2:
                logger.error(f"❌ 重新初始化 checkpointer 失败: {e2}")

    def _get_ollama_llm(self) -> ChatOllama:
        """获取本地 Ollama LLM（懒加载）

        使用 langchain_ollama 的 ChatOllama，支持 Tool Calling
        """
        if self._ollama_llm is None:
            self._ollama_llm = ChatOllama(
                model=ollama_config.model,  # qwen3.5:9b
                base_url=ollama_config.base_url,
                temperature=0.3,  # 低温度以获得稳定的工具调用
            )
        return self._ollama_llm

    def _build_tools(self) -> List[Dict[str, Any]]:
        """构建 LLM Tool Calling 的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "update_stock_position",
                    "description": "更新股票持仓（买入、卖出、设置持仓、清仓）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_code": {
                                "type": "string",
                                "description": "股票代码（6位数字）"
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["set", "buy", "sell", "clear"],
                                "description": "操作类型：set(设置持仓), buy(买入), sell(卖出), clear(清仓)"
                            },
                            "amount": {
                                "type": "number",
                                "description": "交易金额（元）"
                            },
                            "shares": {
                                "type": "number",
                                "description": "交易股数"
                            },
                            "price": {
                                "type": "number",
                                "description": "买入/卖出单价（元）"
                            }
                        },
                        "required": ["stock_code", "operation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_positions",
                    "description": "查询股票持仓",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_code": {
                                "type": "string",
                                "description": "股票代码（可选，不指定则查询所有持仓）"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stock_analysis",
                    "description": "分析股票（获取行情数据和技术指标报告）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_code": {
                                "type": "string",
                                "description": "股票代码（6位数字）"
                            }
                        },
                        "required": ["stock_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "market_overview",
                    "description": "获取市场行情概览（大盘指数、市场整体情况）",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_user_preference",
                    "description": "保存用户投资偏好（风险偏好、投资周期、偏好板块、止损止盈等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": ["risk_tolerance", "analysis_horizon", "preferred_sectors", "stop_loss", "take_profit"],
                                "description": "偏好类型：risk_tolerance(风险偏好), analysis_horizon(投资周期), preferred_sectors(偏好板块), stop_loss(止损比例), take_profit(止盈比例)"
                            },
                            "value": {
                                "type": "string",
                                "description": "偏好值。risk_tolerance: conservative/moderate/aggressive; analysis_horizon: short-term/medium-term/long-term; preferred_sectors: tech/finance/healthcare/consumer/energy/manufacturing; stop_loss/take_profit: 百分比数字如'10'"
                            }
                        },
                        "required": ["key", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_user_preferences",
                    "description": "查询用户投资偏好设置（风险偏好、投资周期、偏好板块等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["stock", "fund", "general"],
                                "description": "偏好类别（可选）：stock(股票), fund(基金), general(通用)。不指定则返回所有偏好"
                            }
                        }
                    }
                }
            }
        ]

    def _build_graph(self, checkpointer=None) -> None:
        """构建 LangGraph StateGraph

        Args:
            checkpointer: 可选的 LangGraph checkpointer，用于多轮对话持久化
        """
        workflow = StateGraph(StockAgentState)

        # 添加节点 - 使用实例方法作为节点函数
        workflow.add_node("add_user_message", self._node_add_user_message)
        workflow.add_node("summarize_history", self._node_summarize_history)  # 新增：摘要节点
        workflow.add_node("parse", self._node_parse_input)
        workflow.add_node("llm_route", self._node_llm_routing)
        workflow.add_node("fetch_stock_data", self._node_fetch_stock_data)
        workflow.add_node("fetch_market_data", self._node_fetch_market_data)
        workflow.add_node("generate_report", self._node_generate_report)
        workflow.add_node("position_op", self._node_position_operation)
        workflow.add_node("preference_op", self._node_preference_operation)
        workflow.add_node("add_ai_response", self._node_add_ai_response)

        # 设置入口点 - 先添加用户消息到历史
        workflow.set_entry_point("add_user_message")

        # 添加边
        workflow.add_edge("add_user_message", "summarize_history")  # 添加消息后先检查是否需要摘要
        workflow.add_edge("summarize_history", "parse")  # 摘要后再解析
        workflow.add_edge("parse", "llm_route")

        # 条件路由: LLM 路由决策后
        workflow.add_conditional_edges(
            "llm_route",
            self._should_route_to_node,
            {
                "stock_data": "fetch_stock_data",
                "market_data": "fetch_market_data",
                "position": "position_op",
                "preference": "preference_op",
                "end": "add_ai_response",
            }
        )

        # 条件路由: 获取股票数据后
        workflow.add_conditional_edges(
            "fetch_stock_data",
            self._should_generate_report,
            {
                "report": "generate_report",
                "end": "add_ai_response",
            }
        )

        # 条件路由: 获取市场数据后
        workflow.add_conditional_edges(
            "fetch_market_data",
            self._should_generate_report,
            {
                "report": "generate_report",
                "end": "add_ai_response",
            }
        )

        # 终点边 - 所有路径最终都经过 add_ai_response
        workflow.add_edge("generate_report", "add_ai_response")
        workflow.add_edge("position_op", "add_ai_response")
        workflow.add_edge("preference_op", "add_ai_response")
        workflow.add_edge("add_ai_response", END)

        # 编译图（带或不带 checkpointer）
        self._graph = workflow.compile(checkpointer=checkpointer)
        if checkpointer:
            logger.info("✅ LangGraph StateGraph 已构建 (带多轮对话支持)")
        else:
            logger.info("✅ LangGraph StateGraph 已构建")

    @property
    def graph(self) -> CompiledStateGraph:
        """导出编译后的图供 LangGraph Studio 使用"""
        if self._graph is None:
            self._build_graph()
        return self._graph

    # =========================================================================
    # 节点函数
    # =========================================================================

    def _node_add_user_message(self, state: StockAgentState) -> Dict[str, Any]:
        """将用户输入添加到消息历史节点

        这是工作流的入口点，将用户消息添加到 messages 列表，
        以便 LangGraph checkpointer 持久化对话历史。
        """
        user_input = state.get("user_input", "")
        if user_input:
            return {
                "messages": [HumanMessage(content=user_input)]
            }
        return {}

    async def _node_summarize_history(self, state: StockAgentState) -> Dict[str, Any]:
        """摘要历史消息节点

        当消息数超过阈值时，使用 LLM 生成摘要，然后：
        1. 删除旧消息（保留最近 N 条）
        2. 在最前面插入摘要消息

        使用 LangGraph 的 RemoveMessage 机制来删除消息。
        """
        messages = state.get("messages", [])
        msg_count = len(messages)

        # 如果消息数未超过阈值，不做处理
        if msg_count <= self.SUMMARIZATION_TRIGGER:
            logger.debug(f"📝 消息数 {msg_count} 未超过阈值 {self.SUMMARIZATION_TRIGGER}，跳过摘要")
            return {}

        logger.info(f"📝 消息数 {msg_count} 超过阈值 {self.SUMMARIZATION_TRIGGER}，开始生成摘要...")

        try:
            # 要摘要的消息（除了最近 N 条）
            messages_to_summarize = messages[:-self.SUMMARIZATION_KEEP]
            messages_to_keep = messages[-self.SUMMARIZATION_KEEP:]

            # 格式化消息用于摘要
            formatted_messages = []
            for msg in messages_to_summarize:
                role = "用户" if isinstance(msg, HumanMessage) else "助手"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                formatted_messages.append(f"{role}: {content}")

            messages_text = "\n".join(formatted_messages)

            # 使用本地 qwen 生成摘要
            llm = self._get_ollama_llm()
            prompt = self._get_summarization_prompt(messages_text)

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            summary = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"✅ 摘要生成成功: {summary[:100]}...")

            # 构建返回值：删除旧消息 + 添加摘要
            # 使用 RemoveMessage 删除要摘要的消息
            remove_messages = [RemoveMessage(id=msg.id) for msg in messages_to_summarize if hasattr(msg, 'id') and msg.id]

            # 创建摘要消息（作为 SystemMessage 添加到历史开头）
            summary_message = SystemMessage(content=f"[对话历史摘要]\n{summary}")

            # 返回更新：删除旧消息 + 添加摘要
            # LangGraph 的 add_messages reducer 会处理 RemoveMessage
            return {
                "messages": remove_messages + [summary_message]
            }

        except Exception as e:
            logger.warning(f"⚠️ 摘要生成失败: {e}，跳过摘要")
            return {}

    def _node_add_ai_response(self, state: StockAgentState) -> Dict[str, Any]:
        """将 AI 响应添加到消息历史节点

        这是工作流的出口点，将最终响应添加到 messages 列表，
        以便 LangGraph checkpointer 持久化对话历史。
        """
        response = state.get("final_response", "")
        if response:
            return {
                "messages": [AIMessage(content=response)]
            }
        return {}

    def _node_parse_input(self, state: StockAgentState) -> Dict[str, Any]:
        """解析用户输入节点

        支持多轮对话：如果当前输入没有股票代码，会尝试从历史消息中查找。
        """
        user_input = state.get("user_input", "")

        if not user_input:
            return {
                "status": "failed",
                "error": "未收到有效输入",
                "final_response": "请提供您想要了解的内容，例如：\n- 今天A股行情怎么样\n- 分析股票 000001\n- 我买入了002364股票300元",
            }

        # 提取股票代码
        stock_code = self._extract_stock_symbol(user_input)
        if not stock_code:
            stock_code = self._extract_stock_name_code(user_input)

        # 如果当前输入没有股票代码，尝试从历史消息中查找（支持"它"、"这只股票"等指代）
        if not stock_code:
            stock_code = self._extract_stock_code_from_history(state.get("messages", []))
            if stock_code:
                logger.info(f"   从历史消息中推断股票代码: {stock_code}")

        logger.info(f"📋 解析用户输入: {user_input[:50]}...")
        logger.info(f"   提取股票代码: {stock_code}")

        return {
            "stock_code": stock_code,
            "status": "parsing",
        }

    async def _node_llm_routing(self, state: StockAgentState) -> Dict[str, Any]:
        """使用 LLM Tool Calling 进行路由决策节点

        注意：使用 state["messages"] 中的历史消息来支持多轮对话，
        让 LLM 能理解"它"、"这只股票"等指代。
        """
        user_input = state.get("user_input", "")
        stock_code = state.get("stock_code")  # 已从历史中提取的股票代码
        history_messages = state.get("messages", [])

        try:
            llm = self._get_ollama_llm()
            llm_with_tools = llm.bind_tools(self._tools)

            # 构建消息列表：系统提示 + 历史消息（已包含当前用户输入）
            messages = [SystemMessage(content=self._get_tool_calling_prompt())]

            # 如果从历史中提取到了股票代码，添加上下文提示
            # 这帮助 LLM 理解"它"、"这只股票"等指代
            if stock_code:
                context_hint = f"[系统提示: 根据对话历史，当前讨论的股票代码是 {stock_code}。如果用户使用\"它\"、\"这只股票\"等指代词，请使用此代码。]"
                messages.append(SystemMessage(content=context_hint))
                logger.info(f"   添加上下文提示: 当前股票代码 {stock_code}")

            # 添加历史消息（checkpointer 加载的 + 当前轮次的 HumanMessage）
            # 限制历史长度避免上下文过长
            recent_history = history_messages[-10:] if len(history_messages) > 10 else history_messages
            messages.extend(recent_history)

            logger.info(f"🤖 调用 LLM ({ollama_config.model}) 进行 Tool Calling...")
            logger.info(f"   消息历史: {len(recent_history)} 条")
            logger.debug(f"   工具列表: {[t['function']['name'] for t in self._tools]}")
            response = await llm_with_tools.ainvoke(messages)

            # 详细日志：查看 LLM 原始响应
            logger.info(f"📝 LLM 响应:")
            logger.info(f"   content: {response.content[:200] if response.content else 'None'}...")
            logger.info(f"   tool_calls: {response.tool_calls}")
            logger.info(f"   additional_kwargs: {response.additional_kwargs}")

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                logger.info(f"🔧 LLM 决定调用工具: {tool_name}")
                logger.info(f"   参数: {tool_args}")

                # 根据工具类型设置状态
                if tool_name == "stock_analysis":
                    return {
                        "task_type": "stock_analysis",
                        "stock_code": tool_args.get("stock_code", stock_code),
                        "status": "fetching_data",
                    }
                elif tool_name == "market_overview":
                    return {
                        "task_type": "market_overview",
                        "status": "fetching_data",
                    }
                elif tool_name == "update_stock_position":
                    return {
                        "task_type": "position_update",
                        "stock_code": tool_args.get("stock_code"),
                        "position_operation": tool_args.get("operation"),
                        "position_amount": tool_args.get("amount"),
                        "position_shares": tool_args.get("shares"),
                        "position_price": tool_args.get("price"),
                        "status": "position_op",
                    }
                elif tool_name == "get_stock_positions":
                    return {
                        "task_type": "position_query",
                        "stock_code": tool_args.get("stock_code"),
                        "position_operation": "query",
                        "status": "position_op",
                    }
                elif tool_name == "save_user_preference":
                    return {
                        "task_type": "preference_save",
                        "preference_key": tool_args.get("key"),
                        "preference_value": tool_args.get("value"),
                        "status": "preference_op",
                    }
                elif tool_name == "get_user_preferences":
                    return {
                        "task_type": "preference_query",
                        "preference_category": tool_args.get("category"),
                        "status": "preference_op",
                    }
                else:
                    return {
                        "status": "failed",
                        "error": f"未知工具: {tool_name}",
                        "final_response": f"内部错误：未知工具调用 {tool_name}",
                    }
            else:
                # LLM 没有调用工具
                content = response.content or "抱歉，我不太理解您的请求。请告诉我您想要：\n- 分析某只股票（如：分析600519）\n- 管理持仓（如：我买入了002364股票300元）\n- 查看市场行情"
                return {
                    "status": "completed",
                    "final_response": content,
                }

        except Exception as e:
            logger.error(f"❌ LLM Tool Calling 失败: {e}")
            # 降级：尝试规则匹配
            return await self._fallback_rule_routing(state)

    async def _fallback_rule_routing(self, state: StockAgentState) -> Dict[str, Any]:
        """降级路由：基于规则匹配"""
        user_input = state.get("user_input", "")
        stock_code = state.get("stock_code")

        task_type = self._classify_task_type(user_input)
        logger.info(f"📊 降级到规则匹配，任务类型: {task_type.value}")

        if task_type == StockTaskType.MARKET_OVERVIEW:
            return {
                "task_type": "market_overview",
                "status": "fetching_data",
            }
        elif task_type == StockTaskType.STOCK_ANALYSIS:
            if not stock_code:
                return {
                    "status": "failed",
                    "error": "未识别股票代码",
                    "final_response": "未能识别股票代码，请提供有效的 A 股代码（如 000001、600519）",
                }
            return {
                "task_type": "stock_analysis",
                "status": "fetching_data",
            }
        else:
            return {
                "status": "failed",
                "error": "无法识别任务类型",
                "final_response": "抱歉，我无法理解您的请求。请尝试：\n- 分析股票 000001\n- 今天A股行情怎么样\n- 我买入了002364股票300元",
            }

    async def _node_fetch_stock_data(self, state: StockAgentState) -> Dict[str, Any]:
        """获取股票数据节点"""
        stock_code = state.get("stock_code", "")

        if not stock_code:
            return {
                "status": "failed",
                "error": "缺少股票代码",
                "final_response": "请提供股票代码进行分析",
            }

        logger.info(f"📊 正在获取股票 {stock_code} 的数据...")

        try:
            result = await self._stock_retriever.execute(symbol=stock_code, days=60)

            if not result.success:
                return {
                    "status": "failed",
                    "error": result.error,
                    "final_response": f"获取股票数据失败: {result.error}",
                }

            logger.info(f"   ✅ 数据获取成功")

            return {
                "stock_data": result.result,
                "status": "generating_report",
            }

        except Exception as e:
            logger.error(f"❌ 获取股票数据异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"获取股票数据异常: {str(e)}",
            }

    async def _node_fetch_market_data(self, state: StockAgentState) -> Dict[str, Any]:
        """获取市场数据节点"""
        logger.info("📊 正在获取市场指数数据...")

        try:
            # 获取指数数据
            index_data = await self._stock_retriever.get_index_data()
            logger.info(f"📈 获取到指数数据: {list(index_data.get('indices', {}).keys())}")

            # 获取市场新闻
            market_news = []
            if self._mcp_search:
                try:
                    news_result = await self._mcp_search.execute(
                        query="A股 股市 今日行情",
                        max_results=5,
                    )
                    if news_result.success and news_result.result:
                        market_news = news_result.result if isinstance(news_result.result, list) else []
                        logger.info(f"📰 获取到 {len(market_news)} 条市场新闻")
                except Exception as e:
                    logger.warning(f"⚠️ 获取市场新闻失败: {e}")

            return {
                "index_data": index_data,
                "market_news": market_news,
                "status": "generating_report",
            }

        except Exception as e:
            logger.error(f"❌ 获取市场数据异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"获取市场数据异常: {str(e)}",
            }

    async def _node_generate_report(self, state: StockAgentState) -> Dict[str, Any]:
        """生成分析报告节点"""
        task_type = state.get("task_type", "")
        user_id = state.get("user_id", "default")

        if task_type == "market_overview":
            # 生成市场报告
            index_data = state.get("index_data", {})
            market_news = state.get("market_news", [])
            report = await self._generate_market_report(index_data, market_news, user_id)
        else:
            # 生成个股报告
            stock_code = state.get("stock_code", "")
            stock_data = state.get("stock_data", "")
            logger.info(f"📝 正在生成股票 {stock_code} 的分析报告...")

            try:
                result = await self._report_generator.execute(
                    stock_data_json=stock_data,
                    user_id=user_id,
                )

                if not result.success:
                    return {
                        "status": "failed",
                        "error": result.error,
                        "final_response": f"生成报告失败: {result.error}",
                    }

                report = result.result
                logger.info(f"   ✅ 报告生成成功 ({len(report)} 字符)")

            except Exception as e:
                logger.error(f"❌ 生成报告异常: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "final_response": f"生成报告异常: {str(e)}",
                }

        return {
            "final_response": report,
            "status": "completed",
        }

    async def _node_position_operation(self, state: StockAgentState) -> Dict[str, Any]:
        """处理持仓操作节点"""
        operation = state.get("position_operation", "")
        stock_code = state.get("stock_code")
        amount = state.get("position_amount")
        shares = state.get("position_shares")
        price = state.get("position_price")
        user_id = state.get("user_id", "default")

        logger.info(f"💼 执行持仓操作: {operation} {stock_code or '全部'}")

        try:
            if operation == "query":
                # 查询持仓
                result = await self._get_positions_skill.execute(
                    user_id=user_id,
                    stock_code=stock_code,
                    checkpoint_manager=self._checkpoint_manager,
                )

                if result.success:
                    message = result.result.get("message", "查询完成")
                    return {
                        "final_response": message,
                        "status": "completed",
                    }
                else:
                    return {
                        "status": "failed",
                        "error": result.error,
                        "final_response": f"查询持仓失败: {result.error}",
                    }

            elif operation in ("set", "buy", "sell", "clear"):
                if not stock_code:
                    return {
                        "status": "failed",
                        "error": "缺少股票代码",
                        "final_response": "请提供股票代码，例如：我买入了002364股票300元",
                    }

                # 检查是否需要询问价格
                if operation in ("buy", "sell", "set"):
                    if shares is not None and amount is None and price is None:
                        return {
                            "status": "failed",
                            "error": "需要买入价格",
                            "final_response": f"请问您 {stock_code} 的{'买入' if operation == 'buy' else '卖出' if operation == 'sell' else '持仓'}单价是多少元？\n（例如：15.5元）",
                        }
                    if shares is not None and price is not None and amount is None:
                        amount = shares * price

                result = await self._update_position_skill.execute(
                    user_id=user_id,
                    stock_code=stock_code,
                    operation=operation,
                    amount=amount,
                    shares=shares,
                    price=price,
                    checkpoint_manager=self._checkpoint_manager,
                )

                if result.success:
                    message = result.result.get("message", "持仓更新成功")
                    return {
                        "final_response": message,
                        "status": "completed",
                    }
                else:
                    return {
                        "status": "failed",
                        "error": result.error,
                        "final_response": f"更新持仓失败: {result.error}",
                    }

            return {
                "status": "failed",
                "error": "未知操作",
                "final_response": "无法识别的持仓操作",
            }

        except Exception as e:
            logger.error(f"❌ 持仓操作异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"持仓操作异常: {str(e)}",
            }

    async def _node_preference_operation(self, state: StockAgentState) -> Dict[str, Any]:
        """处理偏好设置/查询节点"""
        task_type = state.get("task_type")
        user_id = state.get("user_id", "default")

        # 查询偏好
        if task_type == "preference_query":
            return await self._handle_preference_query(state, user_id)

        # 保存偏好
        return await self._handle_preference_save(state, user_id)

    async def _handle_preference_query(self, state: StockAgentState, user_id: str) -> Dict[str, Any]:
        """处理偏好查询"""
        category = state.get("preference_category")

        logger.info(f"🔍 查询用户偏好: user={user_id}, category={category or 'all'}")

        try:
            result = await self._get_preferences_skill.execute(
                user_id=user_id,
                category=category,
                as_context=False,  # 返回结构化数据
                checkpoint_manager=self._checkpoint_manager,
            )

            if result.success:
                if not result.result:
                    message = "📋 您目前还没有设置任何投资偏好。\n\n您可以告诉我您的偏好，例如：\n- 我偏好保守型投资\n- 我喜欢短线交易\n- 我关注科技板块"
                else:
                    # 格式化偏好显示
                    message = self._format_preferences_display(result.result)
                return {
                    "final_response": message,
                    "status": "completed",
                }
            else:
                return {
                    "status": "failed",
                    "error": result.error,
                    "final_response": f"查询偏好失败: {result.error}",
                }

        except Exception as e:
            logger.error(f"❌ 查询偏好异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"查询偏好异常: {str(e)}",
            }

    def _format_preferences_display(self, preferences: List[Dict[str, Any]]) -> str:
        """格式化偏好显示"""
        key_descriptions = {
            "risk_tolerance": "风险偏好",
            "analysis_horizon": "投资周期",
            "preferred_sectors": "偏好板块",
            "stop_loss": "止损比例",
            "take_profit": "止盈比例",
        }
        value_descriptions = {
            "risk_tolerance": {
                "conservative": "保守型（低风险）",
                "moderate": "稳健型（中等风险）",
                "aggressive": "激进型（高风险）",
            },
            "analysis_horizon": {
                "short-term": "短线交易",
                "medium-term": "中线波段",
                "long-term": "长线投资",
            },
            "preferred_sectors": {
                "tech": "科技板块",
                "finance": "金融板块",
                "healthcare": "医药板块",
                "consumer": "消费板块",
                "energy": "能源板块",
                "manufacturing": "制造板块",
            },
        }

        lines = ["📋 **您的投资偏好设置**\n"]
        for pref in preferences:
            key = pref.get("key", "")
            value = pref.get("value", "")
            key_desc = key_descriptions.get(key, key)
            value_desc = value_descriptions.get(key, {}).get(value, value)
            lines.append(f"- {key_desc}: {value_desc}")

        lines.append("\n💡 您可以随时告诉我来更新偏好。")
        return "\n".join(lines)

    async def _handle_preference_save(self, state: StockAgentState, user_id: str) -> Dict[str, Any]:
        """处理偏好保存"""
        key = state.get("preference_key")
        value = state.get("preference_value")

        if not key or not value:
            return {
                "status": "failed",
                "error": "缺少偏好设置参数",
                "final_response": "请提供完整的偏好设置，例如：\n- 我偏好保守型投资\n- 我喜欢短线交易",
            }

        logger.info(f"💾 保存用户偏好: {key}={value}")

        try:
            result = await self._save_preference_skill.execute(
                user_id=user_id,
                category="stock",
                key=key,
                value=value,
                confidence=0.9,
                source="user_explicit",
                checkpoint_manager=self._checkpoint_manager,
            )

            if result.success:
                # 构建友好回复
                preference_descriptions = {
                    "risk_tolerance": {
                        "conservative": "保守型（低风险）",
                        "moderate": "稳健型（中等风险）",
                        "aggressive": "激进型（高风险）",
                    },
                    "analysis_horizon": {
                        "short-term": "短线交易",
                        "medium-term": "中线波段",
                        "long-term": "长线投资",
                    },
                    "preferred_sectors": {
                        "tech": "科技板块",
                        "finance": "金融板块",
                        "healthcare": "医药板块",
                        "consumer": "消费板块",
                        "energy": "能源板块",
                        "manufacturing": "制造板块",
                    },
                }
                value_desc = preference_descriptions.get(key, {}).get(value, value)
                key_desc = {
                    "risk_tolerance": "风险偏好",
                    "analysis_horizon": "投资周期",
                    "preferred_sectors": "偏好板块",
                    "stop_loss": "止损比例",
                    "take_profit": "止盈比例",
                }.get(key, key)

                message = f"✅ 已保存您的{key_desc}偏好：{value_desc}\n\n后续分析将根据您的偏好进行个性化调整。"
                return {
                    "final_response": message,
                    "status": "completed",
                }
            else:
                return {
                    "status": "failed",
                    "error": result.error,
                    "final_response": f"保存偏好失败: {result.error}",
                }

        except Exception as e:
            logger.error(f"❌ 保存偏好异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"保存偏好异常: {str(e)}",
            }

    # =========================================================================
    # 条件路由函数
    # =========================================================================

    def _should_route_to_node(self, state: StockAgentState) -> str:
        """路由决策: 根据状态决定下一个节点"""
        status = state.get("status", "")
        task_type = state.get("task_type", "")

        if status == "failed" or status == "completed":
            return "end"
        elif status == "position_op":
            return "position"
        elif status == "preference_op":
            return "preference"
        elif status == "fetching_data":
            if task_type == "market_overview":
                return "market_data"
            else:
                return "stock_data"
        else:
            return "end"

    def _should_generate_report(self, state: StockAgentState) -> str:
        """获取数据后: 生成报告还是结束"""
        status = state.get("status", "")

        if status == "failed" or status == "completed":
            return "end"
        elif status == "generating_report":
            return "report"
        else:
            return "end"

    def get_skills(self) -> List[AgentSkill]:
        """获取 Agent 技能列表"""
        return [
            AgentSkill(
                id="stock_retriever",
                name="股票数据获取",
                description="获取 A 股历史数据并计算技术指标 (MACD/KDJ/RSI/布林带等)",
                tags=["股票", "数据", "技术指标", "A股"],
                examples=[
                    "获取股票 000001 的数据",
                    "查询贵州茅台(600519)的技术指标",
                    "分析比亚迪(002594)最近60天的走势",
                ],
                input_modes=["text"],
                output_modes=["text", "data"],
            ),
            AgentSkill(
                id="stock_report_generator",
                name="股票报告生成",
                description="基于技术指标数据生成专业投资分析报告和操作建议",
                tags=["股票", "报告", "分析", "投资建议"],
                examples=[
                    "生成股票分析报告",
                    "给出投资建议",
                ],
                input_modes=["text", "data"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="market_overview",
                name="市场行情概览",
                description="获取 A 股大盘指数、市场整体行情和最新市场新闻，生成市场分析报告",
                tags=["大盘", "指数", "市场", "行情", "A股"],
                examples=[
                    "今天A股行情怎么样",
                    "大盘走势如何",
                    "市场整体情况",
                    "看看今天股市",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="save_user_preference",
                name="保存用户偏好",
                description="保存用户的投资偏好设置到长期记忆",
                tags=["偏好", "设置", "记忆"],
                examples=[
                    "记住我喜欢保守型投资",
                    "我偏好科技股",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="update_stock_position",
                name="更新股票持仓",
                description="更新用户股票持仓（设置/买入/卖出/清仓），自动计算成本价",
                tags=["持仓", "买入", "卖出", "股票"],
                examples=[
                    "我持有600588股票2万元",
                    "买入600588 1000股",
                    "卖出002230 500股",
                    "清仓600588",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="get_stock_positions",
                name="查询股票持仓",
                description="查询用户股票持仓，支持查询单个股票或所有持仓",
                tags=["持仓", "查询", "股票"],
                examples=[
                    "我的股票持仓",
                    "查看600588持仓",
                    "我买了哪些股票",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ]

    async def process_task(self, task: Task) -> Task:
        """处理 A2A 任务 - 内部调用 LangGraph

        使用 LangGraph StateGraph 进行流程编排：
        1. 解析用户输入
        2. LLM Tool Calling 路由决策
        3. 执行对应操作（股票分析/市场概览/持仓管理/偏好设置）
        4. 返回结果

        支持多轮对话：
        - 同一 user_id + session_id 的请求共享同一个 thread
        - LangGraph checkpointer 会自动管理消息历史
        - 客户端传入新的 session_id 开启新对话
        """
        user_text = self._extract_text_from_task(task)
        user_id = task.metadata.get("user_id", "default") if task.metadata else "default"
        session_id = task.metadata.get("session_id", "active") if task.metadata else "active"

        if not user_text:
            return self._create_text_response(
                task,
                "请提供您想要了解的内容，例如：\n- 今天A股行情怎么样\n- 分析股票 000001\n- 我买入了002364股票300元\n- 我偏好保守型投资",
                TaskState.INPUT_REQUIRED,
            )

        try:
            task.set_state(TaskState.WORKING, "正在处理您的请求...")

            # 确保异步初始化已完成（启用多轮对话）
            if self._langgraph_checkpointer is None:
                await self.initialize()
            else:
                # 检查并确保 checkpointer 连接有效
                await self._ensure_checkpointer_connection()

            # 生成 thread_id：user_id + session_id 确定唯一会话
            thread_id = generate_thread_id(user_id, self.name, session_id)

            # 构建 config（用于 checkpointer 关联会话）
            config = {"configurable": {"thread_id": thread_id}}

            # 构建初始状态
            # 注意：不再显式传递 messages: []，让 checkpointer 管理历史
            initial_state: StockAgentState = {
                "messages": [],  # checkpointer 会自动加载历史并合并
                "user_input": user_text,
                "user_id": user_id,
                "stock_code": None,
                "task_type": "",
                "position_operation": None,
                "position_amount": None,
                "position_shares": None,
                "position_price": None,
                "preference_key": None,
                "preference_value": None,
                "preference_category": None,
                "stock_data": None,
                "stock_name": None,
                "index_data": None,
                "market_news": None,
                "final_response": None,
                "status": "pending",
                "error": None,
            }

            # 调用 LangGraph（带 config 以关联会话）
            logger.info(f"🤖 StockAgent LangGraph 处理: {user_text[:50]}... (thread: {thread_id})")
            final_state = await self._graph.ainvoke(initial_state, config=config)

            # 提取结果
            response = final_state.get("final_response", "处理完成")
            status = final_state.get("status", "completed")

            if status == "failed":
                return self._create_text_response(task, response, TaskState.FAILED)
            else:
                return self._create_text_response(task, response)

        except Exception as e:
            logger.error(f"❌ 处理任务异常: {e}")
            return self._create_text_response(
                task,
                f"处理请求失败: {str(e)}",
                TaskState.FAILED,
            )

    async def process_task_stream(
        self,
        task: Task,
        include_progress: bool = True,
        include_tokens: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式处理 A2A 任务

        支持两种流式输出模式:
        1. include_progress=True: 输出每个节点的执行进度 (stream_mode="updates")
        2. include_tokens=True: 输出 LLM 的 token 流 (stream_mode="messages")

        Args:
            task: A2A 任务
            include_progress: 是否输出节点进度
            include_tokens: 是否输出 LLM tokens

        Yields:
            StreamEvent: 流式事件
        """
        user_text = self._extract_text_from_task(task)
        user_id = task.metadata.get("user_id", "default") if task.metadata else "default"
        session_id = task.metadata.get("session_id", "active") if task.metadata else "active"

        if not user_text:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error="请提供您想要了解的内容",
            )
            return

        try:
            # 发送元数据
            thread_id = generate_thread_id(user_id, self.name, session_id)
            yield StreamEvent(
                type=StreamEventType.METADATA,
                metadata={
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent": self.name,
                }
            )

            # 确保异步初始化已完成
            if self._langgraph_checkpointer is None:
                await self.initialize()

            # 构建 config
            config = {"configurable": {"thread_id": thread_id}}

            # 构建初始状态
            initial_state: StockAgentState = {
                "messages": [],
                "user_input": user_text,
                "user_id": user_id,
                "stock_code": None,
                "task_type": "",
                "position_operation": None,
                "position_amount": None,
                "position_shares": None,
                "position_price": None,
                "preference_key": None,
                "preference_value": None,
                "preference_category": None,
                "stock_data": None,
                "stock_name": None,
                "index_data": None,
                "market_news": None,
                "final_response": None,
                "status": "pending",
                "error": None,
            }

            logger.info(f"🤖 StockAgent 流式处理: {user_text[:50]}... (thread: {thread_id})")

            # 使用 StreamHandler 进行流式处理
            handler = StreamHandler(
                self._graph,
                include_progress=include_progress,
                include_tokens=include_tokens,
            )

            async for event in handler.stream(initial_state, config):
                yield event

        except Exception as e:
            logger.error(f"❌ 流式处理异常: {e}")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error=str(e),
            )

    # =========================================================================
    # 辅助方法（供 LangGraph 节点使用）
    # =========================================================================

    def _classify_task_type(self, text: str) -> StockTaskType:
        """分类任务类型（规则匹配，用于降级处理）"""
        has_stock_code = self._extract_stock_symbol(text) is not None
        has_stock_name = any(name in text for name in STOCK_NAME_MAPPING.keys())
        has_market_keywords = any(kw in text for kw in MARKET_OVERVIEW_KEYWORDS)
        has_stock_keywords = any(kw in text for kw in STOCK_ANALYSIS_KEYWORDS)

        if has_stock_code or has_stock_name:
            return StockTaskType.STOCK_ANALYSIS
        elif has_market_keywords:
            return StockTaskType.MARKET_OVERVIEW
        elif has_stock_keywords:
            return StockTaskType.STOCK_ANALYSIS
        else:
            return StockTaskType.UNKNOWN

    def _extract_stock_name_code(self, text: str) -> Optional[str]:
        """从文本中提取股票名称对应的代码"""
        for name, code in STOCK_NAME_MAPPING.items():
            if name in text:
                return code
        return None

    def _extract_stock_symbol(self, text: str) -> Optional[str]:
        """从文本中提取股票代码"""
        patterns = [
            r'(\d{6})',
            r'股票[代码]?\s*[:：]?\s*(\d{6})',
            r'(\d{6})\s*[（(][^)）]+[)）]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_stock_code_from_history(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        """从历史消息中提取最近的股票代码

        用于支持多轮对话中的指代消解，如"它的MACD怎么样"。
        从最近的消息开始向前查找。
        """
        # 从后往前遍历，找到最近提到的股票代码
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(content, str):
                # 先尝试提取股票代码
                stock_code = self._extract_stock_symbol(content)
                if stock_code:
                    return stock_code
                # 再尝试提取股票名称
                stock_code = self._extract_stock_name_code(content)
                if stock_code:
                    return stock_code
        return None

    async def _generate_market_report(
        self,
        index_data: Dict[str, Any],
        market_news: List[Dict[str, Any]],
        user_id: str,
    ) -> str:
        """使用模板生成市场分析报告"""
        from datetime import datetime

        # 构建指数数据摘要
        indices_summary = []
        for name, data in index_data.get("indices", {}).items():
            if "error" not in data:
                change_emoji = "📈" if data.get("pct_change", 0) > 0 else "📉" if data.get("pct_change", 0) < 0 else "➡️"
                indices_summary.append(
                    f"- {name}: {data.get('close', 'N/A')} ({change_emoji} {data.get('pct_change', 0):+.2f}%)"
                )

        # 构建新闻摘要
        news_summary = []
        for news in market_news[:5]:
            title = news.get("title", news.get("content", "")[:50])
            if title:
                news_summary.append(f"- {title}")

        # 使用模板生成报告
        report = f"""# A股市场行情报告
📅 {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 📊 主要指数
{chr(10).join(indices_summary) if indices_summary else "数据获取中..."}

## 📰 市场动态
{chr(10).join(news_summary) if news_summary else "暂无最新新闻"}

---
*数据来源: AKShare / Tavily*
"""
        return report


# =============================================================================
# 导出图供 LangGraph Studio 使用
# =============================================================================

def _create_default_agent() -> StockAgent:
    """创建默认的 StockAgent 实例（用于 Studio）"""
    checkpoint_manager = None
    try:
        if postgres_config.connection_string:
            checkpoint_manager = CheckpointManager(postgres_config.connection_string)
    except Exception:
        pass
    return StockAgent(checkpoint_manager=checkpoint_manager, enable_mcp_search=False)


# 延迟初始化，避免导入时连接数据库
_default_agent: Optional[StockAgent] = None


def get_graph() -> CompiledStateGraph:
    """获取编译后的图（用于 langgraph.json 配置）"""
    global _default_agent
    if _default_agent is None:
        _default_agent = _create_default_agent()
    return _default_agent.graph


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """Stock Agent 独立启动入口"""
    parser = argparse.ArgumentParser(description="Stock Agent - 股票分析服务 (LangGraph 架构)")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    parser.add_argument("--no-mcp-search", action="store_true", help="禁用 MCP 搜索")
    parser.add_argument("--no-multi-turn", action="store_true", help="禁用多轮对话支持")
    args = parser.parse_args()

    print(f"启动 Stock Agent v2.0.0 (LangGraph 架构)...")
    print(f"  - 地址: http://{args.host}:{args.port}")
    print(f"  - Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
    print(f"  - 支持: 个股分析 + 市场行情概览 + 持仓管理")

    # 创建 CheckpointManager 以支持用户偏好存储
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
        print("  - ✅ 用户偏好存储已配置 (PostgreSQL)")
    except Exception as e:
        print(f"  - ⚠️ 用户偏好存储未启用: {e}")
        checkpoint_manager = None

    agent = StockAgent(
        checkpoint_manager=checkpoint_manager,
        enable_mcp_search=not args.no_mcp_search,
    )

    # 预初始化多轮对话支持（可选）
    if not args.no_multi_turn:
        try:
            asyncio.run(agent.initialize())
            print("  - ✅ 多轮对话支持已启用 (LangGraph PostgresSaver)")
        except Exception as e:
            print(f"  - ⚠️ 多轮对话支持未启用: {e}")

    print(f"  - ✅ LangGraph 节点: {list(agent.graph.nodes.keys())}")

    agent.start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
