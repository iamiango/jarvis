"""Fund Agent - 基金分析 Agent (LangGraph 架构)

提供基金数据获取、技术分析和投资报告生成功能。
基于 LangGraph StateGraph 实现，支持:
- A2A 协议接口 (HTTP 服务)
- LangGraph Studio 可视化调试

v3.1 架构改进:
- 核心逻辑使用 LangGraph StateGraph 实现
- 一份代码同时支持 A2A 服务和 Studio 可视化
- 使用 LLM Tool Calling 进行路由决策（与 StockAgent 保持一致）
"""
import re
import logging
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence, TypedDict
import argparse
import asyncio

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from .base import BaseAgent
from ..a2a import AgentSkill, Task, TaskState
from ..streaming import StreamEvent, StreamEventType, NodeProgress, TokenChunk, StreamHandler
from ..skills.fund_retriever import FundRetrieverSkill
from ..skills.fund_report import FundReportGeneratorSkill
from ..skills.user_preference import SaveUserPreferenceSkill, GetUserPreferencesSkill
from ..skills.fund_position import (
    UpdateFundPositionSkill,
    GetFundPositionsSkill,
    detect_position_operation,
)
from ..storage.checkpoint import CheckpointManager
from ..storage.langgraph_checkpoint import LangGraphCheckpointer, generate_thread_id
from ..config import postgres_config, ollama_config
from ..prompt_loader import load_prompt, PromptNames


logger = logging.getLogger(__name__)


# =============================================================================
# 状态定义
# =============================================================================

class FundAgentState(TypedDict):
    """FundAgent 图状态"""
    # 消息历史
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 用户输入
    user_input: str
    user_id: str

    # 解析结果
    fund_code: Optional[str]
    task_type: str  # data_query / analysis / position_update / position_query / preference_save / preference_query

    # 持仓操作
    position_operation: Optional[str]  # buy / sell / clear / query / set
    position_amount: Optional[float]

    # 偏好设置
    preference_key: Optional[str]
    preference_value: Optional[str]
    preference_category: Optional[str]  # fund / general

    # 用户偏好检测
    detected_preferences: List[Dict[str, Any]]

    # 执行状态
    fund_data: Optional[str]  # 基金数据 JSON
    fund_name: Optional[str]
    fund_nav: Optional[float]  # 当前净值

    # 最终响应
    final_response: Optional[str]

    # 状态
    status: str  # pending / parsing / llm_routing / fetching_data / generating_report / position_op / preference_op / completed / failed

    # 错误信息
    error: Optional[str]


# =============================================================================
# 基金偏好检测规则
# =============================================================================

FUND_PREFERENCE_PATTERNS = {
    "risk_tolerance": {
        "conservative": ["保守", "稳健", "低风险", "安全", "稳定", "保本"],
        "moderate": ["中等", "平衡", "适中"],
        "aggressive": ["激进", "进取", "高风险", "高收益"],
    },
    "fund_type": {
        "equity": ["股票型", "股票基金", "权益"],
        "bond": ["债券型", "债券基金", "固收", "纯债"],
        "hybrid": ["混合型", "混合基金", "偏股", "偏债"],
        "index": ["指数型", "指数基金", "ETF", "被动"],
        "money_market": ["货币型", "货币基金", "余额宝"],
        "qdii": ["QDII", "海外", "境外", "美股", "港股"],
    },
    "investment_horizon": {
        "short-term": ["短期", "流动性", "随时取"],
        "medium-term": ["中期", "半年", "一年", "中长期"],
        "long-term": ["长期", "长持", "定投", "养老", "中长期投资"],
    },
    "dividend_preference": {
        "reinvest": ["再投资", "复利", "滚存"],
        "cash": ["现金分红", "分红"],
    },
}


# =============================================================================
# FundAgent 类 - LangGraph 架构
# =============================================================================

class FundAgent(BaseAgent):
    """基金分析 Agent - 基于 LangGraph 架构

    技能:
    - fund_retriever: 获取基金数据和技术指标
    - fund_report: 生成专业分析报告
    - save_user_preference: 保存用户偏好
    - get_user_preferences: 查询用户偏好
    - update_fund_position: 更新基金持仓
    - get_fund_positions: 查询基金持仓

    架构特点:
    - 核心逻辑使用 LangGraph StateGraph 实现
    - 通过 self.graph 属性导出编译后的图供 Studio 使用
    - A2A process_task() 方法内部调用 graph.ainvoke()
    - 使用 LLM Tool Calling 进行路由决策
    """

    name = "fund_agent"
    description = "基金分析 Agent - 提供基金净值获取、业绩分析、专业投资分析报告和持仓管理"
    version = "3.1.0"  # LLM 路由版本

    # 摘要配置
    SUMMARIZATION_TRIGGER = 10  # 消息数超过此值时触发摘要
    SUMMARIZATION_KEEP = 5      # 摘要后保留最近 N 条消息

    def _get_tool_calling_prompt(self) -> str:
        """获取 Tool Calling 系统提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.FUND_AGENT_TOOL_CALLING)

    def _get_summarization_prompt(self, messages: str) -> str:
        """获取摘要提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.SUMMARIZATION, asset_type="基金", messages=messages)

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """初始化 Fund Agent

        Args:
            checkpoint_manager: 检查点管理器，用于持久化用户偏好和持仓
        """
        super().__init__()
        self._checkpoint_manager = checkpoint_manager
        self._ollama_llm: Optional[ChatOllama] = None
        self._langgraph_checkpointer = None  # 延迟初始化

        # 初始化技能
        self._retriever = FundRetrieverSkill()
        self._report_generator = FundReportGeneratorSkill(checkpoint_manager)
        self._save_preference_skill = SaveUserPreferenceSkill(checkpoint_manager)
        self._get_preferences_skill = GetUserPreferencesSkill(checkpoint_manager)
        self._update_position_skill = UpdateFundPositionSkill(checkpoint_manager)
        self._get_positions_skill = GetFundPositionsSkill(checkpoint_manager)

        # 注册技能到基类
        self.register_skill(self._retriever)
        self.register_skill(self._report_generator)
        self.register_skill(self._save_preference_skill)
        self.register_skill(self._get_preferences_skill)
        self.register_skill(self._update_position_skill)
        self.register_skill(self._get_positions_skill)

        # 构建 LLM 工具定义
        self._tools = self._build_tools()

        # 构建 LangGraph（不带 checkpointer，需要异步初始化后重建）
        self._graph: Optional[CompiledStateGraph] = None
        self._build_graph()

        logger.info(f"✅ FundAgent v{self.version} 初始化完成 (LLM 路由架构)")

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
            logger.info("✅ FundAgent 多轮对话支持已启用")
        except Exception as e:
            logger.warning(f"⚠️ 无法启用多轮对话支持: {e}")
            self._langgraph_checkpointer = None

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
                    "name": "fund_analysis",
                    "description": "分析基金（获取净值数据和专业分析报告）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fund_code": {
                                "type": "string",
                                "description": "基金代码（6位数字）"
                            }
                        },
                        "required": ["fund_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fund_data",
                    "description": "仅获取基金数据（不生成分析报告）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fund_code": {
                                "type": "string",
                                "description": "基金代码（6位数字）"
                            }
                        },
                        "required": ["fund_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_fund_position",
                    "description": "更新基金持仓（买入、卖出、设置持仓、清仓）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fund_code": {
                                "type": "string",
                                "description": "基金代码（6位数字）"
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["set", "buy", "sell", "clear"],
                                "description": "操作类型：set(设置持仓), buy(买入), sell(卖出), clear(清仓)"
                            },
                            "amount": {
                                "type": "number",
                                "description": "交易金额（元）"
                            }
                        },
                        "required": ["fund_code", "operation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fund_positions",
                    "description": "查询基金持仓",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fund_code": {
                                "type": "string",
                                "description": "基金代码（可选，不指定则查询所有持仓）"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_user_preference",
                    "description": "保存用户投资偏好（风险偏好、投资周期、基金类型等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": ["risk_tolerance", "investment_horizon", "fund_type", "dividend_preference"],
                                "description": "偏好类型：risk_tolerance(风险偏好), investment_horizon(投资周期), fund_type(基金类型), dividend_preference(分红偏好)"
                            },
                            "value": {
                                "type": "string",
                                "description": "偏好值。risk_tolerance: conservative/moderate/aggressive; investment_horizon: short-term/medium-term/long-term; fund_type: equity/bond/hybrid/index/money_market; dividend_preference: reinvest/cash"
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
                    "description": "查询用户投资偏好设置",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["fund", "general"],
                                "description": "偏好类别（可选）：fund(基金), general(通用)。不指定则返回所有偏好"
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
        workflow = StateGraph(FundAgentState)

        # 添加节点 - 使用实例方法作为节点函数
        workflow.add_node("add_user_message", self._node_add_user_message)
        workflow.add_node("summarize_history", self._node_summarize_history)  # 新增：摘要节点
        workflow.add_node("parse", self._node_parse_input)
        workflow.add_node("llm_route", self._node_llm_routing)
        workflow.add_node("fetch_data", self._node_fetch_data)
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
                "fetch": "fetch_data",
                "position": "position_op",
                "preference": "preference_op",
                "end": "add_ai_response",
            }
        )

        # 条件路由: 获取数据后
        workflow.add_conditional_edges(
            "fetch_data",
            self._should_generate_or_end,
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

    def _node_add_user_message(self, state: FundAgentState) -> Dict[str, Any]:
        """将用户输入添加到消息历史节点

        这是工作流的入口点，将用户消息添加到 messages 列表，
        以便 LangGraph checkpointer 持久化对话历史。
        """
        from langchain_core.messages import HumanMessage
        user_input = state.get("user_input", "")
        if user_input:
            return {
                "messages": [HumanMessage(content=user_input)]
            }
        return {}

    async def _node_summarize_history(self, state: FundAgentState) -> Dict[str, Any]:
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

    def _node_add_ai_response(self, state: FundAgentState) -> Dict[str, Any]:
        """将 AI 响应添加到消息历史节点

        这是工作流的出口点，将最终响应添加到 messages 列表，
        以便 LangGraph checkpointer 持久化对话历史。
        """
        from langchain_core.messages import AIMessage
        response = state.get("final_response", "")
        if response:
            return {
                "messages": [AIMessage(content=response)]
            }
        return {}

    def _node_parse_input(self, state: FundAgentState) -> Dict[str, Any]:
        """解析用户输入节点

        支持多轮对话：如果当前输入没有基金代码，会尝试从历史消息中查找。
        """
        user_input = state.get("user_input", "")

        if not user_input:
            return {
                "status": "failed",
                "error": "未收到有效输入",
                "final_response": "请提供基金代码进行分析，例如：分析基金 008089",
            }

        # 提取基金代码
        fund_code = self._extract_fund_code(user_input)

        # 如果当前输入没有基金代码，尝试从历史消息中查找（支持"它"、"这只基金"等指代）
        if not fund_code:
            fund_code = self._extract_fund_code_from_history(state.get("messages", []))
            if fund_code:
                logger.info(f"   从历史消息中推断基金代码: {fund_code}")

        logger.info(f"📋 解析用户输入: {user_input[:50]}...")
        logger.info(f"   提取基金代码: {fund_code}")

        return {
            "fund_code": fund_code,
            "status": "parsing",
        }

    async def _node_llm_routing(self, state: FundAgentState) -> Dict[str, Any]:
        """使用 LLM Tool Calling 进行路由决策节点

        注意：使用 state["messages"] 中的历史消息来支持多轮对话，
        让 LLM 能理解"它"、"这只基金"等指代。
        """
        user_input = state.get("user_input", "")
        fund_code = state.get("fund_code")
        history_messages = state.get("messages", [])

        try:
            llm = self._get_ollama_llm()
            llm_with_tools = llm.bind_tools(self._tools)

            # 构建消息列表：系统提示 + 历史消息（已包含当前用户输入）
            messages = [SystemMessage(content=self._get_tool_calling_prompt())]

            # 添加历史消息（checkpointer 加载的 + 当前轮次的 HumanMessage）
            # 限制历史长度避免上下文过长
            recent_history = history_messages[-10:] if len(history_messages) > 10 else history_messages
            messages.extend(recent_history)

            logger.info(f"🤖 调用 LLM ({ollama_config.model}) 进行 Tool Calling...")
            logger.info(f"   消息历史: {len(recent_history)} 条")
            response = await llm_with_tools.ainvoke(messages)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                logger.info(f"🔧 LLM 决定调用工具: {tool_name}")
                logger.info(f"   参数: {tool_args}")

                # 根据工具类型设置状态
                if tool_name == "fund_analysis":
                    return {
                        "task_type": "analysis",
                        "fund_code": tool_args.get("fund_code", fund_code),
                        "status": "fetching_data",
                    }
                elif tool_name == "get_fund_data":
                    return {
                        "task_type": "data_query",
                        "fund_code": tool_args.get("fund_code", fund_code),
                        "status": "fetching_data",
                    }
                elif tool_name == "update_fund_position":
                    return {
                        "task_type": "position_update",
                        "fund_code": tool_args.get("fund_code"),
                        "position_operation": tool_args.get("operation"),
                        "position_amount": tool_args.get("amount"),
                        "status": "position_op",
                    }
                elif tool_name == "get_fund_positions":
                    return {
                        "task_type": "position_query",
                        "fund_code": tool_args.get("fund_code"),
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
                content = response.content or "抱歉，我不太理解您的请求。请告诉我您想要：\n- 分析某只基金（如：分析基金008089）\n- 管理持仓（如：买入基金021500 5000元）\n- 查看持仓情况"
                return {
                    "status": "completed",
                    "final_response": content,
                }

        except Exception as e:
            logger.error(f"❌ LLM Tool Calling 失败: {e}")
            # 降级：尝试规则匹配
            return await self._fallback_rule_routing(state)

    async def _fallback_rule_routing(self, state: FundAgentState) -> Dict[str, Any]:
        """降级路由：基于规则匹配"""
        user_input = state.get("user_input", "")
        fund_code = state.get("fund_code")

        # 检测持仓操作
        position_op = detect_position_operation(user_input)

        if position_op.get("operation"):
            if position_op["operation"] == "query":
                return {
                    "task_type": "position_query",
                    "position_operation": "query",
                    "status": "position_op",
                }
            return {
                "task_type": "position_update",
                "fund_code": fund_code,
                "position_operation": position_op.get("operation"),
                "position_amount": position_op.get("amount"),
                "status": "position_op",
            }

        # 检测偏好相关
        preference_keywords = ["偏好", "喜欢", "风格", "设置"]
        if any(kw in user_input for kw in preference_keywords):
            if "查看" in user_input or "我的" in user_input or "什么" in user_input:
                return {
                    "task_type": "preference_query",
                    "status": "preference_op",
                }

        # 判断任务类型
        data_keywords = ["获取", "查询", "数据", "净值", "走势"]
        report_keywords = ["分析", "报告", "建议", "评估"]

        has_data = any(kw in user_input for kw in data_keywords)
        has_report = any(kw in user_input for kw in report_keywords)

        if fund_code:
            if has_data and not has_report:
                return {
                    "task_type": "data_query",
                    "status": "fetching_data",
                }
            return {
                "task_type": "analysis",
                "status": "fetching_data",
            }

        return {
            "status": "failed",
            "error": "无法识别任务类型",
            "final_response": "抱歉，我无法理解您的请求。请尝试：\n- 分析基金 008089\n- 买入基金021500 5000元\n- 查看我的持仓",
        }

    async def _node_fetch_data(self, state: FundAgentState) -> Dict[str, Any]:
        """获取基金数据节点"""
        fund_code = state.get("fund_code", "")
        user_id = state.get("user_id", "default")

        logger.info(f"📊 正在获取基金 {fund_code} 的数据...")

        try:
            result = await self._retriever.execute(fund_code=fund_code)

            if not result.success:
                return {
                    "status": "failed",
                    "error": result.error,
                    "final_response": f"获取基金数据失败: {result.error}",
                }

            fund_data = result.result
            logger.info(f"   ✅ 数据获取成功")

            # 保存检测到的偏好
            preferences = state.get("detected_preferences", [])
            if preferences and self._checkpoint_manager:
                await self._save_detected_preferences(user_id, preferences)

            # 如果只是数据查询，直接返回
            if state.get("task_type") == "data_query":
                return {
                    "fund_data": fund_data,
                    "final_response": f"📊 基金 {fund_code} 数据获取成功\n\n{fund_data[:2000]}...",
                    "status": "completed",
                }

            return {
                "fund_data": fund_data,
                "status": "generating_report",
            }

        except Exception as e:
            logger.error(f"❌ 获取基金数据异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"获取基金数据异常: {str(e)}",
            }

    async def _node_generate_report(self, state: FundAgentState) -> Dict[str, Any]:
        """生成分析报告节点"""
        fund_code = state.get("fund_code", "")
        fund_data = state.get("fund_data", "")
        user_id = state.get("user_id", "default")

        logger.info(f"📝 正在生成基金 {fund_code} 的分析报告...")

        try:
            result = await self._report_generator.execute(
                fund_data=fund_data,
                user_id=user_id,
                fund_code=fund_code,
            )

            if not result.success:
                return {
                    "status": "failed",
                    "error": result.error,
                    "final_response": f"生成报告失败: {result.error}",
                }

            report = result.result
            logger.info(f"   ✅ 报告生成成功 ({len(report)} 字符)")

            return {
                "final_response": report,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"❌ 生成报告异常: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_response": f"生成报告异常: {str(e)}",
            }

    async def _node_position_operation(self, state: FundAgentState) -> Dict[str, Any]:
        """处理持仓操作节点"""
        operation = state.get("position_operation", "")
        fund_code = state.get("fund_code")
        amount = state.get("position_amount")
        user_id = state.get("user_id", "default")

        op_names = {
            "query": "查询持仓",
            "buy": "买入",
            "sell": "卖出",
            "clear": "清仓",
            "set": "设置持仓",
        }
        op_name = op_names.get(operation, operation)

        logger.info(f"💼 执行持仓操作: {op_name} {fund_code or '全部'}")

        try:
            if operation == "query":
                # 查询持仓
                if fund_code:
                    result = await self._get_positions_skill.execute(
                        user_id=user_id,
                        fund_code=fund_code,
                        checkpoint_manager=self._checkpoint_manager,
                    )
                else:
                    result = await self._get_positions_skill.execute(
                        user_id=user_id,
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
                # 验证金额
                if operation != "clear" and not amount:
                    return {
                        "status": "failed",
                        "error": "缺少金额",
                        "final_response": f"请提供{op_name}金额，例如：买入{fund_code} 5000元",
                    }

                # 获取基金名称和净值
                fund_name = await self._get_fund_name(fund_code)
                current_nav = await self._get_fund_nav(fund_code)

                if current_nav:
                    logger.info(f"   📊 当前净值: {current_nav:.4f}")

                result = await self._update_position_skill.execute(
                    user_id=user_id,
                    fund_code=fund_code,
                    operation=operation,
                    amount=amount,
                    nav=current_nav,
                    fund_name=fund_name,
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

    async def _node_preference_operation(self, state: FundAgentState) -> Dict[str, Any]:
        """处理偏好设置/查询节点"""
        task_type = state.get("task_type")
        user_id = state.get("user_id", "default")

        # 查询偏好
        if task_type == "preference_query":
            return await self._handle_preference_query(state, user_id)

        # 保存偏好
        return await self._handle_preference_save(state, user_id)

    async def _handle_preference_query(self, state: FundAgentState, user_id: str) -> Dict[str, Any]:
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
                    message = "📋 您目前还没有设置任何投资偏好。\n\n您可以告诉我您的偏好，例如：\n- 我偏好保守型投资\n- 我喜欢长期定投\n- 我关注债券型基金"
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
            "investment_horizon": "投资周期",
            "fund_type": "偏好基金类型",
            "dividend_preference": "分红偏好",
        }
        value_descriptions = {
            "risk_tolerance": {
                "conservative": "保守型（低风险）",
                "moderate": "稳健型（中等风险）",
                "aggressive": "激进型（高风险）",
            },
            "investment_horizon": {
                "short-term": "短期理财",
                "medium-term": "中期投资",
                "long-term": "长期定投",
            },
            "fund_type": {
                "equity": "股票型基金",
                "bond": "债券型基金",
                "hybrid": "混合型基金",
                "index": "指数型基金",
                "money_market": "货币型基金",
            },
            "dividend_preference": {
                "reinvest": "红利再投资",
                "cash": "现金分红",
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

    async def _handle_preference_save(self, state: FundAgentState, user_id: str) -> Dict[str, Any]:
        """处理偏好保存"""
        key = state.get("preference_key")
        value = state.get("preference_value")

        if not key or not value:
            return {
                "status": "failed",
                "error": "缺少偏好设置参数",
                "final_response": "请提供完整的偏好设置，例如：\n- 我偏好保守型投资\n- 我喜欢长期定投",
            }

        logger.info(f"💾 保存用户偏好: {key}={value}")

        try:
            result = await self._save_preference_skill.execute(
                user_id=user_id,
                category="fund",
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
                    "investment_horizon": {
                        "short-term": "短期理财",
                        "medium-term": "中期投资",
                        "long-term": "长期定投",
                    },
                    "fund_type": {
                        "equity": "股票型基金",
                        "bond": "债券型基金",
                        "hybrid": "混合型基金",
                        "index": "指数型基金",
                        "money_market": "货币型基金",
                    },
                    "dividend_preference": {
                        "reinvest": "红利再投资",
                        "cash": "现金分红",
                    },
                }
                value_desc = preference_descriptions.get(key, {}).get(value, value)
                key_desc = {
                    "risk_tolerance": "风险偏好",
                    "investment_horizon": "投资周期",
                    "fund_type": "偏好基金类型",
                    "dividend_preference": "分红偏好",
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

    def _should_route_to_node(self, state: FundAgentState) -> str:
        """路由决策: 根据状态决定下一个节点"""
        status = state.get("status", "")

        if status == "failed" or status == "completed":
            return "end"
        elif status == "position_op":
            return "position"
        elif status == "preference_op":
            return "preference"
        elif status == "fetching_data":
            return "fetch"
        else:
            return "end"

    def _should_generate_or_end(self, state: FundAgentState) -> str:
        """获取数据后: 生成报告还是结束"""
        status = state.get("status", "")

        if status == "failed" or status == "completed":
            return "end"
        elif status == "generating_report":
            return "report"
        else:
            return "end"

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _extract_fund_code(self, text: str) -> Optional[str]:
        """从文本中提取基金代码"""
        patterns = [
            r'(\d{6})',
            r'基金[代码]?\s*[:：]?\s*(\d{6})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_fund_code_from_history(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        """从历史消息中提取最近的基金代码

        用于支持多轮对话中的指代消解，如"它的收益怎么样"。
        从最近的消息开始向前查找。
        """
        # 从后往前遍历，找到最近提到的基金代码
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(content, str):
                fund_code = self._extract_fund_code(content)
                if fund_code:
                    return fund_code
        return None

    async def _save_detected_preferences(self, user_id: str, preferences: List[Dict[str, Any]]) -> None:
        """保存检测到的偏好"""
        for pref in preferences:
            try:
                await self._save_preference_skill.execute(
                    user_id=user_id,
                    category="fund",
                    key=pref["key"],
                    value=pref["value"],
                    confidence=pref["confidence"],
                    source="inferred",
                    checkpoint_manager=self._checkpoint_manager,
                )
                logger.info(f"💾 已保存偏好: {pref['key']}={pref['value']}")
            except Exception as e:
                logger.warning(f"⚠️ 保存偏好失败: {e}")

    async def _get_fund_name(self, fund_code: str) -> Optional[str]:
        """获取基金名称"""
        try:
            import akshare as ak
            df = ak.fund_individual_basic_info_xq(symbol=fund_code)
            if df is not None and not df.empty:
                for idx, row in df.iterrows():
                    item = str(row['item']) if 'item' in df.columns else ''
                    value = str(row['value']) if 'value' in df.columns else ''
                    if item in ('基金名称', '基金全称'):
                        return value
        except Exception:
            pass
        return None

    async def _get_fund_nav(self, fund_code: str) -> Optional[float]:
        """获取基金当前净值"""
        try:
            import akshare as ak
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                if '单位净值' in df.columns:
                    return float(latest['单位净值'])
        except Exception as e:
            logger.debug(f"获取净值失败: {e}")
        return None

    # =========================================================================
    # A2A 接口
    # =========================================================================

    def get_skills(self) -> List[AgentSkill]:
        """获取 Agent 技能列表 (A2A 协议)"""
        return [
            AgentSkill(
                id="fund_retriever",
                name="基金数据获取",
                description="获取基金历史净值、业绩表现和技术指标数据",
                tags=["基金", "净值", "数据"],
                examples=["获取基金 008089 的数据", "查询 161725 的净值走势"],
                input_modes=["text"],
                output_modes=["text", "data"],
            ),
            AgentSkill(
                id="fund_report",
                name="基金报告生成",
                description="基于基金数据生成专业的投资分析报告",
                tags=["基金", "分析", "报告"],
                examples=["分析基金 008089", "生成 161725 的投资报告"],
                input_modes=["text", "data"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="update_fund_position",
                name="更新基金持仓",
                description="记录基金买入/卖出/清仓操作，自动计算持仓成本",
                tags=["持仓", "买入", "卖出"],
                examples=["买入基金021500 5000元", "卖出基金008089 2000元"],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="get_fund_positions",
                name="查询基金持仓",
                description="查询用户的基金持仓情况",
                tags=["持仓", "查询"],
                examples=["我的基金持仓情况", "查看021500的持仓"],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="save_user_preference",
                name="保存用户偏好",
                description="保存用户的基金投资偏好设置到长期记忆",
                tags=["偏好", "设置", "记忆"],
                examples=[
                    "我偏好保守型投资",
                    "我喜欢长期定投",
                    "我关注债券型基金",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="get_user_preferences",
                name="查询用户偏好",
                description="查询用户的投资偏好设置",
                tags=["偏好", "查询"],
                examples=["我的偏好设置", "查看我的投资偏好"],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ]

    async def process_task(self, task: Task) -> Task:
        """处理 A2A 任务 - 内部调用 LangGraph

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
                "请提供基金代码进行分析，例如：分析基金 008089",
                TaskState.INPUT_REQUIRED,
            )

        try:
            task.set_state(TaskState.WORKING, "正在处理您的请求...")

            # 确保异步初始化已完成（启用多轮对话）
            if self._langgraph_checkpointer is None:
                await self.initialize()

            # 生成 thread_id：user_id + session_id 确定唯一会话
            thread_id = generate_thread_id(user_id, self.name, session_id)

            # 构建 config（用于 checkpointer 关联会话）
            config = {"configurable": {"thread_id": thread_id}}

            # 构建初始状态
            # 注意：不再显式传递 messages: []，让 checkpointer 管理历史
            initial_state: FundAgentState = {
                "messages": [],  # checkpointer 会自动加载历史并合并
                "user_input": user_text,
                "user_id": user_id,
                "fund_code": None,
                "task_type": "",
                "position_operation": None,
                "position_amount": None,
                "preference_key": None,
                "preference_value": None,
                "preference_category": None,
                "detected_preferences": [],
                "fund_data": None,
                "fund_name": None,
                "fund_nav": None,
                "final_response": None,
                "status": "pending",
                "error": None,
            }

            # 调用 LangGraph（带 config 以关联会话）
            logger.info(f"🤖 FundAgent LangGraph 处理: {user_text[:50]}... (thread: {thread_id})")
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
            initial_state: FundAgentState = {
                "messages": [],
                "user_input": user_text,
                "user_id": user_id,
                "fund_code": None,
                "task_type": "",
                "position_operation": None,
                "position_amount": None,
                "preference_key": None,
                "preference_value": None,
                "preference_category": None,
                "detected_preferences": [],
                "fund_data": None,
                "fund_name": None,
                "fund_nav": None,
                "final_response": None,
                "status": "pending",
                "error": None,
            }

            logger.info(f"🤖 FundAgent 流式处理: {user_text[:50]}... (thread: {thread_id})")

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


# =============================================================================
# 导出图供 LangGraph Studio 使用
# =============================================================================

def _create_default_agent() -> FundAgent:
    """创建默认的 FundAgent 实例（用于 Studio）"""
    checkpoint_manager = None
    try:
        if postgres_config.connection_string:
            checkpoint_manager = CheckpointManager(postgres_config.connection_string)
    except Exception:
        pass
    return FundAgent(checkpoint_manager=checkpoint_manager)


# 延迟初始化，避免导入时连接数据库
_default_agent: Optional[FundAgent] = None


def get_graph() -> CompiledStateGraph:
    """获取编译后的图（用于 langgraph.json 配置）"""
    global _default_agent
    if _default_agent is None:
        _default_agent = _create_default_agent()
    return _default_agent.graph


# 为 langgraph.json 提供直接引用
# 使用函数避免导入时就初始化
graph = property(lambda self: get_graph())


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """Fund Agent 独立启动入口"""
    parser = argparse.ArgumentParser(description="Fund Agent - 基金分析服务 (LLM 路由架构)")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8003, help="服务端口")
    parser.add_argument("--no-multi-turn", action="store_true", help="禁用多轮对话支持")
    args = parser.parse_args()

    print(f"启动 Fund Agent v3.1.0 (LLM 路由架构)...")
    print(f"  - 地址: http://{args.host}:{args.port}")
    print(f"  - Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
    print(f"  - 支持: 基金分析 + 持仓管理 + 偏好设置")

    # 创建 CheckpointManager
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
        print("  - ✅ PostgreSQL 存储已连接")
    except Exception as e:
        print(f"  - ⚠️ PostgreSQL 未连接: {e}")

    agent = FundAgent(
        checkpoint_manager=checkpoint_manager,
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
