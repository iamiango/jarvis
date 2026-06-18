"""LangGraph Agent Executor - 使用 LangGraph 标准模式实现 LLM 自主工具调用

使用 LangChain 的 create_agent 模式（基于 LangGraph StateGraph），
让本地 Ollama (qwen3.5:9b) 通过 Tool Calling 自主决定调用哪些工具。

Usage:
    executor = LangGraphAgentExecutor(checkpoint_manager)
    result = await executor.run("分析基金008089", user_id="test")
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from pydantic import BaseModel, Field, create_model

from ..config import ollama_config
from ..skills.base import BaseSkill, SkillOutput
from ..storage.checkpoint import CheckpointManager
from ..prompt_loader import load_prompt, PromptNames


logger = logging.getLogger(__name__)


def _create_args_schema_from_params(tool_name: str, params: Dict[str, Any]) -> type:
    """从 skill 的 get_parameters() 返回值动态创建 Pydantic model

    Args:
        tool_name: 工具名称，用于生成 model 名称
        params: skill.get_parameters() 返回的 JSON Schema

    Returns:
        动态创建的 Pydantic BaseModel 类
    """
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    fields = {}
    for name, prop in properties.items():
        field_type = str  # 默认字符串类型
        prop_type = prop.get("type", "string")

        if prop_type == "integer":
            field_type = int
        elif prop_type == "number":
            field_type = float
        elif prop_type == "boolean":
            field_type = bool
        elif prop_type == "string":
            # 字符串类型允许 dict（LLM 可能传递 dict 而非 JSON 字符串）
            # tool_func 中会自动将 dict 转换为 JSON 字符串
            field_type = Union[str, dict]

        description = prop.get("description", "")
        default = prop.get("default", ...)

        if name in required:
            fields[name] = (field_type, Field(description=description))
        else:
            # Optional 字段需要提供默认值
            if default is ...:
                default = None
            fields[name] = (Optional[field_type], Field(default=default, description=description))

    # 动态创建 Pydantic model
    model_name = f"{tool_name.title().replace('_', '')}Input"
    return create_model(model_name, **fields)


def create_skill_tool(
    skill: BaseSkill,
    tool_name: str,
    tool_description: str,
    checkpoint_manager: Optional[CheckpointManager] = None,
    default_user_id: str = "default",
) -> StructuredTool:
    """将 BaseSkill 转换为 LangChain StructuredTool

    Args:
        skill: Skill 实例
        tool_name: 工具名称（LLM 看到的名称）
        tool_description: 工具描述
        checkpoint_manager: 检查点管理器
        default_user_id: 默认用户 ID

    Returns:
        LangChain StructuredTool
    """
    # 获取 skill 的参数 schema 并创建 Pydantic model
    params = skill.get_parameters()
    args_schema = _create_args_schema_from_params(tool_name, params)

    async def tool_func(**kwargs) -> str:
        """工具执行函数"""
        import json as json_module

        logger.debug(f"🔍 原始 kwargs: {list(kwargs.keys())}")

        # 处理 LLM 传递 dict 而非 JSON 字符串的情况
        # 例如: {'fund_data': {...}} -> {'fund_data': '{"..."}'}
        for key, value in list(kwargs.items()):
            if isinstance(value, dict):
                logger.debug(f"🔍 转换 dict -> JSON: {key}")
                kwargs[key] = json_module.dumps(value, ensure_ascii=False)

        # 注入 checkpoint_manager（如果 skill 需要）
        if checkpoint_manager is not None:
            kwargs["checkpoint_manager"] = checkpoint_manager

        # 注入 user_id（如果未提供）
        if "user_id" not in kwargs:
            kwargs["user_id"] = default_user_id

        try:
            logger.info(f"🔧 调用工具 {tool_name}: {list(kwargs.keys())}")
            result: SkillOutput = await skill.execute(**kwargs)

            if result.success:
                # 返回字符串结果
                if isinstance(result.result, str):
                    return result.result
                elif isinstance(result.result, dict):
                    # 如果有 message 字段，优先返回
                    if "message" in result.result:
                        return result.result["message"]
                    return json_module.dumps(result.result, ensure_ascii=False)
                else:
                    return str(result.result) if result.result else "执行成功"
            else:
                return f"执行失败: {result.error}"

        except Exception as e:
            logger.error(f"❌ 工具 {tool_name} 执行异常: {e}")
            return f"执行出错: {str(e)}"

    # 创建 StructuredTool，带有 args_schema
    return StructuredTool.from_function(
        coroutine=tool_func,
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
    )


class LangGraphAgentExecutor:
    """基于 LangGraph 的 Agent 执行器

    使用 LangChain 的 create_agent 模式（返回 LangGraph CompiledStateGraph），
    让本地 Ollama (qwen3.5:9b) 通过 Tool Calling 自主决定调用哪些工具。

    Features:
    - 自动将 BaseSkill 转换为 LangChain Tool
    - 使用本地 Ollama 的 Tool Calling 能力
    - 支持多轮工具调用（Agent Loop）
    - 内置 checkpoint_manager 注入
    """

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        verbose: bool = True,
    ):
        """初始化 Agent 执行器

        Args:
            checkpoint_manager: 检查点管理器，用于持久化
            model: 模型名称，默认使用 ollama_config.model (qwen3.5:9b)
            temperature: 温度参数
            verbose: 是否显示详细日志
        """
        self._checkpoint_manager = checkpoint_manager
        self._model = model or ollama_config.model
        self._temperature = temperature
        self._verbose = verbose

        self._llm: Optional[ChatOllama] = None
        self._tools: List[StructuredTool] = []
        self._agent = None  # LangGraph CompiledStateGraph
        self._initialized = False

    def _get_llm(self) -> ChatOllama:
        """获取本地 Ollama LLM 实例（懒加载）"""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self._model,
                base_url=ollama_config.base_url,
                temperature=self._temperature,
            )
            logger.info(f"✅ 本地 Ollama LLM 已初始化: {self._model} @ {ollama_config.base_url}")

        return self._llm

    def register_skill(
        self,
        skill: BaseSkill,
        tool_name: str,
        tool_description: str,
        default_user_id: str = "default",
    ) -> None:
        """注册 Skill 为 LangChain Tool

        Args:
            skill: Skill 实例
            tool_name: 工具名称
            tool_description: 工具描述
            default_user_id: 默认用户 ID
        """
        tool = create_skill_tool(
            skill=skill,
            tool_name=tool_name,
            tool_description=tool_description,
            checkpoint_manager=self._checkpoint_manager,
            default_user_id=default_user_id,
        )
        self._tools.append(tool)
        logger.info(f"📦 已注册工具: {tool_name}")

        # 标记需要重新初始化
        self._initialized = False

    def _build_agent(self):
        """构建 LangGraph Agent

        Returns:
            LangGraph CompiledStateGraph 实例
        """
        llm = self._get_llm()

        if not self._tools:
            raise ValueError("没有注册任何工具，请先调用 register_skill()")

        # 使用 LangChain 的 create_agent (返回 LangGraph StateGraph)
        agent = create_agent(
            model=llm,
            tools=self._tools,
            system_prompt=load_prompt(PromptNames.FUND_AGENT_EXECUTOR),
            debug=self._verbose,
        )

        logger.info(f"✅ LangGraph Agent 已构建，工具: {[t.name for t in self._tools]}")
        return agent

    async def run(
        self,
        user_input: str,
        user_id: str = "default",
    ) -> str:
        """运行 Agent 处理用户请求

        Args:
            user_input: 用户输入
            user_id: 用户 ID

        Returns:
            Agent 的最终响应
        """
        # 构建或获取 Agent
        if not self._initialized or self._agent is None:
            self._agent = self._build_agent()
            self._initialized = True

        try:
            logger.info(f"🚀 开始处理: {user_input[:50]}...")

            # 运行 LangGraph Agent
            result = await self._agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}],
            })

            # 提取最终响应
            messages = result.get("messages", [])
            if messages:
                # 获取最后一条 AI 消息
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content:
                        output = msg.content
                        logger.info(f"✅ 处理完成: {len(output)} 字符")
                        return output

            return "处理完成，但无响应内容"

        except Exception as e:
            logger.error(f"❌ Agent 执行失败: {e}")
            raise

    def list_tools(self) -> List[str]:
        """列出所有注册的工具"""
        return [tool.name for tool in self._tools]


# 保持向后兼容的别名
LangChainAgentExecutor = LangGraphAgentExecutor


def create_fund_agent_executor(
    checkpoint_manager: Optional[CheckpointManager] = None,
    verbose: bool = True,
) -> LangGraphAgentExecutor:
    """创建基金分析 Agent 执行器的便捷方法

    自动注册所有基金相关的 Skills。

    Args:
        checkpoint_manager: 检查点管理器
        verbose: 是否显示详细日志

    Returns:
        配置好的 LangGraphAgentExecutor
    """
    from ..skills.fund_retriever import FundRetrieverSkill
    from ..skills.fund_report import FundReportGeneratorSkill
    from ..skills.user_preference import SaveUserPreferenceSkill
    from ..skills.fund_position import UpdateFundPositionSkill, GetFundPositionsSkill

    executor = LangGraphAgentExecutor(
        checkpoint_manager=checkpoint_manager,
        verbose=verbose,
    )

    # 注册基金数据获取工具
    executor.register_skill(
        skill=FundRetrieverSkill(),
        tool_name="get_fund_data",
        tool_description="获取基金的原始净值数据和技术指标（返回JSON格式的原始数据，不是分析报告）。这是数据采集工具，获取的数据需要传给 generate_fund_report 才能生成用户可读的分析报告。参数: fund_code (基金代码，6位数字)",
    )

    # 注册基金报告生成工具
    executor.register_skill(
        skill=FundReportGeneratorSkill(checkpoint_manager),
        tool_name="generate_fund_report",
        tool_description="【必须调用】将基金原始数据转换为专业的中文分析报告。使用专业金融模型分析技术指标并给出投资建议。用户要求分析基金时，必须调用此工具生成最终报告。参数: fund_data (来自 get_fund_data 的JSON字符串), fund_code (基金代码)",
    )

    # 注册偏好保存工具
    executor.register_skill(
        skill=SaveUserPreferenceSkill(checkpoint_manager),
        tool_name="save_preference",
        tool_description="保存用户投资偏好。参数: category (类别，如 'fund'), key (偏好键，如 'risk_tolerance'), value (偏好值，如 'conservative')",
    )

    # 注册持仓更新工具
    executor.register_skill(
        skill=UpdateFundPositionSkill(checkpoint_manager),
        tool_name="update_position",
        tool_description="更新基金持仓记录。参数: fund_code (基金代码), operation (操作类型: buy/sell/set/clear), amount (金额，clear时不需要)",
    )

    # 注册持仓查询工具
    executor.register_skill(
        skill=GetFundPositionsSkill(checkpoint_manager),
        tool_name="get_positions",
        tool_description="查询用户的基金持仓情况。参数: fund_code (可选，指定则查询单只基金)",
    )

    return executor
