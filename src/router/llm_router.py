"""LLM 路由器 - 基于本地 Ollama 的智能路由

当规则解析器无法匹配用户意图时，作为第四层 fallback 路由。
使用本地 Ollama (qwen3.5:9b) 分析用户输入，决定：
1. 是否需要调用 Agent
2. 调用哪个 Agent（支持多 Agent 顺序调用）
3. 或者直接给出回答
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import ollama_config
from ..a2a import AgentInfo, AgentSkill
from ..prompt_loader import load_prompt, PromptNames


@dataclass
class LLMRouteStep:
    """LLM 路由步骤"""
    step: int                           # 步骤序号
    agent_name: str                     # Agent 名称
    task_description: str               # 任务描述
    depends_on_previous: bool = False   # 是否依赖前一步结果


@dataclass
class LLMRouteResult:
    """LLM 路由结果"""
    need_agent: bool                    # 是否需要调用 Agent
    steps: List[LLMRouteStep] = field(default_factory=list)  # 执行步骤列表
    direct_answer: Optional[str] = None # 直接回答（不需要 Agent 时）
    confidence: float = 0.0             # 置信度
    reasoning: str = ""                 # LLM 推理过程


class LLMRouter:
    """LLM 路由器 - 基于本地 Ollama 的智能路由

    当规则解析器无法匹配时，使用本地 Ollama 进行智能路由决策。
    支持：
    - 单 Agent 任务路由
    - 多 Agent 顺序执行
    - 直接回答（不需要 Agent 时）
    """

    def _get_system_prompt(self, agent_descriptions: str) -> str:
        """获取系统提示词（从 md 文件加载）"""
        return load_prompt(PromptNames.LLM_ROUTER, available_agents=agent_descriptions)

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        timeout: int = 120,
        max_steps: int = 5,
    ):
        """初始化 LLM 路由器

        Args:
            model: 模型名称，默认从配置读取 (qwen3.5:9b)
            base_url: Ollama Base URL，默认从配置读取
            temperature: 温度参数，建议使用较低值以获得稳定路由
            timeout: 调用超时（秒）
            max_steps: 单次请求最多执行步骤数
        """
        self._model = model or ollama_config.model
        self._base_url = base_url or ollama_config.base_url
        self._temperature = temperature
        self._timeout = timeout
        self._max_steps = max_steps
        self._llm: Optional[ChatOllama] = None

    def _get_llm(self) -> ChatOllama:
        """获取 Ollama LLM 客户端（懒加载）"""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self._model,
                base_url=self._base_url,
                temperature=self._temperature,
                format="json",  # 强制 JSON 输出
            )
        return self._llm

    def _build_agent_descriptions(self, agents: List[AgentInfo]) -> str:
        """构建 Agent 能力描述供 LLM 参考

        Args:
            agents: 已注册的 Agent 信息列表

        Returns:
            Agent 描述文本
        """
        if not agents:
            return "（暂无可用 Agent）"

        descriptions = []
        for agent_info in agents:
            card = agent_info.card
            skills_text = self._format_skills(card.skills)

            desc = f"""### {card.name}
描述: {card.description}
技能:
{skills_text}"""
            descriptions.append(desc)

        return "\n\n".join(descriptions)

    def _format_skills(self, skills: List[AgentSkill]) -> str:
        """格式化技能列表

        Args:
            skills: 技能列表

        Returns:
            格式化的技能文本
        """
        if not skills:
            return "  - （无特定技能）"

        lines = []
        for skill in skills:
            line = f"  - {skill.name}: {skill.description}"
            if skill.examples:
                examples_str = ", ".join(skill.examples[:2])  # 最多显示2个示例
                line += f"\n    示例: {examples_str}"
            lines.append(line)

        return "\n".join(lines)

    def _parse_llm_response(self, response_text: str) -> LLMRouteResult:
        """解析 LLM 返回的 JSON 决策

        Args:
            response_text: LLM 返回的文本

        Returns:
            LLMRouteResult 路由结果
        """
        # 尝试提取 JSON
        json_text = response_text.strip()

        # 处理可能的 markdown 代码块
        if "```json" in json_text:
            start = json_text.find("```json") + 7
            end = json_text.find("```", start)
            if end > start:
                json_text = json_text[start:end].strip()
        elif "```" in json_text:
            start = json_text.find("```") + 3
            end = json_text.find("```", start)
            if end > start:
                json_text = json_text[start:end].strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            # JSON 解析失败，返回直接回答
            print(f"  ⚠️ LLM 返回的 JSON 解析失败: {e}")
            return LLMRouteResult(
                need_agent=False,
                direct_answer="抱歉，我无法理解您的请求。请尝试更具体地描述您需要什么帮助。",
                reasoning=f"JSON 解析失败: {response_text[:200]}..."
            )

        # 解析步骤
        steps = []
        raw_steps = data.get("steps", [])
        for i, step_data in enumerate(raw_steps):
            if i >= self._max_steps:
                break
            step = LLMRouteStep(
                step=step_data.get("step", i + 1),
                agent_name=step_data.get("agent_name", ""),
                task_description=step_data.get("task_description", ""),
                depends_on_previous=step_data.get("depends_on_previous", i > 0),
            )
            steps.append(step)

        return LLMRouteResult(
            need_agent=data.get("need_agent", False),
            steps=steps,
            direct_answer=data.get("direct_answer"),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
        )

    async def route(
        self,
        text: str,
        agents: List[AgentInfo],
    ) -> LLMRouteResult:
        """分析用户输入，决定路由策略

        Args:
            text: 用户输入文本
            agents: 已注册的 Agent 信息列表

        Returns:
            LLMRouteResult 路由决策
        """
        if not text or not text.strip():
            return LLMRouteResult(
                need_agent=False,
                direct_answer="请告诉我您需要什么帮助？",
                reasoning="空输入"
            )

        # 构建 Agent 描述
        agent_descriptions = self._build_agent_descriptions(agents)

        # 构建系统 prompt（从 md 文件加载）
        system_prompt = self._get_system_prompt(agent_descriptions)

        # 构建用户消息
        user_message = f"用户输入: {text}"

        try:
            llm = self._get_llm()

            print(f"  🤖 调用本地 Ollama LLM Router ({self._model})...")

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            response = await llm.ainvoke(messages)
            response_text = response.content or ""

            # 解析响应
            result = self._parse_llm_response(response_text)

            # 验证 Agent 名称
            valid_agent_names = {agent.card.name for agent in agents}
            validated_steps = []
            for step in result.steps:
                if step.agent_name in valid_agent_names:
                    validated_steps.append(step)
                else:
                    print(f"  ⚠️ LLM 返回了无效的 Agent 名称: {step.agent_name}")

            # 如果所有步骤都无效，改为直接回答
            if result.need_agent and not validated_steps:
                return LLMRouteResult(
                    need_agent=False,
                    direct_answer="抱歉，我无法完成您的请求。目前可用的 Agent 无法处理此类任务。",
                    reasoning=f"LLM 返回的 Agent 名称无效: {[s.agent_name for s in result.steps]}"
                )

            result.steps = validated_steps

            print(f"  ✅ LLM 路由完成: need_agent={result.need_agent}, steps={len(result.steps)}")
            if result.reasoning:
                print(f"     推理: {result.reasoning[:100]}...")

            return result

        except Exception as e:
            print(f"  ❌ LLM Router 调用失败: {e}")
            return LLMRouteResult(
                need_agent=False,
                direct_answer=None,  # 返回 None 让上层决定如何处理
                reasoning=f"LLM 调用异常: {str(e)}"
            )

    @property
    def is_available(self) -> bool:
        """检查 LLM Router 是否可用（本地 Ollama 始终可用）"""
        return True
