"""LangGraph Studio Graph - 真实 Jarvis Agent 图定义

此模块导出一个编译好的 LangGraph 图，封装了完整的 JarvisAgent 功能：
- 纯 LLM 路由（本地 Ollama 智能分析）
- A2A 协议子 Agent 调度
- MCP 工具集成（Tavily 搜索）
- PII 防护
- Human-in-the-Loop 确认

使用方式:
    langgraph dev --config langgraph.json
"""

import os
import sys
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# 添加项目根目录到 Python 路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import ollama_config, agent_config
from src.router import LLMRouter
from src.security import PIIGuard, PIIStrategy


# =============================================================================
# 状态定义 - 完整的 Jarvis 工作流状态
# =============================================================================

class JarvisState(TypedDict):
    """Jarvis 图状态 - 真实工作流状态"""
    # 消息历史
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 用户输入
    user_input: str

    # PII 处理
    pii_detected: bool
    pii_count: int
    processed_input: str  # PII 处理后的输入

    # LLM 路由结果
    intent_type: str  # direct_answer / stock_analysis / fund_analysis / email / stock_position / fund_position
    need_agent: bool
    target_agent: Optional[str]
    steps: List[Dict[str, Any]]  # 多步骤任务列表
    confidence: float  # LLM 路由置信度
    reasoning: str  # LLM 推理过程

    # 执行状态
    current_step: int
    step_results: Dict[int, str]
    agent_responses: Dict[str, str]

    # 最终响应
    final_response: Optional[str]

    # 状态
    status: str  # pending / pii_check / routing / executing / completed / failed

    # 错误信息
    error: Optional[str]


# =============================================================================
# 全局组件初始化
# =============================================================================

# PII 防护
_pii_guard = PIIGuard(strategy=PIIStrategy.REDACT)

# LLM 路由器（纯 LLM 路由）
_llm_router = LLMRouter()
print("  ✅ LLM Router 已启用 (本地 Ollama)")

# 可用 Agents 列表 (用于 LLM 路由决策)
_available_agents = [
    {"name": "stock_agent", "description": "股票分析 Agent - 支持个股技术分析、市场概览、持仓管理"},
    {"name": "fund_agent", "description": "基金分析 Agent - 支持基金分析、持仓管理"},
    {"name": "email_agent", "description": "邮件发送 Agent - 支持发送邮件"},
]


# =============================================================================
# 节点函数 - 真实业务逻辑
# =============================================================================

def pii_check(state: JarvisState) -> Dict[str, Any]:
    """PII 防护检查节点

    使用 PIIGuard 检测和处理敏感信息（邮箱、电话、身份证等）
    """
    user_input = state.get("user_input", "")

    if not user_input:
        return {
            "status": "failed",
            "error": "未收到有效输入",
            "final_response": "请告诉我您需要什么帮助？",
        }

    # PII 检测和处理
    pii_result = _pii_guard.process(user_input)

    if pii_result.has_pii:
        print(f"🛡️ PII 防护: 检测到 {len(pii_result.matches)} 个敏感信息")
        for match in pii_result.matches:
            print(f"   - {match.pii_type}: {match.original[:20]}...")

    return {
        "pii_detected": pii_result.has_pii,
        "pii_count": len(pii_result.matches) if pii_result.has_pii else 0,
        "processed_input": pii_result.processed_text,
        "status": "pii_check",
    }


async def parse_intent(state: JarvisState) -> Dict[str, Any]:
    """LLM 路由决策节点 (异步)

    使用本地 Ollama LLM 分析用户意图，决定:
    1. 是否需要调用 Agent
    2. 调用哪个 Agent（支持多 Agent 顺序调用）
    3. 或者直接给出回答
    """
    processed_input = state.get("processed_input", state.get("user_input", ""))

    if not processed_input:
        return {
            "status": "failed",
            "error": "无有效输入",
        }

    # 构建 AgentInfo 列表供 LLM Router 使用
    from src.a2a import AgentInfo, AgentCard, AgentSkill

    agents = []
    for agent_data in _available_agents:
        card = AgentCard(
            name=agent_data["name"],
            description=agent_data["description"],
            url=f"http://localhost:800{1 if 'stock' in agent_data['name'] else (2 if 'email' in agent_data['name'] else 3)}",
            version="1.0.0",
            skills=[],
        )
        agents.append(AgentInfo(card=card, status="available"))

    # 使用 LLM Router 分析意图
    print(f"🤖 LLM Router 分析用户意图...")
    route_result = await _llm_router.route(processed_input, agents)

    print(f"📍 路由决策: need_agent={route_result.need_agent}, steps={len(route_result.steps)}")
    if route_result.reasoning:
        print(f"   推理: {route_result.reasoning[:80]}...")

    # 确定意图类型
    intent_type = "direct_answer"
    target_agent = None

    if route_result.need_agent and route_result.steps:
        first_step = route_result.steps[0]
        target_agent = first_step.agent_name

        # 根据 agent 名称推断意图类型
        if "stock" in target_agent:
            intent_type = "stock_analysis"
        elif "fund" in target_agent:
            intent_type = "fund_analysis"
        elif "email" in target_agent:
            intent_type = "email"

    # 映射步骤格式
    steps = []
    for step in route_result.steps:
        steps.append({
            "step": step.step,
            "agent_name": step.agent_name,
            "task_description": step.task_description,
            "depends_on_previous": step.depends_on_previous,
        })

    return {
        "intent_type": intent_type,
        "need_agent": route_result.need_agent,
        "target_agent": target_agent,
        "steps": steps,
        "confidence": route_result.confidence,
        "reasoning": route_result.reasoning,
        "final_response": route_result.direct_answer if not route_result.need_agent else None,
        "status": "routing",
    }


def route_decision(state: JarvisState) -> Dict[str, Any]:
    """路由决策节点

    根据 LLM 路由结果决定执行路径
    """
    need_agent = state.get("need_agent", False)
    steps = state.get("steps", [])
    direct_answer = state.get("final_response")

    if not need_agent:
        # 如果 LLM 已经给出直接回答，直接完成
        if direct_answer:
            return {"status": "completed"}
        return {"status": "direct_answer"}

    if len(steps) > 1:
        return {
            "status": "compound_task",
            "current_step": 0,
        }

    return {"status": "agent_task"}


def execute_agent_task(state: JarvisState) -> Dict[str, Any]:
    """执行 Agent 任务节点

    模拟 A2A 协议调用子 Agent
    注意: 在 LangGraph Studio 中，实际的 A2A 调用需要子 Agent 服务运行
    """
    target_agent = state.get("target_agent", "")
    user_input = state.get("processed_input", state.get("user_input", ""))
    steps = state.get("steps", [])
    agent_responses = dict(state.get("agent_responses", {}))
    confidence = state.get("confidence", 0.0)
    reasoning = state.get("reasoning", "")

    # 获取任务描述（从第一个步骤）
    task_description = user_input
    if steps:
        task_description = steps[0].get("task_description", user_input)

    # 模拟执行结果 (实际执行需要 A2A 客户端)
    result = f"""📌 Agent 任务已路由 (纯 LLM 路由)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 目标 Agent: {target_agent}
📝 任务描述: {task_description[:100]}...
📊 置信度: {confidence:.2f}
💭 推理: {reasoning[:80]}...

⚠️ 注意: 在 LangGraph Studio 中显示的是路由结果。
   实际执行需要启动对应的子 Agent 服务:
   - stock_agent: python -m src.agents.stock_agent --port 8001
   - email_agent: python -m src.agents.email_agent --port 8002
   - fund_agent: python -m src.agents.fund_agent --port 8003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    agent_responses[target_agent] = result

    return {
        "agent_responses": agent_responses,
        "final_response": result,
        "status": "completed",
    }


def execute_compound_step(state: JarvisState) -> Dict[str, Any]:
    """执行复合任务步骤节点

    逐步执行复合任务中的每个步骤
    """
    steps = state.get("steps", [])
    current_step = state.get("current_step", 0)
    step_results = dict(state.get("step_results", {}))

    if current_step >= len(steps):
        return {"status": "aggregate"}

    step = steps[current_step]
    step_num = step.get("step", current_step + 1)
    agent_name = step.get("agent_name", "unknown")
    task_desc = step.get("task_description", "")

    # 模拟执行步骤
    result = f"[步骤 {step_num}] {agent_name}: {task_desc[:50]}... ✓"
    step_results[step_num] = result

    print(f"🔹 执行步骤 {step_num}: {agent_name} - {task_desc[:30]}...")

    return {
        "current_step": current_step + 1,
        "step_results": step_results,
    }


def generate_direct_response(state: JarvisState) -> Dict[str, Any]:
    """生成直接回答节点

    对于不需要 Agent 的问题，直接生成回答
    在生产环境中会调用 LLM + MCP 工具
    """
    user_input = state.get("processed_input", state.get("user_input", ""))
    messages = list(state.get("messages", []))
    reasoning = state.get("reasoning", "")

    # 添加用户消息到历史
    messages.append(HumanMessage(content=user_input))

    # 生成响应说明
    response_text = f"""💬 直接回答模式 (纯 LLM 路由)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 用户问题: {user_input}
💭 路由推理: {reasoning[:100] if reasoning else 'N/A'}

在完整的 Jarvis 系统中，这里会:
1. 使用 ChatOllama (qwen3.5:9b) 生成回答
2. 如果需要实时信息，调用 MCP Tavily 搜索
3. 搜索结果会被缓存以避免重复调用

要体验完整功能，请使用:
- python main.py (CLI 交互模式)
- 或启动 LangGraph API: python -m src.api.langgraph_adapter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    messages.append(AIMessage(content=response_text))

    return {
        "messages": messages,
        "final_response": response_text,
        "status": "completed",
    }


def aggregate_results(state: JarvisState) -> Dict[str, Any]:
    """聚合复合任务结果节点

    汇总所有步骤的执行结果
    """
    step_results = state.get("step_results", {})
    steps = state.get("steps", [])

    if not step_results:
        return {
            "final_response": "没有执行结果",
            "status": "completed",
        }

    # 构建汇总
    parts = [
        "📊 复合任务执行摘要",
        "━" * 40,
    ]

    for i, step in enumerate(steps, 1):
        result = step_results.get(i, "未执行")
        agent_name = step.get("agent_name", "unknown")
        task_desc = step.get("task_description", "")[:40]
        parts.append(f"\n✅ 步骤 {i}: {agent_name}")
        parts.append(f"   任务: {task_desc}...")
        parts.append(f"   结果: {result}")

    parts.append("\n" + "━" * 40)
    parts.append("⚠️ 这是模拟执行结果，实际执行需要启动子 Agent 服务")

    return {
        "final_response": "\n".join(parts),
        "status": "completed",
    }


# =============================================================================
# 条件路由函数
# =============================================================================

def should_execute_or_respond(state: JarvisState) -> str:
    """路由决策: 执行 Agent 任务、复合任务还是直接回答"""
    status = state.get("status", "")

    if status == "completed":
        # LLM 已给出直接回答
        return "end"
    elif status == "direct_answer":
        return "direct_response"
    elif status == "compound_task":
        return "compound"
    elif status == "agent_task":
        return "agent"
    else:
        return "end"


def should_continue_or_aggregate(state: JarvisState) -> str:
    """复合任务: 继续下一步还是聚合结果"""
    steps = state.get("steps", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(steps):
        return "aggregate"
    else:
        return "next_step"


# =============================================================================
# 构建图
# =============================================================================

def build_jarvis_graph() -> StateGraph:
    """构建真实的 Jarvis LangGraph 状态机

    工作流 (纯 LLM 路由):
    1. PII 检查 -> LLM 路由决策 -> 路由分发
    2. 根据 LLM 决策分发:
       - need_agent=False + direct_answer -> 结束
       - need_agent=False -> 直接回答
       - need_agent=True + 单步骤 -> Agent 执行
       - need_agent=True + 多步骤 -> 复合任务循环执行
    """

    # 创建状态图
    workflow = StateGraph(JarvisState)

    # 添加节点
    workflow.add_node("pii_check", pii_check)
    workflow.add_node("parse_intent", parse_intent)
    workflow.add_node("route", route_decision)
    workflow.add_node("execute_agent", execute_agent_task)
    workflow.add_node("execute_compound_step", execute_compound_step)
    workflow.add_node("direct_response", generate_direct_response)
    workflow.add_node("aggregate", aggregate_results)

    # 设置入口点
    workflow.set_entry_point("pii_check")

    # 添加边
    workflow.add_edge("pii_check", "parse_intent")
    workflow.add_edge("parse_intent", "route")

    # 条件路由: 路由决策后
    workflow.add_conditional_edges(
        "route",
        should_execute_or_respond,
        {
            "agent": "execute_agent",
            "compound": "execute_compound_step",
            "direct_response": "direct_response",
            "end": END,
        }
    )

    # Agent 任务执行后结束
    workflow.add_edge("execute_agent", END)

    # 条件路由: 复合任务步骤执行后
    workflow.add_conditional_edges(
        "execute_compound_step",
        should_continue_or_aggregate,
        {
            "next_step": "execute_compound_step",
            "aggregate": "aggregate",
        }
    )

    # 终点边
    workflow.add_edge("direct_response", END)
    workflow.add_edge("aggregate", END)

    return workflow


# =============================================================================
# 导出编译后的图 (供 LangGraph Studio 使用)
# =============================================================================

# 构建并编译图 (不使用 checkpointer - LangGraph API 会自动处理持久化)
_workflow = build_jarvis_graph()
graph = _workflow.compile()


# 为 Studio 提供入口函数
async def run_jarvis(user_input: str, thread_id: str = "default") -> str:
    """运行 Jarvis 图的入口函数

    Args:
        user_input: 用户输入
        thread_id: 线程 ID（用于状态持久化）

    Returns:
        最终响应文本
    """
    initial_state = {
        "messages": [],
        "user_input": user_input,
        "pii_detected": False,
        "pii_count": 0,
        "processed_input": "",
        "intent_type": "",
        "need_agent": False,
        "target_agent": None,
        "steps": [],
        "confidence": 0.0,
        "reasoning": "",
        "current_step": 0,
        "step_results": {},
        "agent_responses": {},
        "final_response": None,
        "status": "pending",
        "error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.ainvoke(initial_state, config)

    return final_state.get("final_response", "处理完成")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("  Jarvis LangGraph Studio - 测试模式")
        print("=" * 60)

        test_cases = [
            "今天天气怎么样",  # 直接回答
            "分析股票600519",  # 股票分析
            "分析基金000001并发送邮件到test@example.com",  # 复合任务
        ]

        for i, query in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}: {query} ---")
            result = await run_jarvis(query)
            print(result)
            print()

    asyncio.run(test())
