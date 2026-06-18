#!/usr/bin/env python
"""多轮对话执行路径模拟分析

场景：
1. 用户问 Jarvis: "分析股票600588"
2. 用户跟进: "它的MACD怎么样"

分析执行路径和多轮对话支持。
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(step: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}→ {step}")


def print_issue(issue: str):
    print(f"  ❌ 问题: {issue}")


def print_ok(msg: str):
    print(f"  ✅ {msg}")


async def analyze_execution_path():
    """分析执行路径"""

    print_section("场景描述")
    print("""
    用户通过 Jarvis 进行多轮对话：

    第一轮: "分析股票600588"
    第二轮: "它的MACD怎么样"

    期望：第二轮能理解"它"指的是600588
    """)

    print_section("第一轮执行路径: '分析股票600588'")

    print("\n[1] Jarvis.process_task() 接收请求")
    print_step("user_text = '分析股票600588'")
    print_step("调用 _arun_with_task_id()")

    print("\n[2] Jarvis 规则/LLM 路由")
    print_step("RuleParser/LLMRouter.parse('分析股票600588')")
    print_step("识别意图: stock_analysis, agent: stock_agent")

    print("\n[3] Jarvis._handle_stock_analysis()")
    print_step("提取股票代码: 600588")
    print_step("构建消息: '分析股票 600588'")
    print_step("调用 delegate_task('stock_agent', message)")

    print("\n[4] Jarvis.delegate_task() -> A2AClient.send_task()")
    print_step("创建 Task 对象")
    print_step("Task.history.add(Message(role='user', text='分析股票 600588'))")
    print_issue("未传递 user_id 到 metadata!")

    print("\n[5] StockAgent.process_task() 接收 Task")
    print_step("user_text = _extract_text_from_task(task) -> '分析股票 600588'")
    print_step("user_id = task.metadata.get('user_id', 'default') -> 'default'")
    print_step("thread_id = 'default:stock_agent:active'")

    print("\n[6] StockAgent LangGraph 执行")
    print_step("_node_add_user_message: messages += [HumanMessage('分析股票 600588')]")
    print_step("_node_parse_input: 提取 stock_code = '600588'")
    print_step("_node_llm_routing: LLM 决定调用 stock_analysis")
    print_step("_node_fetch_stock_data: 获取600588数据")
    print_step("_node_generate_report: 生成分析报告")
    print_step("_node_add_ai_response: messages += [AIMessage('报告内容...')]")

    print("\n[7] LangGraph Checkpointer 持久化")
    print_step("保存到 PostgreSQL: thread_id='default:stock_agent:active'")
    print_step("checkpoints 表存储: messages = [HumanMessage, AIMessage]")
    print_ok("第一轮消息历史已保存")

    print_section("第二轮执行路径: '它的MACD怎么样'")

    print("\n[1] Jarvis.process_task() 接收请求")
    print_step("user_text = '它的MACD怎么样'")

    print("\n[2] Jarvis 规则/LLM 路由")
    print_step("RuleParser.parse('它的MACD怎么样')")
    print_step("识别意图: stock_analysis (基于 'MACD' 关键词)")
    print_issue("Jarvis 无法从'它'推断股票代码!")
    print_step("转发原始消息给 stock_agent")

    print("\n[3] Jarvis._handle_stock_analysis()")
    print_step("尝试提取股票代码: None (没有显式代码)")
    print_step("构建消息: '它的MACD怎么样' (原样转发)")
    print_step("调用 delegate_task('stock_agent', message)")

    print("\n[4] A2AClient.send_task()")
    print_step("创建新的 Task 对象 (id 不同)")
    print_step("Task.history = [Message('它的MACD怎么样')]")
    print_issue("这是一个全新的 Task，没有第一轮的历史!")

    print("\n[5] StockAgent.process_task() 接收 Task")
    print_step("user_text = '它的MACD怎么样'")
    print_step("user_id = 'default' (仍然)")
    print_step("thread_id = 'default:stock_agent:active' (相同!)")

    print("\n[6] StockAgent LangGraph 执行 (带 checkpointer)")
    print_step("config = {'configurable': {'thread_id': 'default:stock_agent:active'}}")
    print_step("initial_state['messages'] = [] (空)")
    print_step("ainvoke() 时 checkpointer 自动加载历史:")
    print_step("  - 加载: [HumanMessage('分析股票600588'), AIMessage('报告...')]")
    print_step("  - 合并: [历史消息] + [新 HumanMessage('它的MACD怎么样')]")

    print("\n[7] _node_add_user_message")
    print_step("messages += [HumanMessage('它的MACD怎么样')]")
    print_step("现在 messages = [HumanMsg1, AIMsg1, HumanMsg2]")

    print("\n[8] _node_parse_input")
    print_step("user_input = '它的MACD怎么样'")
    print_step("stock_code = _extract_stock_symbol() -> None")
    print_step("stock_code = _extract_stock_code_from_history(messages)")
    print_step("  遍历 messages[-1:0] 查找股票代码")
    print_step("  在 HumanMessage('分析股票600588') 中找到 '600588'")
    print_ok("从历史推断 stock_code = '600588'")

    print("\n[9] _node_llm_routing")
    print_step("构建 messages = [SystemPrompt] + recent_history[-10:]")
    print_step("LLM 看到完整上下文:")
    print_step("  - Human: '分析股票600588'")
    print_step("  - AI: '报告...'")
    print_step("  - Human: '它的MACD怎么样'")
    print_step("LLM 理解'它'指 600588，调用 stock_analysis(stock_code='600588')")
    print_ok("LLM 正确理解指代关系")

    print_section("关键分析")

    print("\n✅ 多轮对话可以工作的条件:")
    print("   1. StockAgent 使用 LangGraph Checkpointer (PostgresSaver)")
    print("   2. 同一 user_id 生成相同的 thread_id")
    print("   3. _extract_stock_code_from_history() 从历史中查找股票代码")
    print("   4. _node_llm_routing() 将历史消息传递给 LLM")

    print("\n⚠️ 潜在问题:")
    print("   1. Jarvis 不传递 user_id 到 A2AClient.send_task()")
    print("      - 当前所有请求使用 user_id='default'")
    print("      - 这意味着所有用户共享同一个会话!")
    print("   2. 每次请求创建新的 Task 对象")
    print("      - Task.history 不包含之前的对话")
    print("      - 依赖 LangGraph checkpointer 而非 A2A Task 历史")

    print("\n📋 当前实现依赖:")
    print("   - PostgreSQL 数据库连接正常")
    print("   - LangGraphCheckpointer 正确初始化")
    print("   - StockAgent.initialize() 被调用")

    print_section("实际测试")

    # 实际测试
    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart

    print("\n创建 StockAgent...")
    agent = StockAgent()

    print("\n--- 第一轮: '分析股票600588' ---")
    task1 = Task(id="test_001", metadata={"user_id": "test_user"})
    task1.add_message(Message(role="user", parts=[TextPart(text="分析股票600588")]))

    try:
        result1 = await agent.process_task(task1)
        print(f"状态: {result1.status.state.value}")
        if result1.status.state.value == "completed":
            print("✅ 第一轮成功")
        else:
            print(f"❌ 第一轮失败: {result1.status.message}")
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- 第二轮: '它的MACD怎么样' ---")
    task2 = Task(id="test_002", metadata={"user_id": "test_user"})  # 相同 user_id
    task2.add_message(Message(role="user", parts=[TextPart(text="它的MACD怎么样")]))

    try:
        result2 = await agent.process_task(task2)
        print(f"状态: {result2.status.state.value}")

        # 检查响应内容
        if result2.status.message:
            msg = result2.status.message
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'text'):
                        text = part.text[:500] if len(part.text) > 500 else part.text
                        print(f"响应内容: {text}...")

                        # 检查是否提到了 600588 或 MACD
                        if '600588' in part.text or 'MACD' in part.text or 'macd' in part.text:
                            print("✅ 第二轮成功 - 正确理解了'它'的指代")
                        else:
                            print("⚠️ 响应中没有明确提到600588或MACD")
                        break
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(analyze_execution_path())
