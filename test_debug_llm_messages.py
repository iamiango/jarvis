#!/usr/bin/env python
"""详细调试 LLM 收到的消息内容"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def debug_llm_messages():
    """调试 LLM 收到的消息"""

    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart
    from src.storage import LangGraphCheckpointer, generate_thread_id
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    print("=" * 80)
    print("  调试 LLM 消息传递")
    print("=" * 80)

    agent = StockAgent()

    # 手动初始化 checkpointer
    print("\n[1] 初始化 checkpointer...")
    await agent.initialize()
    print(f"    checkpointer: {agent._langgraph_checkpointer}")

    # 第一轮
    print("\n[2] 第一轮: '分析股票600588'")
    task1 = Task(id="llm_debug_001", metadata={"user_id": "llm_debug_user"})
    task1.add_message(Message(role="user", parts=[TextPart(text="分析股票600588")]))

    result1 = await agent.process_task(task1)
    print(f"    状态: {result1.status.state.value}")

    # 检查检查点
    print("\n[3] 检查检查点存储的消息...")
    thread_id = generate_thread_id("llm_debug_user", "stock_agent")
    config = {"configurable": {"thread_id": thread_id}}

    checkpoint = await agent._langgraph_checkpointer.aget(config)
    if checkpoint:
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        print(f"    检查点中的消息数量: {len(messages)}")

        print("\n    === 检查点中的消息内容 ===")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, 'content') else str(msg)
            # 截断长内容
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    [{i}] {msg_type}: {content}")
    else:
        print("    ❌ 没有找到检查点")
        return

    # 模拟第二轮的 LLM 调用
    print("\n[4] 模拟第二轮 LLM 调用...")

    # 模拟 _node_add_user_message 的效果
    new_user_msg = HumanMessage(content="它的MACD怎么样")

    # 构建 LLM 会收到的消息列表
    llm_messages = [SystemMessage(content=agent.TOOL_CALLING_SYSTEM_PROMPT)]

    # 添加历史消息 + 新消息（模拟 checkpointer 加载后的状态）
    all_messages = messages + [new_user_msg]
    recent_history = all_messages[-10:] if len(all_messages) > 10 else all_messages
    llm_messages.extend(recent_history)

    print(f"\n    === LLM 将收到的消息 ({len(llm_messages)} 条) ===")
    for i, msg in enumerate(llm_messages):
        msg_type = type(msg).__name__
        content = msg.content if hasattr(msg, 'content') else str(msg)
        # 截断长内容
        if len(content) > 300:
            content = content[:300] + "...[截断]"
        print(f"\n    --- 消息 {i} ({msg_type}) ---")
        print(f"    {content}")

    # 实际调用 LLM
    print("\n\n[5] 实际调用 LLM...")
    from langchain_ollama import ChatOllama
    from src.config import ollama_config

    llm = ChatOllama(
        model=ollama_config.model,
        base_url=ollama_config.base_url,
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(agent._tools)

    print(f"    模型: {ollama_config.model}")
    print(f"    工具数量: {len(agent._tools)}")

    response = await llm_with_tools.ainvoke(llm_messages)

    print(f"\n    === LLM 响应 ===")
    print(f"    tool_calls: {response.tool_calls}")
    print(f"    content: {response.content[:500] if response.content else 'None'}...")

    if response.tool_calls:
        print("\n    ✅ LLM 调用了工具!")
        for tc in response.tool_calls:
            print(f"       工具: {tc.get('name')}")
            print(f"       参数: {tc.get('args')}")
    else:
        print("\n    ❌ LLM 没有调用工具")

    # 真正执行第二轮
    print("\n\n[6] 真正执行第二轮...")
    task2 = Task(id="llm_debug_002", metadata={"user_id": "llm_debug_user"})
    task2.add_message(Message(role="user", parts=[TextPart(text="它的MACD怎么样")]))

    result2 = await agent.process_task(task2)
    print(f"    状态: {result2.status.state.value}")

    if result2.status.message:
        msg = result2.status.message
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'text'):
                    text = part.text[:500] if len(part.text) > 500 else part.text
                    print(f"    响应: {text}")


if __name__ == "__main__":
    asyncio.run(debug_llm_messages())
