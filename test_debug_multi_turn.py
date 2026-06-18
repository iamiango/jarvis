#!/usr/bin/env python
"""深度分析多轮对话失败原因

问题：第二轮 "它的MACD怎么样" 没有正确理解上下文
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def debug_multi_turn():
    """调试多轮对话"""

    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart
    from src.storage import LangGraphCheckpointer

    print("=" * 80)
    print("  调试多轮对话问题")
    print("=" * 80)

    # 检查 LangGraphCheckpointer 状态
    print("\n[1] 检查 LangGraphCheckpointer 状态")
    print(f"    is_initialized: {LangGraphCheckpointer.is_initialized()}")

    agent = StockAgent()

    # 检查 agent 的 checkpointer 状态
    print(f"    agent._langgraph_checkpointer: {agent._langgraph_checkpointer}")

    # 手动初始化
    print("\n[2] 尝试初始化 checkpointer...")
    try:
        await agent.initialize()
        print(f"    初始化后 is_initialized: {LangGraphCheckpointer.is_initialized()}")
        print(f"    agent._langgraph_checkpointer: {agent._langgraph_checkpointer}")
    except Exception as e:
        print(f"    ❌ 初始化失败: {e}")

    # 检查 graph 是否带有 checkpointer
    print("\n[3] 检查 graph 编译状态")
    print(f"    graph 节点: {list(agent.graph.nodes.keys())}")

    # 第一轮
    print("\n[4] 第一轮: '分析股票600588'")
    task1 = Task(id="debug_001", metadata={"user_id": "debug_user"})
    task1.add_message(Message(role="user", parts=[TextPart(text="分析股票600588")]))

    result1 = await agent.process_task(task1)
    print(f"    状态: {result1.status.state.value}")

    # 检查 checkpointer 中是否有数据
    print("\n[5] 检查 checkpointer 存储")
    if agent._langgraph_checkpointer:
        try:
            from src.storage import generate_thread_id
            thread_id = generate_thread_id("debug_user", "stock_agent")
            print(f"    thread_id: {thread_id}")

            # 尝试获取检查点
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = await agent._langgraph_checkpointer.aget(config)
            if checkpoint:
                print(f"    ✅ 找到检查点!")
                messages = checkpoint.get("channel_values", {}).get("messages", [])
                print(f"    消息数量: {len(messages)}")
                for i, msg in enumerate(messages[:5]):  # 只显示前5条
                    content = msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100]
                    print(f"      [{i}] {type(msg).__name__}: {content}...")
            else:
                print("    ❌ 没有找到检查点")
        except Exception as e:
            print(f"    ❌ 获取检查点失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    ❌ checkpointer 未初始化，无法持久化会话!")

    # 第二轮
    print("\n[6] 第二轮: '它的MACD怎么样'")
    task2 = Task(id="debug_002", metadata={"user_id": "debug_user"})
    task2.add_message(Message(role="user", parts=[TextPart(text="它的MACD怎么样")]))

    # 添加调试日志
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("stock_agent")
    logger.setLevel(logging.DEBUG)

    result2 = await agent.process_task(task2)
    print(f"    状态: {result2.status.state.value}")

    if result2.status.message:
        msg = result2.status.message
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'text'):
                    print(f"    响应: {part.text[:300]}...")


if __name__ == "__main__":
    asyncio.run(debug_multi_turn())
