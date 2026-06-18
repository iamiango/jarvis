#!/usr/bin/env python
"""演示客户端如何管理会话

场景：
1. 开启新对话（生成新 session_id）
2. 继续对话（复用 session_id）
3. 切换到新话题（生成新 session_id）
"""

import asyncio
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def demo_session_management():
    """演示会话管理"""

    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart

    user_id = "demo_user"

    print("=" * 80)
    print("  演示：客户端会话管理")
    print("=" * 80)

    agent = StockAgent()
    await agent.initialize()

    # ========================================
    # 场景 1: 开启新对话
    # ========================================
    print("\n" + "=" * 60)
    print("  场景 1: 开启新对话 - 分析 600588")
    print("=" * 60)

    session_1 = str(uuid.uuid4())[:8]  # 新会话 ID
    print(f"  session_id: {session_1}")

    task1 = Task(
        id=f"demo_{uuid.uuid4().hex[:8]}",
        metadata={"user_id": user_id, "session_id": session_1}
    )
    task1.add_message(Message(role="user", parts=[TextPart(text="分析股票600588")]))

    result1 = await agent.process_task(task1)
    print(f"  状态: {result1.status.state.value}")
    print("  ✅ 对话 1 开启成功")

    # ========================================
    # 场景 2: 继续同一对话（复用 session_id）
    # ========================================
    print("\n" + "=" * 60)
    print("  场景 2: 继续对话 - '它的MACD怎么样'")
    print("=" * 60)

    print(f"  session_id: {session_1} (复用)")

    task2 = Task(
        id=f"demo_{uuid.uuid4().hex[:8]}",
        metadata={"user_id": user_id, "session_id": session_1}  # 复用 session_id
    )
    task2.add_message(Message(role="user", parts=[TextPart(text="它的MACD怎么样")]))

    result2 = await agent.process_task(task2)
    print(f"  状态: {result2.status.state.value}")

    # 检查是否正确理解了指代
    if result2.status.message:
        for part in result2.status.message.parts:
            if hasattr(part, 'text'):
                text = part.text
                if '600588' in text or 'MACD' in text.upper() or '用友' in text:
                    print("  ✅ 正确理解了'它'指代600588!")
                else:
                    print(f"  ⚠️ 响应: {text[:200]}...")

    # ========================================
    # 场景 3: 切换到新话题（生成新 session_id）
    # ========================================
    print("\n" + "=" * 60)
    print("  场景 3: 新话题 - 分析 002230")
    print("=" * 60)

    session_2 = str(uuid.uuid4())[:8]  # 新会话 ID
    print(f"  session_id: {session_2} (新)")

    task3 = Task(
        id=f"demo_{uuid.uuid4().hex[:8]}",
        metadata={"user_id": user_id, "session_id": session_2}  # 新 session_id
    )
    task3.add_message(Message(role="user", parts=[TextPart(text="分析股票002230")]))

    result3 = await agent.process_task(task3)
    print(f"  状态: {result3.status.state.value}")
    print("  ✅ 新对话开启成功 (与之前的 600588 对话完全隔离)")

    # ========================================
    # 场景 4: 在新对话中追问
    # ========================================
    print("\n" + "=" * 60)
    print("  场景 4: 在新对话中追问 - '它的支撑位'")
    print("=" * 60)

    print(f"  session_id: {session_2} (复用新会话)")

    task4 = Task(
        id=f"demo_{uuid.uuid4().hex[:8]}",
        metadata={"user_id": user_id, "session_id": session_2}  # 复用新会话
    )
    task4.add_message(Message(role="user", parts=[TextPart(text="它的支撑位在哪")]))

    result4 = await agent.process_task(task4)
    print(f"  状态: {result4.status.state.value}")

    # 检查是否指向 002230 而不是 600588
    if result4.status.message:
        for part in result4.status.message.parts:
            if hasattr(part, 'text'):
                text = part.text
                if '002230' in text or '科大讯飞' in text:
                    print("  ✅ 正确！新对话中'它'指代002230，而非旧对话的600588")
                elif '600588' in text:
                    print("  ❌ 错误！会话隔离失败，混淆了旧对话的股票")
                else:
                    print(f"  ⚠️ 响应: {text[:200]}...")

    print("\n" + "=" * 80)
    print("  总结：客户端通过控制 session_id 来管理会话")
    print("=" * 80)
    print("""
  - 新对话: session_id = uuid.uuid4()
  - 继续对话: session_id = 上一次使用的值
  - 切换话题: session_id = uuid.uuid4() (新值)

  Thread ID 格式: {user_id}:{agent_name}:{session_id}
  例如: demo_user:stock_agent:a1b2c3d4
""")


if __name__ == "__main__":
    asyncio.run(demo_session_management())
