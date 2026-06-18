#!/usr/bin/env python
"""测试改进后的多轮对话"""

import asyncio
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_improved_multi_turn():
    """测试改进后的多轮对话"""

    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart
    from src.storage import LangGraphCheckpointer

    # 使用唯一的 user_id 避免历史数据干扰
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"

    print("=" * 80)
    print(f"  测试改进后的多轮对话 (user_id: {user_id})")
    print("=" * 80)

    agent = StockAgent()

    # 初始化 checkpointer
    print("\n[1] 初始化 checkpointer...")
    await agent.initialize()
    print(f"    ✅ checkpointer 已初始化")

    # 第一轮
    print("\n" + "=" * 80)
    print("  第一轮: '分析股票600588'")
    print("=" * 80)

    task1 = Task(id=f"test_{uuid.uuid4().hex[:8]}", metadata={"user_id": user_id})
    task1.add_message(Message(role="user", parts=[TextPart(text="分析股票600588")]))

    result1 = await agent.process_task(task1)
    print(f"\n状态: {result1.status.state.value}")

    if result1.status.state.value == "completed":
        print("✅ 第一轮成功")
    else:
        print(f"❌ 第一轮失败")
        return

    # 第二轮
    print("\n" + "=" * 80)
    print("  第二轮: '它的MACD怎么样'")
    print("=" * 80)

    task2 = Task(id=f"test_{uuid.uuid4().hex[:8]}", metadata={"user_id": user_id})
    task2.add_message(Message(role="user", parts=[TextPart(text="它的MACD怎么样")]))

    result2 = await agent.process_task(task2)
    print(f"\n状态: {result2.status.state.value}")

    # 检查响应
    if result2.status.message:
        msg = result2.status.message
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'text'):
                    text = part.text
                    print(f"\n响应内容 (前500字):\n{text[:500]}...")

                    # 判断是否成功
                    if '600588' in text or 'MACD' in text.upper() or '用友' in text:
                        print("\n✅ 第二轮成功 - 正确理解了'它'指代600588!")
                    elif '请提供' in text or '请问' in text:
                        print("\n❌ 第二轮失败 - LLM 没有理解指代")
                    else:
                        print("\n⚠️ 结果不确定，请检查响应内容")


if __name__ == "__main__":
    asyncio.run(test_improved_multi_turn())
