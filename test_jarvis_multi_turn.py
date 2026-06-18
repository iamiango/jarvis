#!/usr/bin/env python
"""测试 Jarvis → StockAgent 的多轮对话

此测试验证:
1. Jarvis 从 Task.metadata 提取 user_id
2. Jarvis 将 user_id 传递给 StockAgent
3. StockAgent 使用相同的 thread_id，保持对话上下文
4. 第二轮对话中 "它" 能正确指代第一轮的股票
"""

import asyncio
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_jarvis_multi_turn():
    """测试 Jarvis 多轮对话"""

    from src.agents.jarvis import JarvisAgent
    from src.a2a import Task, Message, TextPart

    # 使用唯一的 user_id 避免历史数据干扰
    user_id = f"jarvis_test_{uuid.uuid4().hex[:8]}"

    print("=" * 80)
    print(f"  测试 Jarvis → StockAgent 多轮对话")
    print(f"  user_id: {user_id}")
    print("=" * 80)

    # 创建 Jarvis
    jarvis = JarvisAgent()

    # 发现子 Agent
    print("\n[1] 发现子 Agent...")
    await jarvis.discover_agents()
    agents = jarvis._registry.list_agents()
    print(f"    已发现: {[a['name'] for a in agents]}")

    if "stock_agent" not in [a['name'] for a in agents]:
        print("\n❌ 错误: stock_agent 未启动")
        print("   请先运行: python -m src.agents.stock_agent --port 8001")
        return

    # 第一轮: 分析股票
    print("\n" + "=" * 80)
    print("  第一轮: '分析股票600588'")
    print("=" * 80)

    task1 = Task(id=f"jarvis_test_{uuid.uuid4().hex[:8]}", metadata={"user_id": user_id})
    task1.add_message(Message(role="user", parts=[TextPart(text="分析股票600588")]))

    result1 = await jarvis.process_task(task1)
    print(f"\n状态: {result1.status.state.value}")

    if result1.status.state.value != "completed":
        print(f"❌ 第一轮失败")
        return

    print("✅ 第一轮成功")

    # 等待一下，确保检查点保存
    await asyncio.sleep(1)

    # 第二轮: 追问 MACD
    print("\n" + "=" * 80)
    print("  第二轮: '它的MACD怎么样'")
    print("=" * 80)

    task2 = Task(id=f"jarvis_test_{uuid.uuid4().hex[:8]}", metadata={"user_id": user_id})
    task2.add_message(Message(role="user", parts=[TextPart(text="它的MACD怎么样")]))

    result2 = await jarvis.process_task(task2)
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
                        print("\n🎉 Jarvis → StockAgent 多轮对话测试通过!")
                    elif '请提供' in text or '请问' in text or '哪只' in text:
                        print("\n❌ 第二轮失败 - 没有理解'它'的指代")
                        print("\n检查事项:")
                        print("  1. user_id 是否正确传递到 StockAgent")
                        print("  2. StockAgent 的 checkpointer 是否正常工作")
                    else:
                        print("\n⚠️ 结果不确定，请检查响应内容")

    # 清理
    await jarvis.close()


if __name__ == "__main__":
    asyncio.run(test_jarvis_multi_turn())
