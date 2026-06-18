#!/usr/bin/env python3
"""流式输出测试脚本

测试 JarvisAgent 的流式输出功能。

Usage:
    python scripts/test_streaming.py "分析股票600588"
    python scripts/test_streaming.py --agent stock "分析茅台股票"
    python scripts/test_streaming.py --agent fund "分析基金008089"
"""

import argparse
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streaming import StreamEvent, StreamEventType


async def test_jarvis_stream(query: str):
    """测试 JarvisAgent 流式输出"""
    from src.agents.jarvis import JarvisAgent
    from src.config import agent_config

    print(f"\n{'='*60}")
    print(f"🤖 测试 Jarvis 流式输出")
    print(f"📝 查询: {query}")
    print(f"{'='*60}\n")

    jarvis = JarvisAgent(sub_agent_urls=agent_config.sub_agents)

    try:
        async for event in jarvis.arun_stream(query, include_progress=True, include_tokens=True):
            _print_event(event)
    finally:
        await jarvis.close()

    print(f"\n{'='*60}")
    print("✅ 测试完成")
    print(f"{'='*60}\n")


async def test_stock_agent_stream(query: str):
    """测试 StockAgent 流式输出"""
    from src.agents.stock_agent import StockAgent
    from src.a2a import Task, Message, TextPart
    from src.config import postgres_config
    from src.storage.checkpoint import CheckpointManager

    print(f"\n{'='*60}")
    print(f"📈 测试 StockAgent 流式输出")
    print(f"📝 查询: {query}")
    print(f"{'='*60}\n")

    # 创建 CheckpointManager
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
    except Exception:
        pass

    agent = StockAgent(checkpoint_manager=checkpoint_manager, enable_mcp_search=False)

    try:
        # 初始化
        await agent.initialize()

        # 创建任务 - 使用正确的格式
        task = Task(
            id="test-stream-001",
            history=[
                Message(
                    role="user",
                    parts=[TextPart(text=query)]
                )
            ],
            metadata={"user_id": "test_user", "session_id": "test_session"}
        )

        async for event in agent.process_task_stream(task, include_progress=True, include_tokens=True):
            _print_event(event)

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("✅ 测试完成")
    print(f"{'='*60}\n")


async def test_fund_agent_stream(query: str):
    """测试 FundAgent 流式输出"""
    from src.agents.fund_agent import FundAgent
    from src.a2a import Task, Message, TextPart
    from src.config import postgres_config
    from src.storage.checkpoint import CheckpointManager

    print(f"\n{'='*60}")
    print(f"💰 测试 FundAgent 流式输出")
    print(f"📝 查询: {query}")
    print(f"{'='*60}\n")

    # 创建 CheckpointManager
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
    except Exception:
        pass

    agent = FundAgent(checkpoint_manager=checkpoint_manager)

    try:
        # 初始化
        await agent.initialize()

        # 创建任务 - 使用正确的格式
        task = Task(
            id="test-stream-001",
            history=[
                Message(
                    role="user",
                    parts=[TextPart(text=query)]
                )
            ],
            metadata={"user_id": "test_user", "session_id": "test_session"}
        )

        async for event in agent.process_task_stream(task, include_progress=True, include_tokens=True):
            _print_event(event)

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("✅ 测试完成")
    print(f"{'='*60}\n")


def _print_event(event: StreamEvent):
    """打印流式事件"""
    if event.type == StreamEventType.METADATA:
        print(f"📋 元数据: {event.metadata}")

    elif event.type == StreamEventType.NODE_START:
        if event.progress:
            print(f"\n🔄 [{event.progress.step}] {event.progress.message}")

    elif event.type == StreamEventType.NODE_END:
        if event.progress:
            duration = f" ({event.progress.duration_ms}ms)" if event.progress.duration_ms else ""
            print(f"✅ [{event.progress.step}] {event.progress.message}{duration}")
            if event.progress.output:
                for key, value in event.progress.output.items():
                    print(f"   📊 {key}: {value}")

    elif event.type == StreamEventType.TOKEN:
        if event.token:
            # Token 输出不换行
            print(event.token.content, end="", flush=True)

    elif event.type == StreamEventType.TOKEN_END:
        print()  # 换行

    elif event.type == StreamEventType.COMPLETE:
        print(f"\n\n🎉 执行完成")
        if event.metadata:
            print(f"   📊 总步骤: {event.metadata.get('total_steps', 'N/A')}")
        if event.result:
            print(f"\n📄 最终结果:\n{'-'*40}")
            print(event.result[:500])  # 只显示前 500 字符
            if len(event.result) > 500:
                print(f"... (共 {len(event.result)} 字符)")

    elif event.type == StreamEventType.ERROR:
        print(f"\n❌ 错误: {event.error}")


def main():
    parser = argparse.ArgumentParser(description="测试流式输出功能")
    parser.add_argument("query", nargs="?", default="分析股票600588", help="查询内容")
    parser.add_argument("--agent", choices=["jarvis", "stock", "fund"], default="jarvis", help="测试的 Agent")
    args = parser.parse_args()

    if args.agent == "jarvis":
        asyncio.run(test_jarvis_stream(args.query))
    elif args.agent == "stock":
        asyncio.run(test_stock_agent_stream(args.query))
    elif args.agent == "fund":
        asyncio.run(test_fund_agent_stream(args.query))


if __name__ == "__main__":
    main()
