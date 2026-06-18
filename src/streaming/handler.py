"""流式输出处理器

将 LangGraph 的流式输出转换为统一的 StreamEvent 格式。
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Union, Sequence

from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.graph.state import CompiledStateGraph

from .types import (
    StreamEvent,
    StreamEventType,
    NodeProgress,
    TokenChunk,
    get_node_description,
)


logger = logging.getLogger(__name__)


class StreamHandler:
    """流式输出处理器

    支持两种模式:
    1. updates 模式 - 输出每个节点的执行进度
    2. messages 模式 - 输出 LLM 的 token 流
    3. 组合模式 - 同时输出进度和 token
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        include_progress: bool = True,
        include_tokens: bool = True,
    ):
        """初始化

        Args:
            graph: LangGraph 编译后的图
            include_progress: 是否包含节点进度
            include_tokens: 是否包含 LLM token 流
        """
        self._graph = graph
        self._include_progress = include_progress
        self._include_tokens = include_tokens

    async def stream(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式执行图

        Args:
            initial_state: 初始状态
            config: LangGraph 配置（包含 thread_id 等）

        Yields:
            StreamEvent: 流式事件
        """
        # 发送元数据事件
        run_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
        yield StreamEvent(
            type=StreamEventType.METADATA,
            metadata={
                "run_id": run_id,
                "thread_id": run_id,
            }
        )

        step = 0
        node_start_times: Dict[str, float] = {}
        final_response: Optional[str] = None

        try:
            # 确定 stream_mode
            stream_modes = []
            if self._include_progress:
                stream_modes.append("updates")
            if self._include_tokens:
                stream_modes.append("messages")

            if not stream_modes:
                stream_modes = ["updates"]  # 默认使用 updates

            # 当只有一个模式时，使用单一模式
            stream_mode = stream_modes[0] if len(stream_modes) == 1 else stream_modes

            # 使用 astream 进行流式处理
            async for chunk in self._graph.astream(
                initial_state,
                config=config,
                stream_mode=stream_mode,  # type: ignore
            ):
                # 处理多模式返回格式: (mode, data)
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    first, second = chunk

                    # 多模式返回: (stream_mode_name, data)
                    if isinstance(first, str) and first in ("updates", "messages", "values"):
                        mode_name = first
                        data = second

                        if mode_name == "updates" and isinstance(data, dict):
                            # updates 模式数据
                            for node_name, output in data.items():
                                if node_name.startswith("__"):
                                    continue
                                async for event in self._process_updates_chunk(
                                    node_name, output, step, node_start_times
                                ):
                                    if event.type == StreamEventType.NODE_END:
                                        step = event.progress.step if event.progress else step
                                    yield event
                                    # 记录最终响应
                                    if isinstance(output, dict) and "final_response" in output:
                                        final_response = output["final_response"]
                                step += 1

                        elif mode_name == "messages":
                            # messages 模式数据: (AIMessageChunk, metadata) 或 AIMessageChunk
                            if isinstance(data, tuple) and len(data) == 2:
                                message, metadata = data
                                if hasattr(message, 'content') and message.content:
                                    yield StreamEvent(
                                        type=StreamEventType.TOKEN,
                                        token=TokenChunk(
                                            content=str(message.content),
                                            node_name=metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown",
                                            step=metadata.get("langgraph_step", step) if isinstance(metadata, dict) else step,
                                            is_final=False,
                                        )
                                    )
                    else:
                        # 单模式 messages 返回: (AIMessageChunk, metadata)
                        message, metadata = first, second
                        if hasattr(message, 'content') and message.content:
                            yield StreamEvent(
                                type=StreamEventType.TOKEN,
                                token=TokenChunk(
                                    content=str(message.content),
                                    node_name=metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown",
                                    step=metadata.get("langgraph_step", step) if isinstance(metadata, dict) else step,
                                    is_final=False,
                                )
                            )

                elif isinstance(chunk, dict):
                    # 单模式 updates 返回: {"node_name": output}
                    for node_name, output in chunk.items():
                        if node_name.startswith("__"):
                            continue
                        step += 1
                        async for event in self._process_updates_chunk(
                            node_name, output, step, node_start_times
                        ):
                            yield event
                        # 记录最终响应
                        if isinstance(output, dict) and "final_response" in output:
                            final_response = output["final_response"]

            # 发送完成事件
            yield StreamEvent(
                type=StreamEventType.COMPLETE,
                result=final_response,
                metadata={"total_steps": step},
            )

        except Exception as e:
            logger.error(f"流式执行失败: {e}")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error=str(e),
            )

    async def _process_updates_chunk(
        self,
        node_name: str,
        output: Any,
        step: int,
        node_start_times: Dict[str, float],
    ) -> AsyncGenerator[StreamEvent, None]:
        """处理 updates 模式的 chunk"""
        node_start_times[node_name] = time.time()

        # 发送节点开始事件
        yield StreamEvent(
            type=StreamEventType.NODE_START,
            progress=NodeProgress(
                node_name=node_name,
                step=step,
                status="running",
                message=f"正在执行: {get_node_description(node_name)}",
            )
        )

        # 提取关键输出信息
        output_summary = self._extract_output_summary(node_name, output)

        # 计算耗时
        duration_ms = None
        if node_name in node_start_times:
            duration_ms = int((time.time() - node_start_times[node_name]) * 1000)

        # 发送节点完成事件
        yield StreamEvent(
            type=StreamEventType.NODE_END,
            progress=NodeProgress(
                node_name=node_name,
                step=step,
                status="completed",
                output=output_summary,
                duration_ms=duration_ms,
                message=f"完成: {get_node_description(node_name)}",
            )
        )

    def _extract_output_summary(
        self,
        node_name: str,
        output: Any,
    ) -> Optional[Dict[str, Any]]:
        """提取节点输出摘要

        只返回关键信息，避免返回大量数据。
        """
        if not isinstance(output, dict):
            return None

        summary = {}

        # 根据节点类型提取关键信息
        if node_name in ("parse_intent", "parse"):
            # 意图解析
            if "task_type" in output:
                summary["task_type"] = output["task_type"]
            if "stock_code" in output:
                summary["stock_code"] = output["stock_code"]

        elif node_name == "llm_route":
            # LLM 路由
            if "task_type" in output:
                summary["task_type"] = output["task_type"]
            if "position_operation" in output:
                summary["operation"] = output["position_operation"]

        elif node_name in ("fetch_stock_data", "fetch_fund_data", "fetch_data"):
            # 数据获取
            if "stock_data" in output:
                summary["data_fetched"] = True
            if "stock_name" in output:
                summary["stock_name"] = output["stock_name"]

        elif node_name == "fetch_market_data":
            # 市场数据
            if "index_data" in output:
                summary["indices_fetched"] = True

        elif node_name in ("position_operation", "preference_operation"):
            # 操作结果
            if "status" in output:
                summary["status"] = output["status"]

        elif node_name == "generate_report":
            # 报告生成
            if "final_response" in output:
                # 只返回前 100 个字符作为预览
                response = output["final_response"]
                summary["preview"] = response[:100] + "..." if len(response) > 100 else response

        elif node_name in ("add_user_message", "add_ai_response"):
            # 消息记录
            if "messages" in output:
                summary["messages_updated"] = True

        elif node_name == "summarize_history":
            # 历史摘要
            if "messages" in output:
                summary["history_summarized"] = True

        # 状态更新
        if "status" in output:
            summary["status"] = output["status"]

        return summary if summary else None


async def stream_with_progress(
    graph: CompiledStateGraph,
    initial_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[StreamEvent, None]:
    """便捷函数: 只流式输出进度

    Args:
        graph: LangGraph 编译后的图
        initial_state: 初始状态
        config: LangGraph 配置

    Yields:
        StreamEvent: 进度事件
    """
    handler = StreamHandler(graph, include_progress=True, include_tokens=False)
    async for event in handler.stream(initial_state, config):
        yield event


async def stream_with_tokens(
    graph: CompiledStateGraph,
    initial_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[StreamEvent, None]:
    """便捷函数: 只流式输出 LLM tokens

    Args:
        graph: LangGraph 编译后的图
        initial_state: 初始状态
        config: LangGraph 配置

    Yields:
        StreamEvent: Token 事件
    """
    handler = StreamHandler(graph, include_progress=False, include_tokens=True)
    async for event in handler.stream(initial_state, config):
        yield event


async def stream_full(
    graph: CompiledStateGraph,
    initial_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[StreamEvent, None]:
    """便捷函数: 流式输出进度和 tokens

    Args:
        graph: LangGraph 编译后的图
        initial_state: 初始状态
        config: LangGraph 配置

    Yields:
        StreamEvent: 进度和 Token 事件
    """
    handler = StreamHandler(graph, include_progress=True, include_tokens=True)
    async for event in handler.stream(initial_state, config):
        yield event
