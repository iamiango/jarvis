"""流式输出类型定义"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List
from datetime import datetime


class StreamEventType(str, Enum):
    """流式事件类型"""
    # 进度事件 (stream_mode="updates")
    NODE_START = "node_start"       # 节点开始执行
    NODE_END = "node_end"           # 节点执行完成
    NODE_ERROR = "node_error"       # 节点执行出错

    # Token 事件 (stream_mode="messages")
    TOKEN = "token"                 # LLM token 输出
    TOKEN_END = "token_end"         # LLM 输出完成

    # 元数据事件
    METADATA = "metadata"           # 元数据（run_id, thread_id 等）

    # 完成事件
    COMPLETE = "complete"           # 整体完成
    ERROR = "error"                 # 整体出错


@dataclass
class NodeProgress:
    """节点执行进度"""
    node_name: str                  # 节点名称
    step: int                       # 步骤序号
    status: str                     # pending / running / completed / failed
    output: Optional[Dict[str, Any]] = None  # 节点输出
    duration_ms: Optional[int] = None  # 执行耗时（毫秒）
    message: Optional[str] = None   # 可读消息


@dataclass
class TokenChunk:
    """Token 输出块"""
    content: str                    # Token 内容
    node_name: str                  # 产生 Token 的节点
    step: int                       # 步骤序号
    is_final: bool = False          # 是否是最后一个 token


@dataclass
class StreamEvent:
    """流式事件"""
    type: StreamEventType           # 事件类型
    timestamp: datetime = field(default_factory=datetime.now)

    # 进度事件数据
    progress: Optional[NodeProgress] = None

    # Token 事件数据
    token: Optional[TokenChunk] = None

    # 元数据
    metadata: Optional[Dict[str, Any]] = None

    # 错误信息
    error: Optional[str] = None

    # 最终结果
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        data = {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.progress:
            data["progress"] = {
                "node_name": self.progress.node_name,
                "step": self.progress.step,
                "status": self.progress.status,
                "message": self.progress.message,
            }
            if self.progress.output is not None:
                data["progress"]["output"] = self.progress.output
            if self.progress.duration_ms is not None:
                data["progress"]["duration_ms"] = self.progress.duration_ms

        if self.token:
            data["token"] = {
                "content": self.token.content,
                "node_name": self.token.node_name,
                "step": self.token.step,
                "is_final": self.token.is_final,
            }

        if self.metadata:
            data["metadata"] = self.metadata

        if self.error:
            data["error"] = self.error

        if self.result:
            data["result"] = self.result

        return data

    def to_sse(self) -> str:
        """转换为 SSE 格式"""
        import json
        return f"data: {json.dumps(self.to_dict(), ensure_ascii=False)}\n\n"


# 节点名称到中文描述的映射
NODE_DESCRIPTIONS: Dict[str, str] = {
    # StockAgent 节点
    "add_user_message": "记录用户消息",
    "summarize_history": "整理对话历史",
    "parse_intent": "解析用户意图",
    "llm_route": "LLM 路由决策",
    "fetch_stock_data": "获取股票数据",
    "fetch_market_data": "获取市场数据",
    "position_operation": "执行持仓操作",
    "preference_operation": "处理偏好设置",
    "generate_report": "生成分析报告",
    "add_ai_response": "记录 AI 响应",

    # FundAgent 节点
    "fetch_fund_data": "获取基金数据",
    "fetch_data": "获取数据",

    # JarvisAgent 节点
    "analyze": "分析用户意图",
    "route": "路由决策",
    "delegate": "委派任务",
    "aggregate": "汇总结果",
    "respond": "生成响应",

    # 通用
    "parse": "解析输入",
    "__start__": "开始",
    "__end__": "结束",
}


def get_node_description(node_name: str) -> str:
    """获取节点的中文描述"""
    return NODE_DESCRIPTIONS.get(node_name, node_name)
