"""LangGraph Studio 入口 - FundAgent 图

此模块为 LangGraph Studio 提供 FundAgent 图的入口。
由于 LangGraph CLI 直接加载文件时无法识别包结构，
需要使用 sys.path 处理来支持导入。

使用方式:
    langgraph dev --config langgraph.json
"""

import os
import sys

# 添加项目根目录到 Python 路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 现在可以使用绝对导入
from src.agents.fund_agent import get_graph

# 导出图供 LangGraph Studio 使用
graph = get_graph()
