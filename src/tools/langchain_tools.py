"""LangChain 工具定义 - 整合 Skills 和 MCP"""
from typing import Any, Dict, List, Optional, Callable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model
import asyncio

from ..skills import skill_manager, SkillOutput
from ..mcp import mcp_manager


def create_skill_tools() -> List[BaseTool]:
    """为每个注册的 Skill 创建对应的 LangChain Tool"""
    tools = []

    for name in skill_manager.list_skills():
        skill = skill_manager.get(name)
        if skill:
            # 获取参数信息
            params = skill.get_parameters()
            properties = params.get("properties", {})
            required = params.get("required", [])

            # 动态创建参数模型
            field_definitions = {}
            for prop_name, prop_info in properties.items():
                prop_type = prop_info.get("type", "string")
                prop_desc = prop_info.get("description", "")

                # 映射类型
                python_type = str
                if prop_type == "number":
                    python_type = float
                elif prop_type == "integer":
                    python_type = int
                elif prop_type == "boolean":
                    python_type = bool

                # 设置默认值
                if prop_name in required:
                    field_definitions[prop_name] = (python_type, Field(description=prop_desc))
                else:
                    field_definitions[prop_name] = (Optional[python_type], Field(default=None, description=prop_desc))

            # 使用工厂函数创建闭包
            def make_tool_func(skill_name: str):
                def tool_func(**kwargs) -> str:
                    """执行 Skill"""
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                skill_manager.execute(skill_name, **kwargs)
                            )
                            if result.success:
                                return str(result.result)
                            return f"Error: {result.error}"
                        finally:
                            loop.close()
                    except Exception as e:
                        return f"Error: {str(e)}"
                return tool_func

            tool = StructuredTool.from_function(
                func=make_tool_func(name),
                name=f"skill_{name}",
                description=skill.description,
            )
            tools.append(tool)

    return tools


def create_mcp_tools() -> List[BaseTool]:
    """为每个 MCP Tool 创建对应的 LangChain Tool"""
    tools = []
    schemas = mcp_manager.get_all_tool_schemas()

    for schema in schemas:
        tool_name = schema["name"]
        tool_desc = schema["description"]

        def make_mcp_func(t_name: str):
            def mcp_func(**kwargs) -> str:
                """执行 MCP Tool"""
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            mcp_manager.call_tool(t_name, kwargs)
                        )
                        if result.get("success"):
                            return str(result.get("result"))
                        return f"Error: {result.get('error')}"
                    finally:
                        loop.close()
                except Exception as e:
                    return f"Error: {str(e)}"
            return mcp_func

        tool = StructuredTool.from_function(
            func=make_mcp_func(tool_name),
            name=tool_name,
            description=tool_desc,
        )
        tools.append(tool)

    return tools


def get_all_tools() -> List[BaseTool]:
    """获取所有可用的工具（Skills + MCP Tools）"""
    return create_skill_tools() + create_mcp_tools()
