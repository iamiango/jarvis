"""示例 Skills"""
from typing import Any, Dict
from .base import BaseSkill, SkillOutput


class CalculatorSkill(BaseSkill):
    """计算器 Skill"""

    name = "calculator"
    description = "执行基本数学运算 (add, subtract, multiply, divide)"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "运算类型"
                },
                "a": {
                    "type": "number",
                    "description": "第一个操作数"
                },
                "b": {
                    "type": "number",
                    "description": "第二个操作数"
                }
            },
            "required": ["operation", "a", "b"]
        }

    async def execute(self, operation: str, a: float, b: float, **kwargs) -> SkillOutput:
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return SkillOutput(success=False, result=None, error="除数不能为零")
                result = a / b
            else:
                return SkillOutput(success=False, result=None, error=f"未知运算: {operation}")

            return SkillOutput(success=True, result=result)
        except Exception as e:
            return SkillOutput(success=False, result=None, error=str(e))


class SearchSkill(BaseSkill):
    """搜索 Skill（示例）"""

    name = "search"
    description = "搜索信息（模拟实现）"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, **kwargs) -> SkillOutput:
        # 模拟搜索结果
        return SkillOutput(
            success=True,
            result=f"搜索 '{query}' 的模拟结果: [结果1, 结果2, 结果3]"
        )


class WeatherSkill(BaseSkill):
    """天气查询 Skill（示例）"""

    name = "weather"
    description = "查询天气信息（模拟实现）"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }

    async def execute(self, city: str, **kwargs) -> SkillOutput:
        # 模拟天气数据
        return SkillOutput(
            success=True,
            result=f"{city} 天气: 晴, 温度: 25°C, 湿度: 60%"
        )
