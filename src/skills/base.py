"""Skill 基类和管理器"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SkillInput(BaseModel):
    """Skill 输入模型"""
    name: str
    args: Dict[str, Any] = {}


class SkillOutput(BaseModel):
    """Skill 输出模型"""
    success: bool
    result: Any
    error: Optional[str] = None


class BaseSkill(ABC):
    """Skill 基类"""

    name: str = "base_skill"
    description: str = "Base skill description"

    @abstractmethod
    async def execute(self, **kwargs) -> SkillOutput:
        """执行 Skill"""
        pass

    def to_tool_schema(self) -> Dict[str, Any]:
        """转换为 Tool Schema 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters()
        }

    def get_parameters(self) -> Dict[str, Any]:
        """获取参数 Schema"""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class SkillManager:
    """Skill 管理器"""

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """注册 Skill"""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """注销 Skill"""
        if name in self._skills:
            del self._skills[name]

    def get(self, name: str) -> Optional[BaseSkill]:
        """获取 Skill"""
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有 Skill"""
        return list(self._skills.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """获取所有 Skill 的 Schema"""
        return [skill.to_tool_schema() for skill in self._skills.values()]

    async def execute(self, name: str, **kwargs) -> SkillOutput:
        """执行指定 Skill"""
        skill = self.get(name)
        if not skill:
            return SkillOutput(
                success=False,
                result=None,
                error=f"Skill '{name}' not found"
            )
        try:
            return await skill.execute(**kwargs)
        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=str(e)
            )


# 全局 Skill 管理器实例
skill_manager = SkillManager()
