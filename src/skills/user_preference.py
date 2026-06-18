"""User Preference Skills - 用户偏好管理

提供用户偏好的保存和获取功能，用于长期记忆存储。
"""
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillOutput
from ..storage.checkpoint import CheckpointManager


class SaveUserPreferenceSkill(BaseSkill):
    """保存用户偏好到长期记忆

    将用户偏好保存到 PostgreSQL 数据库，支持 upsert 更新。
    """

    name = "save_user_preference"
    description = "保存用户偏好到长期记忆存储，支持按类别管理偏好"

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        """初始化

        Args:
            checkpoint_manager: CheckpointManager 实例，如果为 None 则需要通过 execute 参数传入
        """
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户标识符"
                },
                "category": {
                    "type": "string",
                    "description": "偏好类别: stock, fund, email, general",
                    "enum": ["stock", "fund", "email", "general"]
                },
                "key": {
                    "type": "string",
                    "description": "偏好键名，如 risk_tolerance, preferred_sectors"
                },
                "value": {
                    "description": "偏好值，可以是字符串、数字、数组或对象"
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度 (0.0-1.0)，默认 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "source": {
                    "type": "string",
                    "description": "来源: explicit (用户明确设置) 或 inferred (系统推断)",
                    "enum": ["explicit", "inferred"]
                }
            },
            "required": ["user_id", "category", "key", "value"]
        }

    async def execute(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs
    ) -> SkillOutput:
        """保存用户偏好

        Args:
            user_id: 用户标识符
            category: 偏好类别
            key: 偏好键名
            value: 偏好值
            confidence: 置信度
            source: 来源
            checkpoint_manager: CheckpointManager 实例（可选，覆盖构造时的实例）

        Returns:
            SkillOutput 包含保存结果
        """
        try:
            manager = checkpoint_manager or self._checkpoint_manager
            if not manager:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="CheckpointManager 未配置"
                )

            # 验证类别
            valid_categories = ["stock", "fund", "email", "general"]
            if category not in valid_categories:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"无效的类别: {category}，有效类别: {valid_categories}"
                )

            # 保存偏好
            await manager.save_preference(
                user_id=user_id,
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source=source,
            )

            return SkillOutput(
                success=True,
                result={
                    "message": f"偏好已保存: {category}/{key}",
                    "user_id": user_id,
                    "category": category,
                    "key": key,
                    "value": value,
                    "confidence": confidence,
                    "source": source,
                }
            )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"保存偏好失败: {str(e)}"
            )


class GetUserPreferencesSkill(BaseSkill):
    """获取用户偏好

    从 PostgreSQL 数据库获取用户偏好，支持按类别筛选。
    """

    name = "get_user_preferences"
    description = "获取用户偏好列表，支持按类别筛选"

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        """初始化

        Args:
            checkpoint_manager: CheckpointManager 实例
        """
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户标识符"
                },
                "category": {
                    "type": "string",
                    "description": "偏好类别（可选）: stock, fund, email, general",
                    "enum": ["stock", "fund", "email", "general"]
                },
                "as_context": {
                    "type": "boolean",
                    "description": "是否以 LLM 上下文格式返回，默认 False"
                }
            },
            "required": ["user_id"]
        }

    async def execute(
        self,
        user_id: str,
        category: Optional[str] = None,
        as_context: bool = False,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs
    ) -> SkillOutput:
        """获取用户偏好

        Args:
            user_id: 用户标识符
            category: 偏好类别（可选）
            as_context: 是否以 LLM 上下文格式返回
            checkpoint_manager: CheckpointManager 实例（可选）

        Returns:
            SkillOutput 包含偏好列表或格式化上下文
        """
        try:
            manager = checkpoint_manager or self._checkpoint_manager
            if not manager:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="CheckpointManager 未配置"
                )

            if as_context:
                # 返回格式化的上下文字符串
                context = await manager.get_preferences_as_context(user_id, category)
                return SkillOutput(
                    success=True,
                    result=context
                )
            else:
                # 返回偏好列表
                preferences = await manager.get_preferences(user_id, category)
                return SkillOutput(
                    success=True,
                    result={
                        "user_id": user_id,
                        "category": category,
                        "preferences": preferences,
                        "count": len(preferences),
                    }
                )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"获取偏好失败: {str(e)}"
            )


class DeleteUserPreferenceSkill(BaseSkill):
    """删除用户偏好"""

    name = "delete_user_preference"
    description = "删除指定的用户偏好"

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户标识符"
                },
                "category": {
                    "type": "string",
                    "description": "偏好类别: stock, fund, email, general"
                },
                "key": {
                    "type": "string",
                    "description": "偏好键名"
                }
            },
            "required": ["user_id", "category", "key"]
        }

    async def execute(
        self,
        user_id: str,
        category: str,
        key: str,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs
    ) -> SkillOutput:
        """删除用户偏好

        Args:
            user_id: 用户标识符
            category: 偏好类别
            key: 偏好键名
            checkpoint_manager: CheckpointManager 实例（可选）

        Returns:
            SkillOutput 包含删除结果
        """
        try:
            manager = checkpoint_manager or self._checkpoint_manager
            if not manager:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="CheckpointManager 未配置"
                )

            deleted = await manager.delete_preference(user_id, category, key)

            if deleted:
                return SkillOutput(
                    success=True,
                    result={
                        "message": f"偏好已删除: {category}/{key}",
                        "user_id": user_id,
                        "category": category,
                        "key": key,
                    }
                )
            else:
                return SkillOutput(
                    success=True,
                    result={
                        "message": f"偏好不存在: {category}/{key}",
                        "user_id": user_id,
                        "category": category,
                        "key": key,
                    }
                )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"删除偏好失败: {str(e)}"
            )
