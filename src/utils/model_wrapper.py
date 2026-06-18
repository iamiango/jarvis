"""Model Wrapper - LLM 调用装饰器

提供装饰器在调用 LLM 前注入用户上下文：
- @wrap_model_call: 简单版，只注入偏好
- @dynamic_prompt: 完整版，支持偏好、持仓、新闻等多种上下文
"""
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..storage.checkpoint import CheckpointManager


# =============================================================================
# 数据类型定义
# =============================================================================

@dataclass
class PromptContext:
    """动态 Prompt 上下文数据"""
    user_id: str = "default"
    category: str = "stock"

    # 上下文数据
    preferences: str = ""
    positions: str = ""
    news: str = ""

    # 股票相关
    stock_code: Optional[str] = None
    stock_name: Optional[str] = None
    current_price: Optional[float] = None

    # 额外数据
    extra: Dict[str, Any] = field(default_factory=dict)

    def has_positions(self) -> bool:
        return bool(self.positions)

    def has_preferences(self) -> bool:
        return bool(self.preferences)

    def has_news(self) -> bool:
        return bool(self.news)


# =============================================================================
# @dynamic_prompt 装饰器
# =============================================================================

def dynamic_prompt(
    category: str = "stock",
    include_preferences: bool = True,
    include_positions: bool = True,
    include_news: bool = False,
    position_template_path: Optional[Union[str, Path]] = None,
    stock_code_param: str = "stock_code",
    user_id_param: str = "user_id",
):
    """动态 Prompt 装饰器 - 自动获取持仓、偏好并组装到 prompt

    在被装饰方法执行前，自动从数据库获取用户上下文，
    并通过 `_prompt_context: PromptContext` 参数传入。

    Args:
        category: 偏好类别 ('stock', 'fund')
        include_preferences: 是否获取用户偏好
        include_positions: 是否获取用户持仓
        include_news: 是否获取相关新闻（需要 MCP Search）
        position_template_path: 持仓建议模板路径
        stock_code_param: 股票代码参数名
        user_id_param: 用户 ID 参数名

    Usage:
        class StockReportGeneratorSkill(BaseSkill):
            @dynamic_prompt(category="stock", include_positions=True)
            async def translate_to_chinese(
                self,
                english_report: str,
                _prompt_context: PromptContext,  # 自动注入
                **kwargs
            ) -> str:
                # 直接使用 context 中的数据
                if _prompt_context.has_positions():
                    # 构建持仓建议部分
                    ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取 self 实例
            self_instance = args[0] if args else None

            # 获取 CheckpointManager
            manager = _get_checkpoint_manager(self_instance, kwargs)

            # 获取参数值
            user_id = kwargs.get(user_id_param, "default")
            stock_code = kwargs.get(stock_code_param)

            # 尝试从 data 参数提取 stock_code
            if not stock_code:
                data = kwargs.get("data", {})
                if isinstance(data, dict):
                    basic = data.get("basic_info", {})
                    stock_code = basic.get("symbol")

            # 构建上下文
            context = PromptContext(
                user_id=user_id,
                category=category,
                stock_code=stock_code,
            )

            # 获取偏好
            if include_preferences and manager:
                try:
                    context.preferences = await manager.get_preferences_as_context(
                        user_id, category
                    )
                    if context.preferences:
                        print(f"📋 已加载用户偏好: {user_id}")
                except Exception as e:
                    print(f"⚠️ 获取用户偏好失败: {e}")

            # 获取持仓
            if include_positions and manager and stock_code:
                try:
                    context.positions = await manager.get_stock_positions_as_context(
                        user_id, stock_code
                    )
                    if context.positions:
                        print(f"💼 已加载用户持仓: {stock_code}")
                except Exception as e:
                    print(f"⚠️ 获取持仓上下文失败: {e}")

            # 获取新闻（如果启用且有 MCP Search）
            if include_news and stock_code:
                context.news = await _get_news_context(self_instance, stock_code)

            # 从 data 参数提取更多信息
            data = kwargs.get("data", {})
            if isinstance(data, dict):
                basic = data.get("basic_info", {})
                quote = data.get("latest_quote", {})
                context.stock_name = basic.get("name")
                context.current_price = quote.get("close")

            # 加载持仓模板
            if position_template_path and context.has_positions():
                context.extra["position_template"] = _load_template(position_template_path)

            # 注入 context 到 kwargs
            kwargs["_prompt_context"] = context

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def _get_checkpoint_manager(
    self_instance: Any,
    kwargs: Dict[str, Any]
) -> Optional[CheckpointManager]:
    """从实例或 kwargs 获取 CheckpointManager"""
    # 从 kwargs 获取
    manager = kwargs.get("checkpoint_manager")
    if manager:
        return manager

    # 从实例属性获取
    if self_instance:
        for attr_name in ["_checkpoint_manager", "checkpoint_manager", "_cm"]:
            if hasattr(self_instance, attr_name):
                manager = getattr(self_instance, attr_name)
                if manager:
                    return manager

    return None


async def _get_news_context(self_instance: Any, stock_code: str) -> str:
    """获取新闻上下文"""
    if not self_instance:
        return ""

    # 尝试获取 MCP Search Skill
    mcp_search = getattr(self_instance, "_mcp_search_skill", None)
    if not mcp_search:
        # 尝试懒加载
        get_mcp = getattr(self_instance, "_get_mcp_search_skill", None)
        if get_mcp:
            try:
                mcp_search = await get_mcp()
            except Exception:
                pass

    if not mcp_search:
        return ""

    try:
        stock_name = getattr(self_instance, "_current_stock_name", stock_code)
        result = await mcp_search.search_stock_news(stock_name, stock_code)
        if result.success and result.result:
            print(f"📰 [{stock_code}] 获取到新闻信息")
            return result.result
    except Exception as e:
        print(f"⚠️ [{stock_code}] 新闻搜索失败: {e}")

    return ""


def _load_template(path: Union[str, Path]) -> str:
    """加载模板文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ 加载模板失败: {e}")
        return ""


# =============================================================================
# Prompt 构建辅助函数
# =============================================================================

def build_prompt_sections(
    context: PromptContext,
    position_template: Optional[str] = None,
) -> Dict[str, str]:
    """根据上下文构建 prompt 各部分

    Args:
        context: PromptContext 实例
        position_template: 持仓建议模板（可选）

    Returns:
        Dict 包含 'news_section', 'position_section' 等
    """
    sections = {}

    # 新闻部分
    if context.has_news():
        sections["news_section"] = f"""
---
**最新市场新闻**（请在翻译时整合到报告中）：

{context.news[:1500]}
---
"""
    else:
        sections["news_section"] = ""

    # 持仓建议部分
    if context.has_positions():
        template = position_template or context.extra.get("position_template", "")
        if template:
            try:
                sections["position_section"] = f"""
---
**重要：用户持仓操作建议**

{template.format(
    positions_context=context.positions,
    current_price=context.current_price or 0,
    prefs_context=context.preferences or "无特别偏好"
)}
---
"""
            except KeyError as e:
                print(f"⚠️ 模板格式化失败: {e}")
                sections["position_section"] = ""
        else:
            # 无模板时的简单格式
            sections["position_section"] = f"""
---
**用户持仓信息**：
{context.positions}

**用户偏好**：
{context.preferences or "无特别偏好"}
---
"""
    else:
        sections["position_section"] = ""

    return sections


# =============================================================================
# 原有实现（保持兼容）
# =============================================================================


def wrap_model_call(
    category: Optional[str] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    user_id_param: str = "user_id",
):
    """装饰器：在 LLM 调用前注入用户偏好到上下文

    该装饰器会在被装饰函数执行前，从数据库获取用户偏好，
    并将其以 `_user_preferences_context` 参数传入函数。

    Args:
        category: 偏好类别（如 'stock', 'fund'），不指定则获取所有偏好
        checkpoint_manager: CheckpointManager 实例（可选，也可从被装饰方法的 self 获取）
        user_id_param: 用户 ID 参数名，默认为 'user_id'

    Usage:
        class MySkill(BaseSkill):
            def __init__(self, checkpoint_manager):
                self._checkpoint_manager = checkpoint_manager

            @wrap_model_call(category="stock")
            async def execute(self, stock_data_json: str, user_id: str = "default", **kwargs):
                # 从 kwargs 获取注入的偏好上下文
                prefs_context = kwargs.get("_user_preferences_context", "")
                # 将 prefs_context 注入到 LLM prompt 中
                ...

    Notes:
        - 如果未找到 checkpoint_manager，则跳过偏好注入
        - 如果未找到 user_id，使用 "default" 作为默认值
        - 偏好上下文通过 kwargs["_user_preferences_context"] 传递
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 尝试获取 checkpoint_manager
            manager = checkpoint_manager

            # 如果装饰器没有传入 manager，尝试从 self 获取
            if manager is None and args:
                self = args[0]
                # 尝试多种属性名
                for attr_name in ["_checkpoint_manager", "checkpoint_manager", "_cm"]:
                    if hasattr(self, attr_name):
                        manager = getattr(self, attr_name)
                        break

            # 从 kwargs 中提取传入的 checkpoint_manager
            if manager is None:
                manager = kwargs.get("checkpoint_manager")

            # 获取 user_id
            user_id = kwargs.get(user_id_param)
            if user_id is None:
                # 尝试从位置参数获取（如果参数名匹配）
                # 这需要知道函数签名，暂时使用默认值
                user_id = "default"

            # 获取偏好上下文
            prefs_context = ""
            if manager is not None:
                try:
                    prefs_context = await manager.get_preferences_as_context(user_id, category)
                except Exception as e:
                    # 偏好获取失败不应阻止主流程
                    print(f"⚠️ 获取用户偏好失败: {e}")

            # 将偏好上下文注入到 kwargs
            kwargs["_user_preferences_context"] = prefs_context

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def inject_preferences_to_prompt(
    base_prompt: str,
    preferences_context: str,
    position: str = "before",
) -> str:
    """将偏好上下文注入到 LLM prompt

    Args:
        base_prompt: 原始 prompt
        preferences_context: 偏好上下文字符串
        position: 注入位置，'before' 在开头，'after' 在结尾

    Returns:
        注入偏好后的 prompt
    """
    if not preferences_context:
        return base_prompt

    # 添加分隔符使偏好上下文更清晰
    formatted_context = f"""
---
{preferences_context}

Please consider the above user preferences when generating your response.
---

"""

    if position == "before":
        return formatted_context + base_prompt
    else:
        return base_prompt + "\n\n" + formatted_context


class PreferenceAwareModelMixin:
    """偏好感知模型 Mixin

    为需要注入用户偏好的 Skill 提供便捷方法。

    Usage:
        class MySkill(BaseSkill, PreferenceAwareModelMixin):
            async def execute(self, user_id: str, **kwargs):
                prefs = await self.get_user_preferences_context(user_id, "stock")
                prompt = self.build_prompt_with_preferences(base_prompt, prefs)
                ...
    """

    _checkpoint_manager: Optional[CheckpointManager] = None

    def set_checkpoint_manager(self, manager: CheckpointManager):
        """设置 CheckpointManager"""
        self._checkpoint_manager = manager

    async def get_user_preferences_context(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> str:
        """获取用户偏好上下文

        Args:
            user_id: 用户标识符
            category: 偏好类别（可选）

        Returns:
            格式化的偏好上下文字符串
        """
        if self._checkpoint_manager is None:
            return ""

        try:
            return await self._checkpoint_manager.get_preferences_as_context(user_id, category)
        except Exception as e:
            print(f"⚠️ 获取用户偏好失败: {e}")
            return ""

    def build_prompt_with_preferences(
        self,
        base_prompt: str,
        preferences_context: str,
        position: str = "before",
    ) -> str:
        """构建包含用户偏好的 prompt

        Args:
            base_prompt: 原始 prompt
            preferences_context: 偏好上下文
            position: 注入位置

        Returns:
            包含偏好的 prompt
        """
        return inject_preferences_to_prompt(base_prompt, preferences_context, position)
