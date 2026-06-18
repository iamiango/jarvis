"""Prompt 加载工具

提供统一的 prompt 加载和变量替换功能。
所有 system prompt 都存储在 src/prompts/ 目录下的 .md 文件中。
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Prompt 文件目录
PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=32)
def _load_prompt_file(filename: str) -> str:
    """加载 prompt 文件内容（带缓存）

    Args:
        filename: prompt 文件名（不含路径）

    Returns:
        文件内容

    Raises:
        FileNotFoundError: 文件不存在
    """
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(name: str, **variables) -> str:
    """加载并渲染 prompt

    Args:
        name: prompt 名称（不含 .md 后缀）
        **variables: 要替换的变量

    Returns:
        渲染后的 prompt 字符串

    Example:
        >>> prompt = load_prompt("summarization", asset_type="股票", messages="...")
        >>> prompt = load_prompt("jarvis_direct_answer_mcp", max_search_calls=3)
    """
    filename = f"{name}.md"
    content = _load_prompt_file(filename)

    # 变量替换
    if variables:
        try:
            content = content.format(**variables)
        except KeyError as e:
            logger.warning(f"Prompt variable not provided: {e}")
            # 部分替换：只替换提供的变量
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))

    return content


def get_prompt_path(name: str) -> Path:
    """获取 prompt 文件路径

    Args:
        name: prompt 名称（不含 .md 后缀）

    Returns:
        prompt 文件的完整路径
    """
    return PROMPTS_DIR / f"{name}.md"


def list_prompts() -> list[str]:
    """列出所有可用的 prompt

    Returns:
        prompt 名称列表（不含 .md 后缀）
    """
    if not PROMPTS_DIR.exists():
        return []
    return [f.stem for f in PROMPTS_DIR.glob("*.md")]


def reload_prompts() -> None:
    """清除 prompt 缓存，强制重新加载

    用于开发调试时修改 prompt 后立即生效。
    """
    _load_prompt_file.cache_clear()
    logger.info("Prompt cache cleared")


# 预定义的 prompt 名称常量
class PromptNames:
    """Prompt 名称常量"""
    STOCK_AGENT_TOOL_CALLING = "stock_agent_tool_calling"
    FUND_AGENT_TOOL_CALLING = "fund_agent_tool_calling"
    SUMMARIZATION = "summarization"
    LLM_ROUTER = "llm_router"
    JARVIS_DIRECT_ANSWER_MCP = "jarvis_direct_answer_mcp"
    JARVIS_DIRECT_ANSWER = "jarvis_direct_answer"
    TAVILY_TOOL_DESCRIPTION = "tavily_tool_description"
    FUND_AGENT_EXECUTOR = "fund_agent_executor"
