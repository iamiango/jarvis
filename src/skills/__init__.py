from .base import BaseSkill, SkillInput, SkillOutput, SkillManager, skill_manager
from .builtin import CalculatorSkill, SearchSkill, WeatherSkill
from .stock_retriever import StockRetrieverSkill
from .stock_report_generator import StockReportGeneratorSkill
from .email import EmailSkill, EmailTemplateSkill
from .fund_retriever import FundRetrieverSkill
from .fund_report import FundReportGeneratorSkill
from .user_preference import (
    SaveUserPreferenceSkill,
    GetUserPreferencesSkill,
    DeleteUserPreferenceSkill,
)
from .mcp_search import MCPSearchSkill, MCPSearchManager, mcp_search_manager

# 保留旧的导入以保持向后兼容（已废弃）
try:
    from .stock_analysis import StockAnalysisSkill
except ImportError:
    StockAnalysisSkill = None

__all__ = [
    "BaseSkill",
    "SkillInput",
    "SkillOutput",
    "SkillManager",
    "skill_manager",
    "CalculatorSkill",
    "SearchSkill",
    "WeatherSkill",
    "StockRetrieverSkill",
    "StockReportGeneratorSkill",
    "EmailSkill",
    "EmailTemplateSkill",
    "FundRetrieverSkill",
    "FundReportGeneratorSkill",
    "StockAnalysisSkill",  # 已废弃，保留兼容性
    # User Preference Skills
    "SaveUserPreferenceSkill",
    "GetUserPreferencesSkill",
    "DeleteUserPreferenceSkill",
    # MCP Search Skills
    "MCPSearchSkill",
    "MCPSearchManager",
    "mcp_search_manager",
]
