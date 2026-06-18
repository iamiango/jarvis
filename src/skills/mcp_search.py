"""MCP Search Skill - 基于 MCP 的网络搜索技能

使用 Tavily MCP Server 进行网络搜索，为 LLM 提供最新信息。
"""
import os
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillOutput
from ..mcp import MultiServerMCPClient


class MCPSearchSkill(BaseSkill):
    """MCP 网络搜索技能

    使用 Tavily MCP Server 搜索网络获取最新信息。
    可用于：
    - 搜索股票/公司最新新闻
    - 获取市场动态
    - 查找行业信息
    """

    name = "mcp_search"
    description = "使用 Tavily MCP 进行网络搜索，获取最新信息和新闻"

    def __init__(self, mcp_client: Optional[MultiServerMCPClient] = None):
        """初始化 MCP Search Skill

        Args:
            mcp_client: 共享的 MCP 客户端实例，如果为 None 则创建新实例
        """
        super().__init__()
        self._mcp_client = mcp_client
        self._own_client = False  # 是否是自己创建的客户端
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        """确保 MCP 客户端已初始化"""
        if self._initialized:
            return True

        if self._mcp_client is None:
            # 创建自己的客户端
            self._mcp_client = MultiServerMCPClient()
            self._own_client = True

        # 检查是否已有 tavily 服务器
        if "tavily" not in self._mcp_client.list_servers():
            # 尝试添加 Tavily
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                print("⚠️ TAVILY_API_KEY 未配置，MCP 搜索不可用")
                return False

            success = await self._mcp_client.add_stdio_server(
                name="tavily",
                command="npx",
                args=["-y", "tavily-mcp"],
                env={"TAVILY_API_KEY": api_key},
                timeout=60,
            )

            if not success:
                print("❌ 无法连接 Tavily MCP Server")
                return False

        self._initialized = True
        return True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数",
                    "default": 5
                },
                "search_depth": {
                    "type": "string",
                    "description": "搜索深度: basic 或 advanced",
                    "enum": ["basic", "advanced"],
                    "default": "basic"
                }
            },
            "required": ["query"]
        }

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        **kwargs
    ) -> SkillOutput:
        """执行网络搜索

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            search_depth: 搜索深度 (basic/advanced)

        Returns:
            SkillOutput 包含搜索结果
        """
        try:
            # 确保初始化
            if not await self._ensure_initialized():
                return SkillOutput(
                    success=False,
                    result=None,
                    error="MCP 搜索服务未就绪，请检查 TAVILY_API_KEY 配置"
                )

            # 调用 Tavily search 工具
            result = await self._mcp_client.call_tool(
                "tavily",
                "search",
                {
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                }
            )

            if result["success"]:
                return SkillOutput(
                    success=True,
                    result=result["result"]
                )
            else:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"搜索失败: {result.get('error', '未知错误')}"
                )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"MCP 搜索执行失败: {str(e)}"
            )

    async def search_stock_news(self, stock_name: str, stock_code: str) -> SkillOutput:
        """搜索股票相关新闻

        Args:
            stock_name: 股票名称
            stock_code: 股票代码

        Returns:
            SkillOutput 包含新闻搜索结果
        """
        query = f"{stock_name} {stock_code} 最新消息 股票新闻"
        return await self.execute(query, max_results=5, search_depth="basic")

    async def search_company_info(self, company_name: str) -> SkillOutput:
        """搜索公司相关信息

        Args:
            company_name: 公司名称

        Returns:
            SkillOutput 包含公司信息
        """
        query = f"{company_name} 公司 最新动态 业务发展"
        return await self.execute(query, max_results=5, search_depth="advanced")

    async def search_market_news(self, topic: str = "A股市场") -> SkillOutput:
        """搜索市场新闻

        Args:
            topic: 搜索主题

        Returns:
            SkillOutput 包含市场新闻
        """
        query = f"{topic} 最新行情 市场分析"
        return await self.execute(query, max_results=5, search_depth="basic")

    def get_tool_schema_for_llm(self) -> Dict[str, Any]:
        """获取 LLM function calling 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "搜索网络获取最新信息、新闻和数据。当需要了解公司最新动态、市场新闻、行业趋势时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询词，建议使用中文"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def close(self) -> None:
        """关闭资源

        注意：MCP stdio_client 使用 anyio 的 cancel scope 管理生命周期。
        如果在不同的 asyncio Task 中关闭客户端，可能会触发
        "Attempted to exit cancel scope in a different task than it was entered in" 错误。
        这种情况下我们捕获并忽略该错误，因为进程结束时资源会自动释放。
        """
        if self._own_client and self._mcp_client:
            try:
                await self._mcp_client.close_all()
            except RuntimeError as e:
                # 忽略 cancel scope 跨任务错误
                if "cancel scope" in str(e).lower():
                    print(f"⚠️ MCP 客户端关闭时忽略 cancel scope 错误 (预期行为)")
                else:
                    raise
            except Exception as e:
                print(f"⚠️ MCP 客户端关闭失败: {e}")
            finally:
                self._mcp_client = None
                self._initialized = False


class MCPSearchManager:
    """MCP 搜索管理器 - 单例模式管理共享的 MCP 客户端"""

    _instance: Optional["MCPSearchManager"] = None
    _mcp_client: Optional[MultiServerMCPClient] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_client(self) -> Optional[MultiServerMCPClient]:
        """获取共享的 MCP 客户端"""
        if not self._initialized:
            await self._initialize()
        return self._mcp_client

    async def _initialize(self) -> bool:
        """初始化 MCP 客户端"""
        if self._initialized:
            return True

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("⚠️ TAVILY_API_KEY 未配置，MCP 搜索功能不可用")
            return False

        self._mcp_client = MultiServerMCPClient()

        try:
            success = await self._mcp_client.add_stdio_server(
                name="tavily",
                command="npx",
                args=["-y", "tavily-mcp"],
                env={"TAVILY_API_KEY": api_key},
                timeout=60,
            )

            if success:
                self._initialized = True
                print("✅ MCP 搜索服务已初始化 (Tavily)")
                return True
            else:
                print("❌ MCP 搜索服务初始化失败")
                return False

        except Exception as e:
            print(f"❌ MCP 搜索服务初始化异常: {e}")
            return False

    def create_search_skill(self) -> MCPSearchSkill:
        """创建共享客户端的搜索技能"""
        return MCPSearchSkill(mcp_client=self._mcp_client)

    async def close(self) -> None:
        """关闭资源"""
        if self._mcp_client:
            await self._mcp_client.close_all()
            self._mcp_client = None
        self._initialized = False


# 全局单例
mcp_search_manager = MCPSearchManager()
