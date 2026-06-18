"""MultiServerMCPClient - 多 MCP Server 客户端管理器

支持多种传输协议:
- stdio: 本地进程通信 (如 tavily-mcp, filesystem-mcp)
- sse: Server-Sent Events (HTTP SSE)
- http: 标准 HTTP JSON-RPC

使用示例:
    async with MultiServerMCPClient() as client:
        # 从配置文件加载
        await client.load_from_config()

        # 或手动添加服务器
        await client.add_stdio_server(
            "tavily",
            command="npx",
            args=["-y", "tavily-mcp"],
            env={"TAVILY_API_KEY": "your-key"}
        )

        # 调用工具
        result = await client.call_tool("tavily", "search", {"query": "AI news"})
"""
import asyncio
import os
import json
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from pydantic import BaseModel


class TransportType(str, Enum):
    """MCP 传输协议类型"""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPServerConfig(BaseModel):
    """MCP Server 配置"""
    name: str
    transport: TransportType = TransportType.STDIO
    # stdio 配置
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}
    # sse/http 配置
    url: Optional[str] = None
    # 通用配置
    enabled: bool = True
    timeout: int = 30


@dataclass
class MCPServerConnection:
    """MCP Server 连接状态"""
    config: MCPServerConfig
    session: Optional[ClientSession] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    _cleanup: Optional[Any] = None  # 清理函数


class MultiServerMCPClient:
    """多 MCP Server 客户端管理器

    管理多个 MCP Server 的连接，支持:
    - 统一的工具调用接口
    - 自动重连
    - 工具发现

    使用 AsyncExitStack 正确管理 async context managers
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerConnection] = {}
        self._initialized = False
        self._exit_stack: Optional[AsyncExitStack] = None

    async def __aenter__(self) -> "MultiServerMCPClient":
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_all()
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None

    # ==================== Server Management ====================

    async def add_stdio_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> bool:
        """添加 stdio 传输的 MCP Server

        Args:
            name: 服务器名称
            command: 启动命令 (如 "npx", "python", "uvx")
            args: 命令参数
            env: 环境变量
            timeout: 超时时间(秒)

        Returns:
            是否添加成功
        """
        config = MCPServerConfig(
            name=name,
            transport=TransportType.STDIO,
            command=command,
            args=args or [],
            env=env or {},
            timeout=timeout,
        )
        return await self._connect_server(config)

    async def add_sse_server(
        self,
        name: str,
        url: str,
        timeout: int = 30,
    ) -> bool:
        """添加 SSE 传输的 MCP Server

        Args:
            name: 服务器名称
            url: SSE 服务器 URL
            timeout: 超时时间(秒)

        Returns:
            是否添加成功
        """
        config = MCPServerConfig(
            name=name,
            transport=TransportType.SSE,
            url=url,
            timeout=timeout,
        )
        return await self._connect_server(config)

    async def _connect_server(self, config: MCPServerConfig) -> bool:
        """连接到 MCP Server"""
        if config.name in self._servers:
            print(f"⚠️ MCP Server '{config.name}' 已存在，将重新连接")
            await self.remove_server(config.name)

        connection = MCPServerConnection(config=config)

        try:
            if config.transport == TransportType.STDIO:
                success = await self._connect_stdio(connection)
            elif config.transport == TransportType.SSE:
                success = await self._connect_sse(connection)
            else:
                print(f"❌ 不支持的传输类型: {config.transport}")
                return False

            if success:
                self._servers[config.name] = connection
                print(f"✅ MCP Server '{config.name}' 连接成功")
                print(f"   - 工具数量: {len(connection.tools)}")
                if connection.tools:
                    for tool in connection.tools[:5]:  # 只显示前5个
                        print(f"     • {tool.get('name', 'unknown')}")
                    if len(connection.tools) > 5:
                        print(f"     ... 还有 {len(connection.tools) - 5} 个工具")
                return True
            else:
                return False

        except Exception as e:
            print(f"❌ 连接 MCP Server '{config.name}' 失败: {e}")
            return False

    async def _connect_stdio(self, connection: MCPServerConnection) -> bool:
        """连接 stdio 传输的服务器

        使用 AsyncExitStack 正确管理 async context manager 生命周期
        """
        config = connection.config

        if not config.command:
            print(f"❌ stdio 传输需要指定 command")
            return False

        # 确保 exit_stack 存在
        if self._exit_stack is None:
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

        # 合并环境变量
        env = {**os.environ, **config.env}

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env,
        )

        try:
            # 使用 exit_stack 管理 stdio_client 上下文
            read, write = await asyncio.wait_for(
                self._exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=config.timeout
            )

            # 使用 exit_stack 管理 ClientSession 上下文
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # 初始化
            await session.initialize()

            # 获取工具列表
            tools_result = await session.list_tools()
            connection.tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                }
                for tool in tools_result.tools
            ]

            # 获取资源列表 (可选)
            try:
                resources_result = await session.list_resources()
                connection.resources = [
                    {
                        "uri": res.uri,
                        "name": res.name,
                        "description": res.description or "",
                    }
                    for res in resources_result.resources
                ]
            except Exception:
                connection.resources = []

            connection.session = session
            connection.connected = True

            return True

        except asyncio.TimeoutError:
            print(f"❌ MCP Server '{config.name}' 连接超时 ({config.timeout}秒)")
            print(f"   提示: 首次运行需要下载依赖，请稍后重试")
            return False
        except Exception as e:
            error_msg = str(e)
            print(f"❌ MCP Server '{config.name}' 连接失败: {error_msg}")
            if "Connection closed" in error_msg:
                print(f"   可能原因: API Key 无效或 MCP Server 启动失败")
                print(f"   请检查 TAVILY_API_KEY 是否正确配置")
            return False

    async def _connect_sse(self, connection: MCPServerConnection) -> bool:
        """连接 SSE 传输的服务器"""
        config = connection.config

        if not config.url:
            print(f"❌ SSE 传输需要指定 url")
            return False

        # 确保 exit_stack 存在
        if self._exit_stack is None:
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

        try:
            # 使用 exit_stack 管理 sse_client 上下文
            read, write = await asyncio.wait_for(
                self._exit_stack.enter_async_context(sse_client(config.url)),
                timeout=config.timeout
            )

            # 使用 exit_stack 管理 ClientSession 上下文
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # 初始化
            await session.initialize()

            # 获取工具列表
            tools_result = await session.list_tools()
            connection.tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                }
                for tool in tools_result.tools
            ]

            connection.session = session
            connection.connected = True

            return True

        except asyncio.TimeoutError:
            print(f"❌ MCP Server '{config.name}' 连接超时 ({config.timeout}秒)")
            return False
        except Exception as e:
            print(f"❌ MCP Server '{config.name}' 连接失败: {e}")
            return False

    async def remove_server(self, name: str) -> bool:
        """移除 MCP Server

        注意：由于使用 AsyncExitStack 管理上下文，
        单独移除服务器不会关闭其连接，需要调用 close_all()
        """
        if name not in self._servers:
            return False

        connection = self._servers[name]
        connection.connected = False
        connection.session = None

        del self._servers[name]
        print(f"🔌 MCP Server '{name}' 已从列表移除")
        return True

    async def close_all(self) -> None:
        """关闭所有 MCP Server 连接

        通过关闭 AsyncExitStack 来正确清理所有上下文
        """
        # 标记所有服务器为断开
        for name, connection in self._servers.items():
            connection.connected = False
            connection.session = None

        self._servers.clear()

        # AsyncExitStack 会在 __aexit__ 中自动清理

    # ==================== Tool Operations ====================

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """调用 MCP Tool

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            {"success": True/False, "result": ..., "error": ...}
        """
        if server_name not in self._servers:
            return {"success": False, "error": f"MCP Server '{server_name}' 不存在"}

        connection = self._servers[server_name]

        if not connection.connected or not connection.session:
            return {"success": False, "error": f"MCP Server '{server_name}' 未连接"}

        try:
            result = await connection.session.call_tool(tool_name, arguments)

            # 解析结果
            if result.content:
                # 提取文本内容
                text_parts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_parts.append(content.text)
                    elif hasattr(content, 'data'):
                        text_parts.append(str(content.data))

                return {
                    "success": True,
                    "result": "\n".join(text_parts) if text_parts else str(result.content),
                }

            return {"success": True, "result": str(result)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_tools(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取工具列表

        Args:
            server_name: 指定服务器名称，None 表示所有服务器

        Returns:
            工具列表，每个工具包含 server_name 字段
        """
        tools = []

        servers = [server_name] if server_name else self._servers.keys()

        for name in servers:
            if name not in self._servers:
                continue
            connection = self._servers[name]
            for tool in connection.tools:
                tools.append({
                    **tool,
                    "server_name": name,
                    "full_name": f"mcp_{name}_{tool['name']}",
                })

        return tools

    def get_tool_schemas_for_llm(self) -> List[Dict[str, Any]]:
        """获取 LLM 可用的工具 Schema 格式

        Returns:
            OpenAI function calling 格式的工具定义
        """
        schemas = []

        for name, connection in self._servers.items():
            for tool in connection.tools:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": f"mcp_{name}_{tool['name']}",
                        "description": f"[MCP:{name}] {tool.get('description', '')}",
                        "parameters": tool.get('inputSchema', {"type": "object", "properties": {}}),
                    }
                })

        return schemas

    async def call_tool_by_full_name(
        self,
        full_tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """通过完整工具名调用

        Args:
            full_tool_name: 格式为 mcp_{server_name}_{tool_name}
            arguments: 工具参数

        Returns:
            {"success": True/False, "result": ..., "error": ...}
        """
        # 解析: mcp_{server_name}_{tool_name}
        if not full_tool_name.startswith("mcp_"):
            return {"success": False, "error": "工具名必须以 'mcp_' 开头"}

        parts = full_tool_name[4:].split("_", 1)  # 去掉 "mcp_" 前缀
        if len(parts) < 2:
            return {"success": False, "error": "无效的工具名格式"}

        server_name = parts[0]
        tool_name = parts[1]

        return await self.call_tool(server_name, tool_name, arguments)

    # ==================== Resource Operations ====================

    async def read_resource(
        self,
        server_name: str,
        uri: str,
    ) -> Dict[str, Any]:
        """读取 MCP Resource

        Args:
            server_name: MCP Server 名称
            uri: 资源 URI

        Returns:
            {"success": True/False, "result": ..., "error": ...}
        """
        if server_name not in self._servers:
            return {"success": False, "error": f"MCP Server '{server_name}' 不存在"}

        connection = self._servers[server_name]

        if not connection.connected or not connection.session:
            return {"success": False, "error": f"MCP Server '{server_name}' 未连接"}

        try:
            result = await connection.session.read_resource(uri)

            if result.contents:
                text_parts = []
                for content in result.contents:
                    if hasattr(content, 'text'):
                        text_parts.append(content.text)

                return {
                    "success": True,
                    "result": "\n".join(text_parts) if text_parts else str(result.contents),
                }

            return {"success": True, "result": str(result)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_resources(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取资源列表"""
        resources = []

        servers = [server_name] if server_name else self._servers.keys()

        for name in servers:
            if name not in self._servers:
                continue
            connection = self._servers[name]
            for resource in connection.resources:
                resources.append({
                    **resource,
                    "server_name": name,
                })

        return resources

    # ==================== Configuration ====================

    async def load_from_config(self, config: Optional[Dict[str, Any]] = None) -> int:
        """从配置加载 MCP Servers

        配置格式:
        {
            "mcpServers": {
                "tavily": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "tavily-mcp"],
                    "env": {"TAVILY_API_KEY": "..."}
                },
                "filesystem": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/dir"]
                }
            }
        }

        Args:
            config: 配置字典，None 则从环境变量 MCP_CONFIG_PATH 读取

        Returns:
            成功连接的服务器数量
        """
        if config is None:
            config_path = os.getenv("MCP_CONFIG_PATH")
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                print("⚠️ 未找到 MCP 配置文件")
                return 0

        mcp_servers = config.get("mcpServers", {})
        success_count = 0

        for name, server_config in mcp_servers.items():
            if not server_config.get("enabled", True):
                print(f"⏭️ MCP Server '{name}' 已禁用，跳过")
                continue

            transport = server_config.get("transport", "stdio")

            if transport == "stdio":
                success = await self.add_stdio_server(
                    name=name,
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    timeout=server_config.get("timeout", 30),
                )
            elif transport == "sse":
                success = await self.add_sse_server(
                    name=name,
                    url=server_config.get("url", ""),
                    timeout=server_config.get("timeout", 30),
                )
            else:
                print(f"⚠️ 不支持的传输类型: {transport}")
                continue

            if success:
                success_count += 1

        return success_count

    # ==================== Status ====================

    def list_servers(self) -> List[str]:
        """列出所有 MCP Server 名称"""
        return list(self._servers.keys())

    def get_server_status(self, name: str) -> Optional[Dict[str, Any]]:
        """获取服务器状态"""
        if name not in self._servers:
            return None

        connection = self._servers[name]
        return {
            "name": name,
            "transport": connection.config.transport.value,
            "connected": connection.connected,
            "tools_count": len(connection.tools),
            "resources_count": len(connection.resources),
            "tools": [t["name"] for t in connection.tools],
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务器状态"""
        return {
            name: self.get_server_status(name)
            for name in self._servers
        }


# ==================== 便捷函数 ====================

async def create_tavily_client(api_key: Optional[str] = None) -> MultiServerMCPClient:
    """创建包含 Tavily MCP Server 的客户端

    Args:
        api_key: Tavily API Key，None 则从环境变量 TAVILY_API_KEY 读取

    Returns:
        配置好的 MultiServerMCPClient
    """
    api_key = api_key or os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("需要提供 TAVILY_API_KEY")

    client = MultiServerMCPClient()

    await client.add_stdio_server(
        name="tavily",
        command="npx",
        args=["-y", "tavily-mcp"],
        env={"TAVILY_API_KEY": api_key},
    )

    return client


# 全局 MultiServerMCPClient 实例
multi_mcp_client = MultiServerMCPClient()
