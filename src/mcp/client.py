"""MCP Client 实现"""
import httpx
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import json


class MCPTool(BaseModel):
    """MCP Tool 定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPResource(BaseModel):
    """MCP Resource 定义"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


class MCPClient:
    """MCP Client - 与 MCP Server 通信"""

    def __init__(self, server_url: str, name: str = "mcp_server"):
        self.server_url = server_url.rstrip("/")
        self.name = name
        self._tools: List[MCPTool] = []
        self._resources: List[MCPResource] = []
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def initialize(self) -> bool:
        """初始化连接并获取可用工具和资源"""
        try:
            # 获取工具列表
            await self._fetch_tools()
            # 获取资源列表
            await self._fetch_resources()
            return True
        except Exception as e:
            print(f"MCP Client 初始化失败: {e}")
            return False

    async def _fetch_tools(self) -> None:
        """获取 MCP Server 提供的工具列表"""
        try:
            response = await self._http_client.post(
                f"{self.server_url}",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                }
            )
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "tools" in data["result"]:
                    self._tools = [
                        MCPTool(**tool) for tool in data["result"]["tools"]
                    ]
        except Exception as e:
            print(f"获取 MCP Tools 失败: {e}")

    async def _fetch_resources(self) -> None:
        """获取 MCP Server 提供的资源列表"""
        try:
            response = await self._http_client.post(
                f"{self.server_url}",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "resources/list",
                    "params": {}
                }
            )
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "resources" in data["result"]:
                    self._resources = [
                        MCPResource(**res) for res in data["result"]["resources"]
                    ]
        except Exception as e:
            print(f"获取 MCP Resources 失败: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用 MCP Tool"""
        try:
            response = await self._http_client.post(
                f"{self.server_url}",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
            )
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    return {"success": True, "result": data["result"]}
                elif "error" in data:
                    return {"success": False, "error": data["error"]}
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """读取 MCP Resource"""
        try:
            response = await self._http_client.post(
                f"{self.server_url}",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "resources/read",
                    "params": {"uri": uri}
                }
            )
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    return {"success": True, "result": data["result"]}
                elif "error" in data:
                    return {"success": False, "error": data["error"]}
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_tools(self) -> List[MCPTool]:
        """获取已缓存的工具列表"""
        return self._tools

    def get_resources(self) -> List[MCPResource]:
        """获取已缓存的资源列表"""
        return self._resources

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取工具的 Schema 格式（用于 LLM）"""
        return [
            {
                "name": f"mcp_{self.name}_{tool.name}",
                "description": tool.description,
                "parameters": tool.input_schema
            }
            for tool in self._tools
        ]

    async def close(self) -> None:
        """关闭客户端"""
        await self._http_client.aclose()


class MCPManager:
    """MCP Server 管理器"""

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}

    async def add_server(self, name: str, url: str) -> bool:
        """添加 MCP Server"""
        client = MCPClient(url, name)
        if await client.initialize():
            self._clients[name] = client
            return True
        return False

    async def remove_server(self, name: str) -> None:
        """移除 MCP Server"""
        if name in self._clients:
            await self._clients[name].close()
            del self._clients[name]

    def get_client(self, name: str) -> Optional[MCPClient]:
        """获取指定的 MCP Client"""
        return self._clients.get(name)

    def list_servers(self) -> List[str]:
        """列出所有 MCP Server"""
        return list(self._clients.keys())

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取所有 MCP Server 的工具 Schema"""
        schemas = []
        for client in self._clients.values():
            schemas.extend(client.get_tool_schemas())
        return schemas

    async def call_tool(self, full_tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用 MCP Tool（根据完整工具名解析）"""
        # 解析工具名: mcp_{server_name}_{tool_name}
        parts = full_tool_name.split("_", 2)
        if len(parts) < 3 or parts[0] != "mcp":
            return {"success": False, "error": "Invalid MCP tool name format"}

        server_name = parts[1]
        tool_name = parts[2]

        client = self.get_client(server_name)
        if not client:
            return {"success": False, "error": f"MCP Server '{server_name}' not found"}

        return await client.call_tool(tool_name, arguments)

    async def close_all(self) -> None:
        """关闭所有客户端"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


# 全局 MCP 管理器实例
mcp_manager = MCPManager()
