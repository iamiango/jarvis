"""Agent Registry - Agent 注册表

支持 Agent 发现、注册和管理。
"""
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from ..a2a import AgentCard, AgentInfo, A2AClient, A2AClientError


class AgentRegistry:
    """Agent 注册表 - 支持 A2A 发现和管理"""

    def __init__(self):
        """初始化注册表"""
        self._agents: Dict[str, AgentInfo] = {}
        self._client = A2AClient()

    async def discover(self, url: str) -> AgentCard:
        """发现 Agent - 通过 URL 获取 Agent Card

        Args:
            url: Agent 服务的 URL

        Returns:
            AgentCard 对象

        Raises:
            A2AClientError: 发现失败时抛出
        """
        card = await self._client.discover(url)
        # 自动注册发现的 Agent
        await self.register(card, url)
        return card

    async def register(self, agent_card: AgentCard, url: str) -> None:
        """注册 Agent

        Args:
            agent_card: Agent Card
            url: Agent 服务的 URL
        """
        agent_info = AgentInfo(
            card=agent_card,
            url=url,
            last_seen=datetime.now(),
            healthy=True,
        )
        self._agents[agent_card.name] = agent_info

    def register_local(self, agent_card: AgentCard, url: str) -> None:
        """同步注册本地 Agent（无需网络请求）

        Args:
            agent_card: Agent Card
            url: Agent 服务的 URL
        """
        agent_info = AgentInfo(
            card=agent_card,
            url=url,
            last_seen=datetime.now(),
            healthy=True,
        )
        self._agents[agent_card.name] = agent_info

    def unregister(self, name: str) -> bool:
        """注销 Agent

        Args:
            name: Agent 名称

        Returns:
            True 如果成功注销，否则 False
        """
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """获取 Agent 信息

        Args:
            name: Agent 名称

        Returns:
            AgentInfo 对象，如果不存在则返回 None
        """
        return self._agents.get(name)

    def get_agent_url(self, name: str) -> Optional[str]:
        """获取 Agent URL

        Args:
            name: Agent 名称

        Returns:
            Agent URL，如果不存在则返回 None
        """
        agent = self._agents.get(name)
        return agent.url if agent else None

    def list_agents(self) -> List[AgentInfo]:
        """列出所有已注册的 Agent

        Returns:
            AgentInfo 列表
        """
        return list(self._agents.values())

    def list_agent_names(self) -> List[str]:
        """列出所有 Agent 名称

        Returns:
            Agent 名称列表
        """
        return list(self._agents.keys())

    def find_agent_by_skill(self, skill_id: str) -> Optional[AgentInfo]:
        """通过技能 ID 查找 Agent

        Args:
            skill_id: 技能 ID

        Returns:
            AgentInfo 对象，如果未找到则返回 None
        """
        for agent_info in self._agents.values():
            for skill in agent_info.card.skills:
                if skill.id == skill_id:
                    return agent_info
        return None

    def find_agents_by_tag(self, tag: str) -> List[AgentInfo]:
        """通过标签查找 Agent

        Args:
            tag: 技能标签

        Returns:
            匹配的 AgentInfo 列表
        """
        result = []
        for agent_info in self._agents.values():
            for skill in agent_info.card.skills:
                if tag in skill.tags:
                    result.append(agent_info)
                    break
        return result

    async def health_check(self, name: str) -> bool:
        """检查 Agent 健康状态

        Args:
            name: Agent 名称

        Returns:
            True 如果 Agent 健康，否则 False
        """
        agent = self._agents.get(name)
        if not agent:
            return False

        healthy = await self._client.health_check(agent.url)
        agent.healthy = healthy
        agent.last_seen = datetime.now()
        return healthy

    async def health_check_all(self) -> Dict[str, bool]:
        """检查所有 Agent 的健康状态

        Returns:
            Agent 名称到健康状态的映射
        """
        results = {}
        for name in self._agents:
            results[name] = await self.health_check(name)
        return results

    async def discover_all(self, urls: List[str]) -> List[AgentCard]:
        """批量发现 Agent

        Args:
            urls: Agent URL 列表

        Returns:
            成功发现的 AgentCard 列表
        """
        cards = []
        for url in urls:
            try:
                card = await self.discover(url)
                cards.append(card)
            except A2AClientError as e:
                print(f"发现 Agent 失败 ({url}): {e}")
                continue
        return cards

    async def close(self) -> None:
        """关闭客户端连接"""
        await self._client.close()

    def __len__(self) -> int:
        """返回已注册 Agent 数量"""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """检查 Agent 是否已注册"""
        return name in self._agents


# 全局注册表实例
agent_registry = AgentRegistry()
