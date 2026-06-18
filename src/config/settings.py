"""配置文件"""
from pydantic import BaseModel
from typing import Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class OllamaConfig(BaseModel):
    """Ollama 配置"""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    temperature: float = 0.7
    timeout: int = 120


class MCPConfig(BaseModel):
    """MCP Server 配置"""
    servers: dict = {}  # MCP Server 配置字典


class EmailConfig(BaseModel):
    """Email 配置

    支持的邮箱服务商配置示例:

    Gmail:
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_PASSWORD=应用专用密码 (需在 Google 账户设置中生成)

    163邮箱:
        SMTP_HOST=smtp.163.com
        SMTP_PORT=465 (SSL) 或 25 (非SSL)
        SMTP_PASSWORD=授权码 (需在 163 邮箱设置中开启 SMTP 并获取)

    QQ邮箱:
        SMTP_HOST=smtp.qq.com
        SMTP_PORT=465 (SSL) 或 587 (TLS)
        SMTP_PASSWORD=授权码 (需在 QQ 邮箱设置中开启 SMTP 并获取)

    126邮箱:
        SMTP_HOST=smtp.126.com
        SMTP_PORT=465 (SSL) 或 25 (非SSL)

    企业微信邮箱:
        SMTP_HOST=smtp.exmail.qq.com
        SMTP_PORT=465 (SSL)

    阿里企业邮箱:
        SMTP_HOST=smtp.qiye.aliyun.com
        SMTP_PORT=465 (SSL)
    """
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.163.com")  # 默认使用 163
    smtp_port: int = int(os.getenv("SMTP_PORT", "465"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")  # 授权码，非登录密码
    use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    use_ssl: bool = os.getenv("SMTP_USE_SSL", "false").lower() == "true"  # 465端口使用SSL
    default_to: str = os.getenv("SMTP_DEFAULT_TO", "")  # 默认收件人邮箱


class AgentConfig(BaseModel):
    """Agent 配置"""
    max_iterations: int = 10
    verbose: bool = True
    # A2A Agent URLs - 子 Agent 的服务地址
    sub_agents: Dict[str, str] = {
        "stock_agent": os.getenv("STOCK_AGENT_URL", "http://localhost:8001"),
        "email_agent": os.getenv("EMAIL_AGENT_URL", "http://localhost:8002"),
        "fund_agent": os.getenv("FUND_AGENT_URL", "http://localhost:8003"),
    }


class QwenConfig(BaseModel):
    """Qwen 远程 API 配置 (用于翻译)"""
    api_key: str = os.getenv("QWEN_API_KEY", "")
    api_base: str = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    temperature: float = 0.3  # 翻译使用较低温度
    timeout: int = int(os.getenv("QWEN_TIMEOUT", "120"))  # 增加超时时间到 120 秒
    max_retries: int = int(os.getenv("QWEN_MAX_RETRIES", "3"))  # 重试次数


class PostgresConfig(BaseModel):
    """PostgreSQL 数据库配置"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "jarvis")
    user: str = os.getenv("POSTGRES_USER", "jarvis_user")
    password: str = os.getenv("POSTGRES_PASSWORD", "jarvis_pass")

    @property
    def connection_string(self) -> str:
        """返回 PostgreSQL 连接字符串"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# 全局配置实例
ollama_config = OllamaConfig()
mcp_config = MCPConfig()
email_config = EmailConfig()
agent_config = AgentConfig()
qwen_config = QwenConfig()
postgres_config = PostgresConfig()
