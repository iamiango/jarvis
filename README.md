# Jarvis - Multi-Agent Orchestration System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-0.3+-green.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-1.2+-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

Jarvis 是一个基于 **LangChain** 和 **LangGraph** 框架构建的多智能体调度系统。它通过 **A2A (Agent-to-Agent) 协议** 协调多个专业化 AI Agent（股票分析、邮件发送、基金分析），使用规则引擎进行意图路由，支持本地 Ollama 模型部署。

## ✨ 核心特性

- **多智能体协作**: Jarvis 作为调度器，协调 Stock、Email、Fund 等专业 Agent
- **A2A 协议**: 标准化的 Agent 间通信协议，支持任务委派和状态追踪
- **规则引擎路由**: 三层优先级路由（固定流程 → 动态拆分 → 单一意图 → LLM 兜底）
- **技术指标分析**: 集成 AKShare，支持 MACD、KDJ、RSI 等技术指标计算
- **MCP 协议支持**: 集成 Tavily 搜索等 MCP Server，获取实时信息
- **流式输出**: 支持 SSE 流式响应，实时展示执行进度和 LLM 输出
- **定时任务**: 内置调度器，支持盘中股票监控、定时报告等
- **安全机制**: PII 脱敏、Human-in-the-Loop 邮件确认
- **状态持久化**: PostgreSQL checkpoint 系统，支持会话恢复

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   CLI/REPL  │  │  Chat UI    │  │  LangGraph Studio   │ │
│  │  (main.py)  │  │  (port 3000)│  │    (port 8123)      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼───────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (port 2024)                  │
│              LangGraph API Adapter (FastAPI)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Jarvis Orchestrator (port 8000)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Rule Parser  │  │  LLM Router  │  │  Task Delegator  │  │
│  │ (3-layer)    │  │  (fallback)  │  │  (A2A Client)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ A2A Protocol
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│  StockAgent     │ │ EmailAgent  │ │   FundAgent     │
│  (port 8001)    │ │ (port 8002) │ │   (port 8003)   │
│                 │ │             │ │                 │
│ • 股票数据获取  │ │ • SMTP发送  │ │ • 基金数据获取  │
│ • 技术指标分析  │ │ • 模板渲染  │ │ • 净值分析      │
│ • 报告生成      │ │ • PII脱敏   │ │ • 报告生成      │
│ • 持仓管理      │ │             │ │ • 持仓管理      │
└─────────────────┘ └─────────────┘ └─────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      External Services                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ AKShare  │  │  Ollama  │  │  Tavily  │  │ PostgreSQL │  │
│  │ (股票)   │  │  (LLM)   │  │  (搜索)  │  │ (持久化)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 依赖要求

### 系统要求
- Python 3.10+
- PostgreSQL 14+ (可选，用于状态持久化)
- Node.js 18+ (可选，用于 Agent Chat UI)

### 核心框架
| 依赖 | 版本 | 说明 |
|------|------|------|
| langgraph | ≥1.2.0 | 状态机编排框架 |
| langchain | ≥1.3.0 | LLM 应用框架 |
| langchain-ollama | ≥1.1.0 | Ollama 集成 |
| fastapi | ≥0.100.0 | A2A 服务端 |
| pydantic | ≥2.0.0 | 数据验证 |

### 业务功能
| 依赖 | 版本 | 说明 |
|------|------|------|
| akshare | ≥1.18.0 | A股数据获取 |
| pandas | ≥2.0.0 | 数据处理 |
| aiosmtplib | ≥2.0.0 | 异步邮件发送 |
| apscheduler | ≥3.10.0 | 定时任务 |
| mcp | ≥1.0.0 | MCP 协议客户端 |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/iamiango/jarvis.git
cd jarvis
```

### 2. 创建虚拟环境

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装并启动 Ollama

```bash
# macOS
brew install ollama

# 启动服务
ollama serve

# 下载模型
ollama pull qwen3:8b
```

### 5. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# LLM 配置
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

# 邮件配置 (可选)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your-email@example.com
SMTP_PASSWORD=your-password
SMTP_DEFAULT_TO=recipient@example.com

# PostgreSQL (可选)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=jarvis
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# MCP 搜索 (可选)
TAVILY_API_KEY=your-tavily-api-key

# 股票监控 (可选)
STOCK_MONITOR_ENABLED=true
STOCK_CODES=600588,002230
```

### 6. 启动服务

**方式一：一键启动所有 Agent**

```bash
./scripts/start_all.sh
```

**方式二：单独启动各 Agent**

```bash
# 终端 1 - Stock Agent
python -m src.agents.stock_agent --port 8001

# 终端 2 - Email Agent
python -m src.agents.email_agent --port 8002

# 终端 3 - Fund Agent
python -m src.agents.fund_agent --port 8003

# 终端 4 - Jarvis Orchestrator
python -m src.agents.jarvis --port 8000
```

### 7. 使用 CLI 交互

```bash
python main.py
```

示例对话：
```
You: 分析股票600588
Jarvis: 正在分析用友网络(600588)...

You: 分析600588并发送邮件到test@example.com
Jarvis: 需要发送邮件，请确认 [Y/n]: Y
```

## 📖 使用指南

### CLI 命令

```bash
# 交互式 CLI
python main.py

# 股票专用 CLI
python stock_cli.py

# 运行测试
python test_framework.py
python test_stock_skill.py
```

### 定时任务

```bash
# 启动调度器
python -m src.scheduler.run

# 查看所有任务
python -m src.scheduler.run --list

# 立即执行任务
python -m src.scheduler.run --run-now daily_fund_report
python -m src.scheduler.run --run-now hourly_stock_monitor
```

### Agent Chat UI (Web 界面)

```bash
# 1. 启动 LangGraph API
./scripts/start_langgraph_api.sh

# 2. 安装并启动 Chat UI
./scripts/setup_agent_chat_ui.sh
cd agent-chat-ui && pnpm dev

# 3. 访问
open http://localhost:3000
```

### LangGraph Studio (可视化调试)

```bash
# 启动 Studio
./scripts/start_langgraph_studio.sh

# 访问
open "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123"
```

## 🔧 A2A 协议

每个 Agent 暴露标准的 A2A 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/.well-known/agent.json` | GET | Agent 能力卡片 |
| `/tasks/send` | POST | 提交任务 |
| `/tasks/{id}` | GET | 获取任务状态 |
| `/tasks/cancel` | POST | 取消任务 |

### 示例：调用 Stock Agent

```bash
# 获取 Agent 信息
curl http://localhost:8001/.well-known/agent.json

# 提交分析任务
curl -X POST http://localhost:8001/tasks/send \
  -H "Content-Type: application/json" \
  -d '{
    "id": "task-001",
    "history": [{"role": "user", "parts": [{"text": "分析股票600588"}]}]
  }'
```

## 🧩 扩展开发

### 添加新 Agent

```python
# src/agents/my_agent.py
from src.agents.base import BaseAgent
from src.a2a.types import Task, AgentSkill

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "我的自定义 Agent"

    def get_skills(self) -> list[AgentSkill]:
        return [AgentSkill(id="skill1", name="技能1", ...)]

    async def process_task(self, task: Task) -> Task:
        # 处理任务逻辑
        result = await self._execute_skill(task)
        return self._create_text_response(task, result)
```

### 添加新 Skill

```python
# src/skills/my_skill.py
from src.skills.base import BaseSkill, SkillOutput

class MySkill(BaseSkill):
    name = "my_skill"
    description = "我的自定义 Skill"

    async def execute(self, param1: str, **kwargs) -> SkillOutput:
        # 执行逻辑
        return SkillOutput(success=True, result="执行结果")

    def get_parameters(self) -> dict:
        return {
            "properties": {"param1": {"type": "string", "description": "参数1"}},
            "required": ["param1"]
        }
```

### 使用 MCP 搜索

```python
from src.skills.mcp_search import MCPSearchSkill

search = MCPSearchSkill()
result = await search.execute("最新 AI 新闻", max_results=5)
print(result.result)
```

## 📁 项目结构

```
jarvis/
├── src/
│   ├── agents/           # Agent 实现
│   │   ├── base.py       # BaseAgent 抽象类
│   │   ├── jarvis.py     # 调度器 Agent
│   │   ├── stock_agent.py
│   │   ├── email_agent.py
│   │   └── fund_agent.py
│   ├── skills/           # Skill 系统
│   │   ├── base.py       # BaseSkill 抽象类
│   │   ├── stock_*.py    # 股票相关 Skills
│   │   ├── fund_*.py     # 基金相关 Skills
│   │   └── mcp_search.py # MCP 搜索 Skill
│   ├── a2a/              # A2A 协议实现
│   │   ├── server.py     # FastAPI 服务端
│   │   ├── client.py     # HTTP 客户端
│   │   └── types.py      # 数据类型定义
│   ├── router/           # 路由模块
│   │   ├── parser.py     # 规则解析器
│   │   └── llm_router.py # LLM 路由兜底
│   ├── mcp/              # MCP 客户端
│   ├── scheduler/        # 定时任务
│   ├── security/         # 安全模块
│   ├── storage/          # 持久化
│   ├── streaming/        # 流式输出
│   ├── api/              # LangGraph API 适配
│   └── prompts/          # Prompt 模板
├── config/               # 配置文件
├── scripts/              # 启动脚本
├── main.py               # CLI 入口
└── requirements.txt      # Python 依赖
```

## 🔐 安全特性

- **PII 脱敏**: 自动检测并脱敏邮箱、手机号、身份证等敏感信息
- **Human-in-the-Loop**: 邮件发送前需用户确认
- **环境变量**: 敏感配置通过环境变量管理，不进入代码库

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

<p align="center">
  Built with ❤️ using LangChain & LangGraph
</p>
