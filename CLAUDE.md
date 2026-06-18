# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jarvis is a **multi-agent orchestration system** built on **LangChain** and **LangGraph** frameworks with FastAPI. It coordinates specialized AI agents (Stock, Email, Fund) using the **A2A (Agent-to-Agent) protocol** for inter-agent communication. The system uses a **rule-based router** (v2.0) for intent analysis with LLM fallback for direct answers.

## Development Guidelines

**All development must follow LangChain and LangGraph standard patterns:**

- **Agent Development**: Use `create_tool_calling_agent` + `AgentExecutor` pattern
- **Tool Registration**: Use `StructuredTool` / `Tool` from `langchain_core.tools`
- **Complex Workflows**: Use LangGraph's `StateGraph` for state machine orchestration
- **LLM Integration**: Use `ChatOllama` (local) or `ChatOpenAI` (remote) from LangChain
- **Avoid Custom Implementations**: Prefer framework-provided abstractions over custom solutions

Example - LangChain Agent with Tool Calling:
```python
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Create LLM
llm = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")

# 2. Convert skills to LangChain tools
tool = StructuredTool.from_function(coroutine=skill_func, name="tool_name", description="...")

# 3. Create agent with tool calling
prompt = ChatPromptTemplate.from_messages([
    ("system", "System prompt here"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. Create executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = await executor.ainvoke({"input": user_message})
```

## Prompt Management (System Prompts)

**所有 System Prompt 必须外部化到 `src/prompts/` 目录的 `.md` 文件中，禁止在代码中硬编码。**

### 规则

1. **存储位置**: 所有 prompt 存放在 `src/prompts/*.md`
2. **加载方式**: 使用 `src/prompt_loader.py` 中的 `load_prompt()` 函数
3. **变量替换**: 使用 `{variable_name}` 占位符，通过 `load_prompt(name, var=value)` 替换
4. **命名规范**: 在 `PromptNames` 类中定义常量，避免硬编码字符串

### 用法示例

```python
from ..prompt_loader import load_prompt, PromptNames

# 加载无变量的 prompt
system_prompt = load_prompt(PromptNames.STOCK_AGENT_TOOL_CALLING)

# 加载带变量的 prompt
summarization_prompt = load_prompt(
    PromptNames.SUMMARIZATION,
    asset_type="股票",
    messages=messages_text
)
```

### 现有 Prompts

| Prompt 名称 | 文件 | 变量 |
|------------|------|------|
| `STOCK_AGENT_TOOL_CALLING` | `stock_agent_tool_calling.md` | - |
| `FUND_AGENT_TOOL_CALLING` | `fund_agent_tool_calling.md` | - |
| `SUMMARIZATION` | `summarization.md` | `{asset_type}`, `{messages}` |
| `LLM_ROUTER` | `llm_router.md` | `{available_agents}` |
| `JARVIS_DIRECT_ANSWER_MCP` | `jarvis_direct_answer_mcp.md` | `{max_search_calls}` |
| `JARVIS_DIRECT_ANSWER` | `jarvis_direct_answer.md` | - |
| `TAVILY_TOOL_DESCRIPTION` | `tavily_tool_description.md` | `{max_calls}` |
| `FUND_AGENT_EXECUTOR` | `fund_agent_executor.md` | - |

### 添加新 Prompt

1. 在 `src/prompts/` 目录创建 `my_prompt.md` 文件
2. 在 `src/prompt_loader.py` 的 `PromptNames` 类中添加常量:
   ```python
   MY_PROMPT = "my_prompt"
   ```
3. 在代码中使用:
   ```python
   prompt = load_prompt(PromptNames.MY_PROMPT, var1="value1")
   ```

## Development Commands

```bash
# Setup
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3.5:9b

# Start all agents (recommended)
./scripts/start_all.sh

# Start individual agents
python -m src.agents.stock_agent --port 8001
python -m src.agents.email_agent --port 8002
python -m src.agents.fund_agent --port 8003
python -m src.agents.jarvis --port 8000

# Interactive CLI
python main.py
python stock_cli.py

# Run tests
python test_framework.py
python test_stock_skill.py
python scripts/test_router.py  # Test rule-based routing

# Scheduler (定时任务)
python -m src.scheduler.run                          # Start scheduler
python -m src.scheduler.run --list                   # List all jobs
python -m src.scheduler.run --run-now daily_fund_report     # Run fund report now
python -m src.scheduler.run --run-now hourly_stock_monitor  # Run stock monitor now

# Cleanup
./scripts/stop_all.sh
```

## Architecture

### Multi-Agent System
```
Jarvis (Orchestrator) - port 8000
    ├── StockAgent - port 8001 (stock analysis with MACD/KDJ/RSI)
    ├── EmailAgent - port 8002 (SMTP email sending)
    └── FundAgent - port 8003 (fund analysis)
```

Jarvis uses **rule-based routing** (replacing LLM routing in v2.0) with three-layer priority:
1. **Fixed flow templates** - Phrase matching for predefined workflows
2. **Dynamic splitting** - Connector words split + single intent matching
3. **Single intent** - Keyword matching for atomic tasks
4. **Fallback** - LLM direct answer for unrecognized input

### Key Abstractions

**BaseAgent** (`src/agents/base.py`): Abstract base for all agents. Subclasses must implement:
- `get_skills()` - Declare agent capabilities
- `process_task(task: Task) -> Task` - Handle A2A tasks

**BaseSkill** (`src/skills/base.py`): Abstract base for skills. Subclasses must implement:
- `execute(**kwargs) -> SkillOutput` - Skill execution logic
- `get_parameters() -> Dict` - JSON schema for parameters

**SkillManager**: Singleton registry for skill registration and execution.

### A2A Protocol Endpoints
Each agent exposes:
- `GET /.well-known/agent.json` - Agent capabilities card
- `POST /tasks/send` - Submit task
- `GET /tasks/{id}` - Get task status
- `POST /tasks/cancel` - Cancel task

### Data Flow Example
```
User: "分析股票600588并发送邮件到test@example.com"
  ↓
RuleParser.parse() → dynamic split into 2 steps
  ↓
Step 1: delegate_task("stock_agent", "分析股票600588")
  ↓
Step 2: HumanInTheLoop confirmation (email)
  ↓
Step 3: delegate_task("email_agent", send report)
```

### Router Module (`src/router/`)
- **config.py**: Loads YAML config, defines IntentRule, FlowTemplate
- **parser.py**: RuleParser with three-layer parsing logic
- **graph.py**: LangGraph RouterGraph state machine
- **router_rules.yaml**: YAML config for keywords, connectors, templates

## Important Patterns

### Async-First
All agents and skills are async. Both `run()` (sync) and `arun()` (async) are available.

### Security Features
- **PIIGuard** (`src/security/pii_guard.py`): Auto-detects and redacts PII (email, phone, ID card)
- **HumanInTheLoop** (`src/security/human_in_the_loop.py`): Requires user confirmation for email operations

### State Persistence
PostgreSQL checkpoint system (`src/storage/checkpoint.py`) tracks:
- Agent intent analysis
- Routing decisions
- Tool calls and results
- Workflow progress

### Configuration
All config in `src/config/settings.py` as Pydantic models. Environment variables in `.env`:
- `OLLAMA_MODEL`, `OLLAMA_BASE_URL` - LLM settings
- `SMTP_*` - Email configuration
- `POSTGRES_*` - Database settings
- `*_AGENT_URL` - Sub-agent URLs

## Key Files

| File | Purpose |
|------|---------|
| `src/agents/jarvis.py` | Orchestrator with rule-based routing (v2.0) |
| `src/agents/base.py` | BaseAgent abstract class |
| `src/router/parser.py` | RuleParser for intent analysis |
| `src/router/router_rules.yaml` | YAML config for routing rules |
| `src/skills/stock_retriever.py` | Fetch stock data from AKShare |
| `src/skills/stock_report_generator.py` | Generate analysis reports |
| `src/skills/stock_monitor.py` | Intraday stock monitoring with buy/sell alerts |
| `src/skills/stock_position.py` | Stock position tracking (buy/sell/query) |
| `src/scheduler/jobs.py` | Scheduled job definitions |
| `src/scheduler/run.py` | Scheduler entry point |
| `src/a2a/server.py` | FastAPI A2A protocol server |
| `src/a2a/types.py` | Task, Message, AgentCard data models |

## Scheduled Jobs

### 盘中股票监控 (`hourly_stock_monitor`)
工作日 9:00-14:00 每小时执行，仅在"买入"或"卖出"信号时发送邮件。

**流程**:
1. 读取 `STOCK_CODES` 环境变量中的股票代码
2. 使用本地 Finance LLM (finance-llama-8b) 生成技术分析
3. 获取用户持仓和偏好 (PostgreSQL)
4. 调用远端 Qwen 模型给出结构化建议 (持有/买入/卖出)
5. 若为买入或卖出，调用 Email Agent 发送提醒

**配置**:
```bash
STOCK_MONITOR_ENABLED=true    # 启用盘中监控
STOCK_CODES=600588,002230     # 监控股票列表
SMTP_DEFAULT_TO=xxx@xx.com    # 邮件接收地址
```

## Adding New Agents

1. Create `src/agents/my_agent.py`:
```python
class MyAgent(BaseAgent):
    name = "my_agent"
    description = "My custom agent"

    def get_skills(self) -> List[AgentSkill]:
        return [AgentSkill(id="...", name="...", ...)]

    async def process_task(self, task: Task) -> Task:
        # Handle task
        return self._create_text_response(task, result)
```

2. Register in `src/agents/__init__.py`
3. Add URL to Jarvis config (`agent_config.sub_agents`)

## Adding New Skills

1. Create skill class extending `BaseSkill`:
```python
class MySkill(BaseSkill):
    name = "my_skill"
    description = "..."

    async def execute(self, param1: str, **kwargs) -> SkillOutput:
        # Implementation
        return SkillOutput(success=True, result="...")

    def get_parameters(self) -> Dict:
        return {"properties": {"param1": {"type": "string"}}, "required": ["param1"]}
```

2. Register in agent's `_register_default_skills()` method

## MCP Integration

Jarvis supports **Model Context Protocol (MCP)** servers via `MultiServerMCPClient` (`src/mcp/multi_server_client.py`).

### Supported Transports
- **stdio**: Local process communication (Tavily, filesystem, etc.)
- **sse**: Server-Sent Events

### MCP Search in Agents

MCP search (Tavily) is integrated into agents for real-time information retrieval:

| Agent/Skill | Usage |
|-------------|-------|
| **JarvisAgent** | Direct answers with web search for time-sensitive queries |
| **StockReportGeneratorSkill** | Fetches latest stock news before report generation |
| **FundReportGeneratorSkill** | Fetches latest fund news before report generation |

Each component uses **MCPSearchSkill** (`src/skills/mcp_search.py`) which auto-initializes Tavily MCP.

### Configuration
MCP servers are configured in `config/mcp_servers.json`:
```json
{
  "mcpServers": {
    "tavily": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {"TAVILY_API_KEY": "${TAVILY_API_KEY}"}
    }
  }
}
```

### Usage
```python
from src.mcp import MultiServerMCPClient, create_tavily_client
from src.skills.mcp_search import MCPSearchSkill

# Option 1: Use MCPSearchSkill directly
search_skill = MCPSearchSkill()
result = await search_skill.execute("AI news", max_results=5)

# Option 2: Manual MCP client setup
async with MultiServerMCPClient() as client:
    await client.add_stdio_server("tavily", "npx", ["-y", "tavily-mcp"], env={"TAVILY_API_KEY": "..."})
    result = await client.call_tool("tavily", "search", {"query": "AI news"})

# Option 3: Convenience function
client = await create_tavily_client(api_key="...")
schemas = client.get_tool_schemas_for_llm()  # For LLM function calling

# Option 4: Load from config
await client.load_from_config()  # Uses MCP_CONFIG_PATH env var
```

### Environment Variables
- `MCP_CONFIG_PATH`: Path to MCP servers config file
- `TAVILY_API_KEY`: Tavily search API key

## Agent Chat UI Integration

Jarvis 支持 **LangChain Agent Chat UI** 作为 Web 前端界面，通过 LangGraph API 适配层实现。

### 架构

```
┌─────────────────────┐
│   Agent Chat UI     │  (Next.js, port 3000)
│   浏览器前端        │
└──────────┬──────────┘
           │ HTTP/SSE
           ▼
┌─────────────────────┐
│  LangGraph API      │  (FastAPI, port 2024)
│  适配层             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   JarvisAgent       │  (port 8000)
│   调度器            │
└──────────┬──────────┘
           │ A2A Protocol
     ┌─────┼─────┐
     ▼     ▼     ▼
  Stock  Email  Fund
  8001   8002   8003
```

### 启动方式

```bash
# 1. 启动所有子 Agent
./scripts/start_all.sh

# 2. 启动 LangGraph API Server (新终端)
./scripts/start_langgraph_api.sh
# 或者
python -m src.api.langgraph_adapter --port 2024

# 3. 安装和启动 Agent Chat UI (首次需要)
./scripts/setup_agent_chat_ui.sh
cd agent-chat-ui && pnpm dev

# 4. 打开浏览器
open http://localhost:3000
```

### LangGraph API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/assistants` | GET | 列出可用的 assistants |
| `/threads` | POST | 创建会话线程 |
| `/threads/{id}/runs/stream` | POST | 流式执行 (SSE) |
| `/threads/{id}/state` | GET | 获取线程状态 |
| `/health` | GET | 健康检查 |

### Agent Chat UI 配置

在 `agent-chat-ui/.env` 中配置:
```bash
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=jarvis
```

### 关键文件

| 文件 | 说明 |
|------|------|
| `src/api/langgraph_adapter.py` | LangGraph API 适配层 |
| `scripts/start_langgraph_api.sh` | 启动脚本 |
| `scripts/setup_agent_chat_ui.sh` | Agent Chat UI 安装脚本 |

## LangGraph Studio 集成

Jarvis 支持 **LangSmith Studio** 进行图可视化和调试。

### 启动 LangGraph Studio

```bash
# 方式 1: 使用脚本
./scripts/start_langgraph_studio.sh

# 方式 2: 直接使用 CLI
langgraph dev --config langgraph.json --port 8123
```

### 访问 Studio

启动后，打开浏览器访问:
- **Studio UI**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
- **API Docs**: http://127.0.0.1:8123/docs

### 配置文件

`langgraph.json` 定义了图配置:
```json
{
  "python_version": "3.11",
  "dependencies": [".", "langgraph>=0.2.0", "langchain>=0.3.0"],
  "graphs": {
    "jarvis": "./src/studio/graph.py:graph"
  },
  "env": ".env"
}
```

### 关键文件

| 文件 | 说明 |
|------|------|
| `langgraph.json` | LangGraph Studio 配置 |
| `src/studio/graph.py` | Studio 图定义 |
| `scripts/start_langgraph_studio.sh` | Studio 启动脚本 |

### 环境变量

在 `.env` 中配置:
```bash
LANGSMITH_API_KEY=lsv2_pt_xxx  # LangSmith API Key
LANGCHAIN_TRACING_V2=true      # 启用追踪
LANGCHAIN_PROJECT=jarvis       # 项目名称
```

## 流式输出 (Streaming)

Jarvis 支持 LangGraph 的流式输出功能，提供两种模式：

### 流式模式

| 模式 | 说明 |
|------|------|
| `updates` | 输出每个节点的执行进度（开始/完成/耗时） |
| `messages` | 输出 LLM 的 token 流（逐字返回） |
| 组合模式 | 同时输出进度和 token |

### 核心类型

```python
from src.streaming import StreamEvent, StreamEventType, NodeProgress, TokenChunk

# StreamEventType 枚举
class StreamEventType(str, Enum):
    METADATA = "metadata"      # 运行元数据
    NODE_START = "node_start"  # 节点开始执行
    NODE_END = "node_end"      # 节点执行完成
    TOKEN = "token"            # LLM token 输出
    TOKEN_END = "token_end"    # token 流结束
    COMPLETE = "complete"      # 执行完成
    ERROR = "error"            # 错误
```

### 使用方式

**1. JarvisAgent 流式调用**
```python
from src.agents.jarvis import JarvisAgent

jarvis = JarvisAgent(sub_agent_urls=agent_config.sub_agents)

async for event in jarvis.arun_stream(
    "分析股票600588",
    include_progress=True,   # 输出节点进度
    include_tokens=True,     # 输出 LLM token
):
    if event.type == StreamEventType.NODE_START:
        print(f"🔄 {event.progress.message}")
    elif event.type == StreamEventType.NODE_END:
        print(f"✅ {event.progress.message} ({event.progress.duration_ms}ms)")
    elif event.type == StreamEventType.TOKEN:
        print(event.token.content, end="", flush=True)
    elif event.type == StreamEventType.COMPLETE:
        print(f"🎉 完成，共 {event.metadata['total_steps']} 步")
```

**2. StockAgent/FundAgent 流式调用**
```python
from src.agents.stock_agent import StockAgent
from src.a2a import Task, Message, TextPart

agent = StockAgent(checkpoint_manager=checkpoint_manager)
await agent.initialize()

task = Task(
    id="stream-001",
    history=[Message(role="user", parts=[TextPart(text="分析600588")])],
    metadata={"user_id": "user123"}
)

async for event in agent.process_task_stream(
    task,
    include_progress=True,
    include_tokens=True,
):
    # 处理事件...
```

**3. HTTP SSE 流式响应**

LangGraph API 适配层支持 SSE 流式响应：
```bash
curl -X POST http://localhost:2024/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "分析股票600588"}], "stream_mode": ["updates", "messages"]}'
```

### 测试脚本

```bash
# 测试 Jarvis 流式输出
python scripts/test_streaming.py "今天A股行情怎么样"

# 测试 StockAgent 流式输出
python scripts/test_streaming.py --agent stock "分析600588"

# 测试 FundAgent 流式输出
python scripts/test_streaming.py --agent fund "分析基金008089"
```

### 关键文件

| 文件 | 说明 |
|------|------|
| `src/streaming/__init__.py` | 模块导出 |
| `src/streaming/types.py` | StreamEvent, NodeProgress, TokenChunk 类型定义 |
| `src/streaming/handler.py` | StreamHandler 处理 LangGraph 流式输出 |
| `scripts/test_streaming.py` | 流式输出测试脚本 |
