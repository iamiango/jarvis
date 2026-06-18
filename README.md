# Jarvis Agent

基于 LangGraph 框架的智能 Agent，支持 Skill 调用和 MCP Server 集成，对接本地 Ollama 部署的 Qwen3.5:7b 模型。

## 环境要求

- Python 3.10+
- Ollama 本地服务
- qwen3:7b 模型

## 快速开始

### 1. 安装 Ollama 和模型

```bash
# 安装 Ollama (macOS)
brew install ollama

# 启动 Ollama 服务
ollama serve

# 下载模型
ollama pull qwen3:7b
```

### 2. 激活虚拟环境

```bash
source venv/bin/activate
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件修改配置
```

### 4. 运行测试

```bash
python test_framework.py
```

### 5. 启动交互式 Agent

```bash
python main.py
```

## 项目结构

```
jarvis/
├── src/
│   ├── agents/          # Agent 实现
│   │   └── jarvis.py    # 主 Agent 类
│   ├── skills/          # Skill 系统
│   │   ├── base.py      # Skill 基类和管理器
│   │   └── builtin.py   # 内置 Skills
│   ├── mcp/             # MCP Client
│   │   └── client.py    # MCP 客户端实现
│   ├── tools/           # LangChain Tools
│   │   └── langchain_tools.py
│   └── config/          # 配置
│       └── settings.py
├── main.py              # 主入口
├── test_framework.py    # 测试脚本
└── .env.example         # 环境变量示例
```

## 核心功能

### 1. Skill 系统

创建自定义 Skill:

```python
from src.skills import BaseSkill, SkillOutput, skill_manager

class MySkill(BaseSkill):
    name = "my_skill"
    description = "我的自定义 Skill"

    async def execute(self, **kwargs) -> SkillOutput:
        return SkillOutput(success=True, result="执行结果")

# 注册 Skill
skill_manager.register(MySkill())
```

### 2. MCP Server 集成

```python
from src.mcp import mcp_manager

# 添加 MCP Server
await mcp_manager.add_server("my_server", "http://localhost:3000")

# 调用 MCP Tool
result = await mcp_manager.call_tool("mcp_my_server_tool_name", {"arg": "value"})
```

### 3. Agent 使用

```python
from src import JarvisAgent

agent = JarvisAgent()
agent.build()

# 同步调用
response = agent.run("你好")

# 异步调用
response = await agent.arun("你好")

# 流式调用
for event in agent.stream("你好"):
    print(event)
```

## 内置 Skills

- `calculator`: 基本数学运算
- `search`: 搜索功能（模拟）
- `weather`: 天气查询（模拟）

## 许可证

MIT
