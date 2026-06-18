# Jarvis LLM 调用场景整理

本文档整理了 Jarvis 项目中所有调用 LLM 的场景，包括本地 Ollama 模型和远端 Qwen API。

## 模型配置概览

| 模型 | 类型 | 用途 | 配置来源 |
|------|------|------|----------|
| `qwen3.5:9b` | 本地 Ollama | 通用对话、Tool Calling、路由 | `OLLAMA_MODEL` 环境变量 |
| `martain7r/finance-llama-8b:q4_k_m` | 本地 Ollama | 金融技术分析（英文） | 代码硬编码 |
| `qwen-plus` | 远端 DashScope API | 翻译、路由决策、操作建议 | `QWEN_MODEL` 环境变量 |

---

## 一、JarvisAgent (调度器)

### 1.1 直接回答 - 本地 Ollama
**位置**: `src/agents/jarvis.py:690`
**模型**: `qwen3.5:9b` (本地 Ollama)
**用途**: 当 RuleParser 解析失败或无法匹配任何 Agent 时，使用本地 LLM 直接回答用户问题
**触发条件**: `action_type == "direct_answer"` 且未命中任何路由规则

```python
llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=0.7)
response = await llm.ainvoke(messages)
```

### 1.2 MCP Tool Calling - 本地 Ollama
**位置**: `src/agents/jarvis.py:718`
**模型**: `qwen3.5:9b` (本地 Ollama)
**用途**: 当 MCP 服务可用时，进行搜索工具调用，获取实时信息后生成回答

---

## 二、StockAgent (股票 Agent)

### 2.1 偏好设置 Tool Calling - 本地 Ollama
**位置**: `src/agents/stock_agent.py:442`
**模型**: `qwen3.5:9b` (本地 Ollama)
**用途**: 解析用户的偏好设置意图，通过 Tool Calling 调用 `set_stock_preference` 工具

```python
llm_with_tools = self._ollama_llm.bind_tools(tools)
response = await llm_with_tools.ainvoke(messages)
```

### 2.2 投资建议总结 - 远端 Qwen
**位置**: `src/agents/stock_agent.py:788`
**模型**: `qwen-plus` (远端 DashScope)
**用途**: 基于技术分析报告生成 300 字以内的投资建议总结

```python
response = client.chat.completions.create(
    model=qwen_config.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
```

---

## 三、FundAgent (基金 Agent)

### 3.1 LangGraph 状态机执行
**位置**: `src/agents/fund_agent.py:690`
**模型**: 通过 LangGraph StateGraph 调用
**用途**: 执行基金分析的完整工作流

---

## 四、StockReportGeneratorSkill (股票报告生成)

### 4.1 技术分析 - 本地 Finance LLM
**位置**: `src/skills/stock_report_generator.py:826`
**模型**: `martain7r/finance-llama-8b:q4_k_m` (本地 Ollama)
**用途**: 基于原始 OHLCV 数据生成英文技术分析报告（包含 MACD、RSI、KDJ 等指标分析）

```python
llm = ChatOllama(
    model=self.FINANCE_MODEL,
    base_url=ollama_config.base_url,
    temperature=0.7,
)
response = await llm.ainvoke(prompt)
```

### 4.2 中文翻译 - 远端 Qwen
**位置**: `src/skills/stock_report_generator.py:435`
**模型**: `qwen-plus` (远端 DashScope)
**用途**: 将 Finance LLM 生成的英文分析报告翻译成中文

```python
response = await client.chat.completions.create(
    model=qwen_config.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)
```

---

## 五、StockRetrieverSkill (股票数据检索)

### 5.1 技术分析 - 本地 Finance LLM
**位置**: `src/skills/stock_retriever.py:111`
**模型**: `martain7r/finance-llama-8b:q4_k_m` (本地 Ollama)
**用途**: 获取股票数据后进行技术指标分析

```python
llm = ChatOllama(
    model=self.FINANCE_MODEL,
    base_url=ollama_config.base_url,
    temperature=0.3,
)
response = await llm.ainvoke(prompt)
```

---

## 六、StockMonitorSkill (盘中股票监控)

### 6.1 技术分析 - 本地 Finance LLM
**位置**: `src/skills/stock_monitor.py:279`
**模型**: `martain7r/finance-llama-8b:q4_k_m` (本地 Ollama)
**用途**: 盘中监控时对股票进行技术面分析（简要分析）

```python
llm = ChatOllama(
    model=self.FINANCE_MODEL,
    base_url=ollama_config.base_url,
    temperature=0.3,
)
response = await llm.ainvoke(prompt)
```

### 6.2 操作建议决策 - 远端 Qwen
**位置**: `src/skills/stock_monitor.py:366`
**模型**: `qwen-plus` (远端 DashScope)
**用途**: 结合技术分析、用户持仓、风险偏好，输出结构化操作建议（持有/买入/卖出/建仓）

```python
response = await client.chat.completions.create(
    model=qwen_config.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)
```

**Prompt 模板**: `src/skills/prompts/qwen_trade_advice.md`

### 6.3 邮件内容整理 - 本地 Ollama
**位置**: `src/skills/stock_monitor.py:450`
**模型**: `qwen3.5:9b` (本地 Ollama)
**用途**: 将操作建议整理成自然语言邮件内容

```python
llm = ChatOllama(
    model=ollama_config.model,  # qwen3.5:9b
    base_url=ollama_config.base_url,
    temperature=0.5,
)
response = await llm.ainvoke(prompt)
```

---

## 七、FundReportSkill (基金报告生成)

### 7.1 技术分析 - 本地 Finance LLM
**位置**: `src/skills/fund_report.py:914`
**模型**: `martain7r/finance-llama-8b:q4_k_m` (本地 Ollama)
**用途**: 基于基金数据生成英文技术分析报告

```python
llm = ChatOllama(
    model=self.FINANCE_MODEL,
    base_url=ollama_config.base_url,
    temperature=0.7,
)
response = await llm.ainvoke(prompt)
```

### 7.2 中文翻译 - 远端 Qwen
**位置**: `src/skills/fund_report.py:418`
**模型**: `qwen-plus` (远端 DashScope)
**用途**: 将 Finance LLM 生成的英文分析报告翻译成中文

```python
response = await client.chat.completions.create(
    model=qwen_config.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)
```

---

## 八、LLM Router (LLM 路由器)

### 8.1 意图路由 - 本地 Ollama
**位置**: `src/router/llm_router.py:274`
**模型**: `qwen3.5:9b` (本地 Ollama)
**用途**: 分析用户输入，决定应该路由到哪个 Agent 或是否需要直接回答

```python
llm = ChatOllama(
    model=self._model,
    base_url=self._base_url,
    temperature=self._temperature,
    format="json",  # 强制 JSON 输出
)
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_message),
]
response = await llm.ainvoke(messages)
```

---

## 九、ToolExecutorAgent (工具执行 Agent)

### 9.1 LangGraph Agent 执行 - 本地 Ollama
**位置**: `src/agents/tool_executor.py:283`
**模型**: `qwen3.5:9b` (本地 Ollama)
**用途**: 使用 LangGraph 的 `create_react_agent` 执行带工具调用的对话

```python
llm = ChatOllama(
    model=self._model,
    base_url=ollama_config.base_url,
    temperature=self._temperature,
)
result = await self._agent.ainvoke({
    "messages": [{"role": "user", "content": user_input}],
})
```

---

## 十、LangGraph Studio (开发调试)

### 10.1 RouterGraph 状态机
**位置**: `src/router/graph.py:160`
**用途**: LangGraph StateGraph 状态机执行，用于路由决策流程

### 10.2 Studio Graph
**位置**: `src/studio/graph.py:514`
**用途**: LangGraph Studio 开发调试使用的状态机

---

## 调用场景汇总表

| 场景 | 模型 | 类型 | 文件位置 | 用途 |
|------|------|------|----------|------|
| Jarvis 直接回答 | qwen3.5:9b | 本地 | jarvis.py:690 | 无法路由时直接回答 |
| Jarvis MCP 搜索 | qwen3.5:9b | 本地 | jarvis.py:718 | 搜索后生成回答 |
| Stock 偏好设置 | qwen3.5:9b | 本地 | stock_agent.py:442 | Tool Calling |
| Stock 建议总结 | qwen-plus | 远端 | stock_agent.py:788 | 投资建议 |
| Stock 报告分析 | finance-llama-8b | 本地 | stock_report_generator.py:826 | 技术分析 |
| Stock 报告翻译 | qwen-plus | 远端 | stock_report_generator.py:435 | 中文翻译 |
| Stock 数据分析 | finance-llama-8b | 本地 | stock_retriever.py:111 | 技术分析 |
| Stock 盘中分析 | finance-llama-8b | 本地 | stock_monitor.py:279 | 简要分析 |
| Stock 操作建议 | qwen-plus | 远端 | stock_monitor.py:366 | 买卖决策 |
| Stock 邮件整理 | qwen3.5:9b | 本地 | stock_monitor.py:450 | 邮件内容 |
| Fund 报告分析 | finance-llama-8b | 本地 | fund_report.py:914 | 技术分析 |
| Fund 报告翻译 | qwen-plus | 远端 | fund_report.py:418 | 中文翻译 |
| LLM 路由决策 | qwen3.5:9b | 本地 | llm_router.py:274 | 意图路由 |
| 工具执行 Agent | qwen3.5:9b | 本地 | tool_executor.py:283 | Tool Calling |

---

## 调用链路图

```
用户输入
    │
    ▼
┌─────────────────┐
│  RuleParser     │  (规则解析，无 LLM)
└────────┬────────┘
         │ 若无法匹配
         ▼
┌─────────────────┐
│  LLM Router     │  远端 Qwen (意图路由)
│  qwen-plus      │
└────────┬────────┘
         │
    ┌────┴────┬────────────────────┐
    ▼         ▼                    ▼
┌────────┐ ┌────────┐          ┌────────┐
│ Stock  │ │ Fund   │          │ Direct │
│ Agent  │ │ Agent  │          │ Answer │
└────┬───┘ └────┬───┘          └────┬───┘
     │          │                   │
     ▼          ▼                   ▼
┌─────────────────────────────────────────────┐
│  本地 Finance LLM (技术分析)                  │
│  martain7r/finance-llama-8b:q4_k_m          │
└────────────────────┬────────────────────────┘
                     │ 英文报告
                     ▼
┌─────────────────────────────────────────────┐
│  远端 Qwen (翻译/建议)                        │
│  qwen-plus                                  │
└────────────────────┬────────────────────────┘
                     │ 中文报告 / 操作建议
                     ▼
                  用户输出
```

---

## 模型选型说明

### 本地 Ollama 模型

1. **qwen3.5:9b** - 通用对话模型
   - 用于：对话、Tool Calling、简单任务
   - 优点：响应快、无网络依赖
   - 配置：`OLLAMA_MODEL` 环境变量

2. **martain7r/finance-llama-8b:q4_k_m** - 金融专用模型
   - 用于：股票/基金技术分析
   - 优点：金融领域专业、英文输出质量高
   - 注意：输出为英文，需配合翻译

### 远端 Qwen API

1. **qwen-plus** - 阿里云 DashScope API
   - 用于：翻译、复杂推理、路由决策
   - 优点：中文能力强、推理质量高
   - 配置：`QWEN_API_KEY`、`QWEN_MODEL` 环境变量
   - 费用：按 token 计费

---

## 优化建议

1. **本地化翻译**：考虑使用本地 qwen3.5:9b 替代远端翻译，减少 API 成本
2. **缓存机制**：对相同股票代码的分析结果进行缓存，避免重复调用
3. **批量处理**：盘中监控多只股票时，考虑批量调用优化
4. **模型切换**：根据任务复杂度动态选择模型（简单任务用小模型）
