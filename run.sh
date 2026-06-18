#!/bin/bash
# Jarvis 一键启动脚本
# 停止所有进程 -> 启动所有 Agent -> 打开对话框

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}              Jarvis 一键启动脚本${NC}"
echo -e "${CYAN}============================================================${NC}"
echo

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}❌ 未找到虚拟环境，请先运行: python3.10 -m venv venv${NC}"
    exit 1
fi

# ============================================================
# 1. 停止所有进程
# ============================================================
echo -e "${YELLOW}🛑 [1/3] 停止所有 Agent 进程...${NC}"

pkill -9 -f "src.agents.stock_agent" 2>/dev/null && echo "  ✅ Stock Agent 已停止" || echo "  ⚪ Stock Agent 未运行"
pkill -9 -f "src.agents.email_agent" 2>/dev/null && echo "  ✅ Email Agent 已停止" || echo "  ⚪ Email Agent 未运行"
pkill -9 -f "src.agents.fund_agent" 2>/dev/null && echo "  ✅ Fund Agent 已停止" || echo "  ⚪ Fund Agent 未运行"
pkill -9 -f "src.agents.jarvis" 2>/dev/null && echo "  ✅ Jarvis Agent 已停止" || echo "  ⚪ Jarvis Agent 未运行"
pkill -9 -f "src.scheduler.run" 2>/dev/null && echo "  ✅ 调度器 已停止" || echo "  ⚪ 调度器 未运行"
pkill -9 -f "src.api.langgraph_adapter" 2>/dev/null && echo "  ✅ LangGraph API 已停止" || echo "  ⚪ LangGraph API 未运行"

sleep 2
echo

# ============================================================
# 2. 启动所有 Agent
# ============================================================
echo -e "${YELLOW}🚀 [2/3] 启动所有 Agent...${NC}"

# 创建日志目录
mkdir -p logs

# 启动 Stock Agent
echo -e "  启动 Stock Agent (port 8001)..."
python -m src.agents.stock_agent --port 8001 > logs/stock_agent.log 2>&1 &

# 启动 Email Agent
echo -e "  启动 Email Agent (port 8002)..."
python -m src.agents.email_agent --port 8002 > logs/email_agent.log 2>&1 &

# 启动 Fund Agent
echo -e "  启动 Fund Agent (port 8003)..."
python -m src.agents.fund_agent --port 8003 > logs/fund_agent.log 2>&1 &

# 启动 Scheduler 调度器（盘中股票监控等定时任务）
echo -e "  启动 Scheduler 调度器（盘中股票监控）..."
python -m src.scheduler.run > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!

# 启动 LangGraph API Server
LANGGRAPH_PORT=${LANGGRAPH_PORT:-2024}
echo -e "  启动 LangGraph API Server (port $LANGGRAPH_PORT)..."
python -m src.api.langgraph_adapter --port $LANGGRAPH_PORT > logs/langgraph_api.log 2>&1 &

# 等待 Agent 启动（带重试机制）
echo
echo -e "${YELLOW}📊 等待 Agent 启动...${NC}"

max_retries=10
retry_interval=2

for port in 8001 8002 8003; do
    retries=0
    while [ $retries -lt $max_retries ]; do
        status=$(curl -s http://localhost:$port/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','failed'))" 2>/dev/null || echo "failed")
        if [ "$status" = "healthy" ]; then
            echo -e "  ✅ Port $port: ${GREEN}healthy${NC}"
            break
        fi
        retries=$((retries + 1))
        if [ $retries -lt $max_retries ]; then
            echo -e "  ⏳ Port $port: 等待中... ($retries/$max_retries)"
            sleep $retry_interval
        fi
    done
    if [ "$status" != "healthy" ]; then
        echo -e "  ❌ Port $port: ${RED}failed${NC}"
        all_healthy=false
    fi
done

# 检查 LangGraph API Server
retries=0
while [ $retries -lt $max_retries ]; do
    if curl -s http://localhost:$LANGGRAPH_PORT/health > /dev/null 2>&1; then
        echo -e "  ✅ LangGraph API (port $LANGGRAPH_PORT): ${GREEN}healthy${NC}"
        break
    fi
    retries=$((retries + 1))
    if [ $retries -lt $max_retries ]; then
        echo -e "  ⏳ LangGraph API: 等待中... ($retries/$max_retries)"
        sleep $retry_interval
    fi
done
if [ $retries -eq $max_retries ]; then
    echo -e "  ❌ LangGraph API (port $LANGGRAPH_PORT): ${RED}failed${NC}"
fi

# 检查 Scheduler 调度器
sleep 1
if ps -p $SCHEDULER_PID > /dev/null 2>&1; then
    echo -e "  ✅ Scheduler 调度器: ${GREEN}running${NC} (PID: $SCHEDULER_PID)"
else
    echo -e "  ❌ Scheduler 调度器: ${RED}failed${NC}"
fi

all_healthy=true
for port in 8001 8002 8003; do
    status=$(curl -s http://localhost:$port/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','failed'))" 2>/dev/null || echo "failed")
    if [ "$status" != "healthy" ]; then
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo
    echo -e "${RED}⚠️ 部分 Agent 启动失败，请检查日志:${NC}"
    echo "  - logs/stock_agent.log"
    echo "  - logs/email_agent.log"
    echo "  - logs/fund_agent.log"
    echo "  - logs/scheduler.log"
    echo "  - logs/langgraph_api.log"
    echo
    read -p "是否继续启动对话框? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo
echo -e "${CYAN}----------------------------------------------${NC}"
echo -e "${CYAN}  LangGraph API: http://localhost:$LANGGRAPH_PORT${NC}"
echo -e "${CYAN}  API 文档: http://localhost:$LANGGRAPH_PORT/docs${NC}"
echo -e "${CYAN}  调度器日志: tail -f logs/scheduler.log${NC}"
echo -e "${CYAN}----------------------------------------------${NC}"

echo

# ============================================================
# 3. 启动对话框
# ============================================================
echo -e "${YELLOW}💬 [3/3] 启动 Jarvis 对话框...${NC}"
echo
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Jarvis 准备就绪！输入 'exit' 或 'quit' 退出${NC}"
echo -e "${GREEN}============================================================${NC}"
echo

# 运行 main.py 对话框
python main.py

# 对话结束后的清理提示
echo
echo -e "${YELLOW}💡 对话已结束。Agent 仍在后台运行。${NC}"
echo "  - 查看日志: tail -f logs/*.log"
echo "  - LangGraph API: http://localhost:$LANGGRAPH_PORT"
echo "  - 停止服务: ./scripts/stop_all.sh"
