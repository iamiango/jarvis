#!/bin/bash
# Jarvis 服务启动脚本
# 启动所有 Agent 和调度器

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}           Jarvis Multi-Agent System Startup${NC}"
echo -e "${GREEN}============================================================${NC}"
echo
echo "📁 工作目录: $PROJECT_DIR"
echo

# 激活虚拟环境
source venv/bin/activate

# 停止已有进程
echo -e "${YELLOW}🛑 停止已有进程...${NC}"
pkill -9 -f "src.agents.stock_agent" 2>/dev/null
pkill -9 -f "src.agents.email_agent" 2>/dev/null
pkill -9 -f "src.agents.fund_agent" 2>/dev/null
pkill -9 -f "src.scheduler.run" 2>/dev/null
sleep 3

# 确认进程已停止
remaining=$(pgrep -f "scheduler.run" 2>/dev/null | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo -e "  ${RED}⚠️ 仍有 $remaining 个残留进程，强制终止...${NC}"
    pkill -9 -f "scheduler.run" 2>/dev/null
    sleep 2
fi

# 创建日志目录
mkdir -p logs

# 启动 Stock Agent
echo -e "${YELLOW}🚀 启动 Stock Agent (port 8001)...${NC}"
python -m src.agents.stock_agent --port 8001 > logs/stock_agent.log 2>&1 &
sleep 2

# 启动 Email Agent
echo -e "${YELLOW}🚀 启动 Email Agent (port 8002)...${NC}"
python -m src.agents.email_agent --port 8002 > logs/email_agent.log 2>&1 &
sleep 2

# 启动 Fund Agent
echo -e "${YELLOW}🚀 启动 Fund Agent (port 8003)...${NC}"
python -m src.agents.fund_agent --port 8003 > logs/fund_agent.log 2>&1 &
sleep 3

# 检查 Agent 状态
echo
echo -e "${YELLOW}📊 检查 Agent 状态...${NC}"
for port in 8001 8002 8003; do
    status=$(curl -s http://localhost:$port/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','failed'))" 2>/dev/null || echo "failed")
    if [ "$status" = "healthy" ]; then
        echo -e "  ✅ Port $port: ${GREEN}healthy${NC}"
    else
        echo -e "  ❌ Port $port: ${RED}failed${NC}"
    fi
done

# 启动调度器
echo
echo -e "${YELLOW}🚀 启动调度器...${NC}"
python -m src.scheduler.run > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
sleep 2

# 检查调度器状态
if ps -p $SCHEDULER_PID > /dev/null 2>&1; then
    echo -e "  ✅ 调度器: ${GREEN}running${NC} (PID: $SCHEDULER_PID)"
else
    echo -e "  ❌ 调度器: ${RED}failed${NC}"
fi

echo
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}                    启动完成!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo
echo "📋 服务列表:"
echo "  - Stock Agent:  http://localhost:8001"
echo "  - Email Agent:  http://localhost:8002"
echo "  - Fund Agent:   http://localhost:8003"
echo "  - 调度器:       每个工作日 14:20 发送基金报告"
echo
echo "📁 日志文件:"
echo "  - logs/stock_agent.log"
echo "  - logs/email_agent.log"
echo "  - logs/fund_agent.log"
echo "  - logs/scheduler.log"
echo
echo "💡 常用命令:"
echo "  - 查看调度器日志: tail -f logs/scheduler.log"
echo "  - 立即执行任务:   python -m src.scheduler.run --run-now daily_fund_report"
echo "  - 停止所有服务:   ./scripts/stop_all.sh"
echo
