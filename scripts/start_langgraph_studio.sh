#!/bin/bash
# 启动 LangGraph Studio 开发服务器
#
# 使用方式:
#   ./scripts/start_langgraph_studio.sh
#
# Studio UI 地址:
#   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}虚拟环境不存在，请先运行: python -m venv venv${NC}"
    exit 1
fi

source venv/bin/activate

# 检查 langgraph CLI
if ! command -v langgraph &> /dev/null; then
    echo -e "${YELLOW}langgraph CLI 未安装，正在安装...${NC}"
    pip install langgraph-cli
fi

# 检查配置文件
if [ ! -f "langgraph.json" ]; then
    echo -e "${YELLOW}langgraph.json 配置文件不存在${NC}"
    exit 1
fi

# 停止已运行的实例
pkill -f "langgraph dev" 2>/dev/null || true
sleep 1

PORT=${LANGGRAPH_STUDIO_PORT:-8123}

echo "============================================================"
echo "  LangGraph Studio - Jarvis"
echo "============================================================"
echo ""
echo -e "  🚀 API:        ${GREEN}http://127.0.0.1:${PORT}${NC}"
echo -e "  🎨 Studio UI:  ${GREEN}https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:${PORT}${NC}"
echo -e "  📚 API Docs:   ${GREEN}http://127.0.0.1:${PORT}/docs${NC}"
echo ""
echo "  按 Ctrl+C 停止服务器"
echo "============================================================"
echo ""

# 启动 LangGraph Studio
langgraph dev --config langgraph.json --port "$PORT"
