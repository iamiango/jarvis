#!/bin/bash
# 启动 Jarvis LangGraph API Server
# 供 Agent Chat UI 使用

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 默认端口
PORT=${LANGGRAPH_PORT:-2024}

echo "=============================================="
echo "  Jarvis LangGraph API Server"
echo "=============================================="
echo "  端口: $PORT"
echo "  API 文档: http://localhost:$PORT/docs"
echo "----------------------------------------------"
echo "  Agent Chat UI 配置:"
echo "    NEXT_PUBLIC_API_URL=http://localhost:$PORT"
echo "    NEXT_PUBLIC_ASSISTANT_ID=jarvis"
echo "=============================================="
echo ""

# 检查子 Agent 是否运行
check_agent() {
    local url=$1
    local name=$2
    if curl -s "${url}/health" > /dev/null 2>&1; then
        echo "✅ $name 已运行"
        return 0
    else
        echo "⚠️  $name 未运行 ($url)"
        return 1
    fi
}

echo "检查子 Agent 状态..."
check_agent "http://localhost:8001" "Stock Agent" || true
check_agent "http://localhost:8002" "Email Agent" || true
check_agent "http://localhost:8003" "Fund Agent" || true
echo ""

# 启动 LangGraph API Server
echo "启动 LangGraph API Server..."
python -m src.api.langgraph_adapter --port $PORT
