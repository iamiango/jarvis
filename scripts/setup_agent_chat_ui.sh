#!/bin/bash
# 安装和配置 Agent Chat UI
# 用于与 Jarvis LangGraph API 交互

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHAT_UI_DIR="$PROJECT_DIR/agent-chat-ui"

echo "=============================================="
echo "  Agent Chat UI 安装脚本"
echo "=============================================="

# 检查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "安装 pnpm..."
    npm install -g pnpm
fi

# 检查是否已存在
if [ -d "$CHAT_UI_DIR" ]; then
    echo "Agent Chat UI 目录已存在: $CHAT_UI_DIR"
    echo "跳过克隆，进入配置..."
else
    echo "克隆 Agent Chat UI..."
    git clone https://github.com/langchain-ai/agent-chat-ui.git "$CHAT_UI_DIR"
fi

cd "$CHAT_UI_DIR"

# 创建环境配置
echo ""
echo "配置环境变量..."
cat > .env << 'EOF'
# Jarvis LangGraph API 配置
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=jarvis

# 认证方案 (本地开发留空)
NEXT_PUBLIC_AUTH_SCHEME=
EOF

echo "✅ 环境配置已创建: $CHAT_UI_DIR/.env"

# 安装依赖
echo ""
echo "安装依赖..."
pnpm install

echo ""
echo "=============================================="
echo "  安装完成!"
echo "=============================================="
echo ""
echo "使用方法:"
echo ""
echo "1. 启动 Jarvis 子 Agent (在 jarvis 目录):"
echo "   ./scripts/start_all.sh"
echo ""
echo "2. 启动 LangGraph API Server (在 jarvis 目录):"
echo "   ./scripts/start_langgraph_api.sh"
echo ""
echo "3. 启动 Agent Chat UI (在 agent-chat-ui 目录):"
echo "   cd agent-chat-ui && pnpm dev"
echo ""
echo "4. 打开浏览器访问:"
echo "   http://localhost:3000"
echo ""
echo "=============================================="
