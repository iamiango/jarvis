#!/bin/bash
# Jarvis 服务停止脚本
# 停止所有 Agent 和调度器

echo "🛑 停止 Jarvis 服务..."

pkill -f "stock_agent" 2>/dev/null && echo "  ✅ Stock Agent 已停止" || echo "  ⚪ Stock Agent 未运行"
pkill -f "email_agent" 2>/dev/null && echo "  ✅ Email Agent 已停止" || echo "  ⚪ Email Agent 未运行"
pkill -f "fund_agent" 2>/dev/null && echo "  ✅ Fund Agent 已停止" || echo "  ⚪ Fund Agent 未运行"
pkill -f "scheduler.run" 2>/dev/null && echo "  ✅ 调度器 已停止" || echo "  ⚪ 调度器 未运行"

echo
echo "✅ 所有服务已停止"
