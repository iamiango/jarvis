#!/usr/bin/env python3
"""数据库迁移脚本 - 添加 stock_monitor_logs 表

用于在现有数据库中添加盘中股票监控日志表。
"""
import asyncio
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import asyncpg


CREATE_STOCK_MONITOR_LOGS_SQL = """
-- 盘中股票监控日志
CREATE TABLE IF NOT EXISTS stock_monitor_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL DEFAULT 'default',
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(255),
    job_time TIMESTAMP NOT NULL DEFAULT NOW(),
    current_price DECIMAL(10,4),
    action VARCHAR(20) NOT NULL,               -- '持有', '买入', '卖出', '建仓'
    reason TEXT,                               -- 操作建议理由
    analysis_summary TEXT,                     -- 技术分析摘要
    llm_request TEXT,                          -- 发送给远端 LLM 的请求报文
    llm_response JSONB,                        -- 远端 LLM 完整返回
    position_shares DECIMAL(18,4),             -- 当时持仓股数
    cost_price DECIMAL(10,4),                  -- 当时成本价
    profit_loss_pct DECIMAL(10,4),             -- 当时盈亏百分比
    email_sent BOOLEAN DEFAULT FALSE,          -- 是否发送了邮件
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_user ON stock_monitor_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_stock ON stock_monitor_logs(stock_code);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_time ON stock_monitor_logs(job_time);
CREATE INDEX IF NOT EXISTS idx_stock_monitor_logs_action ON stock_monitor_logs(action);
"""

# 添加 llm_request 字段的迁移 SQL
ADD_LLM_REQUEST_COLUMN_SQL = """
ALTER TABLE stock_monitor_logs ADD COLUMN IF NOT EXISTS llm_request TEXT;
"""


async def migrate():
    """执行数据库迁移"""
    from src.config import postgres_config

    print("=" * 60)
    print("📦 数据库迁移 - 添加 stock_monitor_logs 表")
    print("=" * 60)

    print(f"\n🔗 连接数据库: {postgres_config.host}:{postgres_config.port}/{postgres_config.database}")

    try:
        conn = await asyncpg.connect(postgres_config.connection_string)

        # 检查表是否已存在
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'stock_monitor_logs'
            )
            """
        )

        if exists:
            print("\n⚠️ 表 stock_monitor_logs 已存在")

            # 检查是否有 llm_request 字段
            has_llm_request = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'stock_monitor_logs' AND column_name = 'llm_request'
                )
                """
            )

            if not has_llm_request:
                print("\n🔧 添加 llm_request 字段...")
                await conn.execute(ADD_LLM_REQUEST_COLUMN_SQL)
                print("✅ llm_request 字段已添加")

            # 检查表结构
            columns = await conn.fetch(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'stock_monitor_logs'
                ORDER BY ordinal_position
                """
            )
            print("\n📋 现有表结构:")
            for col in columns:
                print(f"   - {col['column_name']}: {col['data_type']}")
        else:
            print("\n🚀 创建表 stock_monitor_logs...")
            await conn.execute(CREATE_STOCK_MONITOR_LOGS_SQL)
            print("✅ 表创建成功!")

            # 验证
            columns = await conn.fetch(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'stock_monitor_logs'
                ORDER BY ordinal_position
                """
            )
            print("\n📋 新表结构:")
            for col in columns:
                print(f"   - {col['column_name']}: {col['data_type']}")

        await conn.close()
        print("\n✅ 迁移完成!")

    except Exception as e:
        print(f"\n❌ 迁移失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(migrate())
