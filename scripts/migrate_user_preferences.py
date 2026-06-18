#!/usr/bin/env python3
"""迁移脚本：创建 user_preferences 表

如果数据库已存在但没有 user_preferences 表，运行此脚本添加该表。

使用方法:
    python scripts/migrate_user_preferences.py

说明:
    - 此脚本是幂等的，可以安全地多次运行
    - 如果表已存在，不会有任何影响
    - 会自动创建所需的索引
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import postgres_config

# 创建 user_preferences 表的 SQL
CREATE_USER_PREFERENCES_SQL = """
-- 用户偏好存储（长期记忆）
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,  -- 'stock', 'fund', 'email', 'general'
    preference_key VARCHAR(255) NOT NULL,
    preference_value JSONB NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(50) DEFAULT 'explicit',  -- 'explicit' | 'inferred'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, category, preference_key)
);

-- 创建索引（IF NOT EXISTS 确保幂等性）
CREATE INDEX IF NOT EXISTS idx_pref_user ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_pref_category ON user_preferences(user_id, category);
"""


async def main():
    """执行迁移"""
    print("=" * 60)
    print("  User Preferences 表迁移脚本")
    print("=" * 60)

    print(f"\n连接信息:")
    print(f"  - Host: {postgres_config.host}")
    print(f"  - Port: {postgres_config.port}")
    print(f"  - Database: {postgres_config.database}")
    print(f"  - User: {postgres_config.user}")

    try:
        import asyncpg

        print(f"\n正在连接数据库...")
        conn = await asyncpg.connect(postgres_config.connection_string)

        # 检查表是否已存在
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'user_preferences'
            )
        """)

        if table_exists:
            print("ℹ️  user_preferences 表已存在")
        else:
            print("📝 正在创建 user_preferences 表...")

        # 执行创建语句（IF NOT EXISTS 确保幂等性）
        await conn.execute(CREATE_USER_PREFERENCES_SQL)
        print("✅ 表结构已确认")

        # 验证表结构
        print("\n验证表结构...")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'user_preferences'
            ORDER BY ordinal_position
        """)

        print(f"\n📋 user_preferences 表结构:")
        for col in columns:
            nullable = "" if col['is_nullable'] == 'YES' else " NOT NULL"
            default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
            print(f"    - {col['column_name']}: {col['data_type']}{nullable}{default}")

        # 验证索引
        indexes = await conn.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'user_preferences'
        """)

        print(f"\n📇 索引:")
        for idx in indexes:
            print(f"    - {idx['indexname']}")

        # 显示现有记录数
        count = await conn.fetchval("SELECT COUNT(*) FROM user_preferences")
        print(f"\n📊 现有记录数: {count}")

        await conn.close()

        print("\n" + "=" * 60)
        print("  ✅ 迁移完成!")
        print("=" * 60)

    except ImportError:
        print("\n❌ 缺少 asyncpg 依赖")
        print("   请运行: pip install asyncpg")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 迁移失败: {e}")
        print("\n请检查:")
        print("  1. PostgreSQL 服务是否运行")
        print("  2. 数据库连接配置是否正确")
        print("  3. 用户是否有创建表的权限")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
