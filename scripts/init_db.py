#!/usr/bin/env python3
"""初始化 PostgreSQL 数据库

创建 Jarvis 系统所需的数据库表和索引。

使用方法:
    python scripts/init_db.py

前置条件:
    1. PostgreSQL 已安装并运行
    2. .env 文件中已配置 PostgreSQL 连接信息
    3. 数据库和用户已创建:
       CREATE DATABASE jarvis;
       CREATE USER jarvis_user WITH PASSWORD 'jarvis_pass';
       GRANT ALL PRIVILEGES ON DATABASE jarvis TO jarvis_user;
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import postgres_config
from src.storage import CheckpointManager


async def main():
    """初始化数据库"""
    print("=" * 60)
    print("  Jarvis PostgreSQL 数据库初始化")
    print("=" * 60)

    print(f"\n连接信息:")
    print(f"  - Host: {postgres_config.host}")
    print(f"  - Port: {postgres_config.port}")
    print(f"  - Database: {postgres_config.database}")
    print(f"  - User: {postgres_config.user}")

    print(f"\n正在连接数据库...")

    try:
        manager = CheckpointManager(postgres_config.connection_string)
        await manager.initialize()
        print("✅ 数据库连接成功")
        print("✅ 数据表创建/更新完成")

        # 验证表结构
        print("\n验证数据表...")
        async with manager.pool.acquire() as conn:
            # 查询所有表
            tables = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)

            print(f"\n已创建的表:")
            for row in tables:
                table_name = row['table_name']
                # 获取表的列信息
                columns = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = $1
                    ORDER BY ordinal_position
                """, table_name)

                print(f"\n  📋 {table_name}")
                for col in columns:
                    nullable = "" if col['is_nullable'] == 'YES' else " NOT NULL"
                    print(f"      - {col['column_name']}: {col['data_type']}{nullable}")

        await manager.close()
        print("\n" + "=" * 60)
        print("  ✅ 数据库初始化完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 数据库初始化失败: {e}")
        print("\n请检查:")
        print("  1. PostgreSQL 服务是否运行")
        print("  2. 数据库和用户是否已创建")
        print("  3. .env 文件中的配置是否正确")
        print("\n创建数据库的 SQL 命令:")
        print("  CREATE DATABASE jarvis;")
        print("  CREATE USER jarvis_user WITH PASSWORD 'jarvis_pass';")
        print("  GRANT ALL PRIVILEGES ON DATABASE jarvis TO jarvis_user;")
        print("  -- 连接到 jarvis 数据库后执行:")
        print("  GRANT ALL ON SCHEMA public TO jarvis_user;")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
