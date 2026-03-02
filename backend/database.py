# ===================== 数据库引擎与会话 =====================
#
# 当前使用 SQLite（零配置，文件即数据库）。
# 如需迁移到 PostgreSQL，只需完成以下两步：
#
#   1. 安装驱动：
#      pip install psycopg2-binary          # 同步驱动（对应本文件当前使用的同步 SQLAlchemy）
#      # 或 pip install asyncpg             # 异步驱动（如改用 async SQLAlchemy）
#
#   2. 修改 DATABASE_URL（下方第二行取消注释，第一行注释掉）：
#      DATABASE_URL = "postgresql://user:password@localhost:5432/ai_ts_platform"
#      # 异步版本：
#      DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/ai_ts_platform"
#
#   生产环境建议通过环境变量读取：
#      import os
#      DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_ts.db")
#
# ─────────────────────────────────────────────────────────────

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── 当前：SQLite ──
# connect_args 中的 check_same_thread=False 是 SQLite 专用参数，
# 允许同一连接被多个线程使用（FastAPI 多线程场景需要）。
# 迁移到 PostgreSQL 后，删除 connect_args 参数即可。
DATABASE_URL = "sqlite:///./ai_ts.db"
# DATABASE_URL = "postgresql://user:password@localhost:5432/ai_ts_platform"  # ← PostgreSQL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # ← PostgreSQL 时删除此行
)

# SessionLocal: 每次请求创建一个独立的数据库会话
# autocommit=False → 需要手动 commit，确保事务安全
# autoflush=False  → 不自动 flush，减少意外 SQL
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 所有 ORM 模型的基类（从此处继承即可自动映射到数据库表）
class Base(DeclarativeBase):
    pass


# ── FastAPI 依赖注入：获取 DB Session ──
# 用法：在路由函数参数中写 db: Session = Depends(get_db)
# yield 保证请求结束后一定关闭 session，即使发生异常
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
