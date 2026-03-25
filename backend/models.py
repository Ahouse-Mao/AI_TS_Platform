# ===================== ORM 数据模型 =====================
#
# 迁移到 PostgreSQL 时，模型本身无需改动。
# SQLAlchemy ORM 层与底层数据库无关，只需修改 database.py 中的连接字符串。
#
# 扩展字段示例（按需取消注释）：
#   - email       : 邮箱（唯一性约束）
#   - role        : 权限角色（'admin' / 'user'）
#   - is_active   : 软删除标记
#   - last_login  : 最近登录时间

from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """用户账户表"""
    __tablename__ = "users"

    # 主键：自增整数 ID
    id = Column(Integer, primary_key=True, index=True)

    # 用户名：唯一，不可为空，建立索引方便快速查询
    username = Column(String(50), unique=True, index=True, nullable=False)

    # 哈希后的密码（bcrypt 格式，原始密码不存储）
    hashed_password = Column(String(200), nullable=False)

    # 账户启用状态（False = 已禁用，不允许登录）
    is_active = Column(Boolean, default=True, nullable=False)

    # 创建时间：server_default 由数据库生成，不依赖应用层时区
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 最近登录时间：登录成功后更新
    last_login = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r}>"
