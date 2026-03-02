# ===================== 认证工具：密码哈希 + JWT =====================
#
# 生产环境务必将 SECRET_KEY 移入环境变量：
#   import os
#   SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-only-for-dev")
#
# 生成一个安全随机密钥：
#   python -c "import secrets; print(secrets.token_hex(32))"

from datetime import datetime, timedelta, timezone
from typing import Optional

from passlib.context import CryptContext
from jose import JWTError, jwt

# ── 密码哈希 ──
# 使用 sha256_crypt 方案（规避 passlib + bcrypt>=4.0 的兼容性 bug）：
#   - 内置随机 salt，相同密码每次哈希结果不同
#   - 计算缓慢（可调节 rounds），有效防止暴力破解
# 如需换回 bcrypt，将 bcrypt 降级至 <4.0.0：
#   pip install "bcrypt==3.2.2"
#   并将下面改回 schemes=["bcrypt"]
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """将明文密码哈希为 bcrypt 字符串（存入数据库）"""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码是否与哈希值匹配"""
    return pwd_context.verify(plain, hashed)


# ── JWT 配置 ──
# ACCESS_TOKEN 短期有效（默认 7 天，登录后刷新）
# 如需 Refresh Token 机制，可另创建一个长期 token 并存入数据库做吊销控制

SECRET_KEY  = "ai-ts-platform-dev-secret-change-in-production-use-env-var"
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7   # 可改为分钟级别的短期 token（更安全）


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    签发 JWT Access Token。

    data 中通常放 {"sub": username}，sub (subject) 是 JWT 标准字段名。
    exp (expiration) 由此函数自动注入，不需要调用方传入。
    """
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """
    解码并验证 JWT Token，返回 username（sub 字段）。
    Token 无效、过期、篡改时返回 None。
    """
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        return str(username) if username else None
    except JWTError:
        return None
