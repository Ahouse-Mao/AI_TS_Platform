# ===================== 认证路由 =====================
# 提供三个端点：注册、登录、获取当前用户信息

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models   import User
from auth     import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Bearer Token 解析器
# auto_error=False → Token 缺失时不自动抛 401，由路由自行处理
bearer = HTTPBearer(auto_error=False)


# ─────────────────── Pydantic 数据模型 ───────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50,  description="用户名，2-50 个字符")
    password: str = Field(..., min_length=6, max_length=128, description="密码，至少 6 位")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str


class UserInfo(BaseModel):
    id:         int
    username:   str
    is_active:  bool
    created_at: str
    last_login: str | None


# ─────────────────── 依赖：解析当前登录用户 ───────────────────

def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Session = Depends(get_db),
) -> User:
    """
    从请求头 Authorization: Bearer <token> 中解析当前用户。
    Token 无效或用户不存在时抛出 401 异常。

    在需要登录才能访问的路由中，将此函数作为依赖注入：
        @router.get("/protected")
        def protected(user: User = Depends(get_current_user)):
            ...
    """
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供认证令牌")

    username = decode_access_token(credentials.credentials)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌无效或已过期")

    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已被禁用")

    return user


# ─────────────────── 端点 ───────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    注册新用户。
    - 用户名已存在 → 400
    - 成功 → 直接返回 JWT Token（注册即登录）
    """
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"用户名 '{req.username}' 已被占用",
        )

    user = User(username=req.username, hashed_password=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    用户登录，验证用户名 + 密码，成功后返回 JWT Token。
    故意不区分"用户不存在"与"密码错误"（统一返回 401），防止用户枚举攻击。
    """
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用")

    # 更新最近登录时间
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username)


@router.get("/me", response_model=UserInfo)
def get_me(current_user: User = Depends(get_current_user)):
    """返回当前登录用户的基本信息（需要携带有效 Token）"""
    return UserInfo(
        id         = current_user.id,
        username   = current_user.username,
        is_active  = current_user.is_active,
        created_at = current_user.created_at.isoformat(),
        last_login = current_user.last_login.isoformat() if current_user.last_login else None,
    )
