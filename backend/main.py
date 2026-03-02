# ===================== 应用入口 =====================
# 职责：注册路由、配置中间件、启动时建表。
# 具体业务逻辑均在 routers/ 各子模块中实现。

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── model_src 加入 Python 搜索路径（推理时导入模型/数据集）──
MODEL_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_src')
if MODEL_SRC_DIR not in sys.path:
    sys.path.insert(0, MODEL_SRC_DIR)

# ── 数据库 & ORM（导入 User 确保建表时注册到 Base.metadata）──
from .database     import engine, Base
from .models       import User          # noqa: F401

# ── 路由模块 ──
from .routers.auth      import router as auth_router
from .routers            import train    as train_mod
from .routers            import predict  as predict_mod
from .routers.assistant import router as assistant_router
from .routers.rag       import router as rag_router

# ── 向各路由模块注入路径常量（避免硬编码）──
train_mod.MODEL_SRC_DIR     = MODEL_SRC_DIR
predict_mod.MODEL_SRC_DIR   = MODEL_SRC_DIR
predict_mod.CHECKPOINTS_DIR = os.path.join(MODEL_SRC_DIR, 'checkpoints')
predict_mod.SCRIPTS_DIR     = os.path.join(MODEL_SRC_DIR, 'scripts')

# ── FastAPI 实例 ──
app = FastAPI(title="AI 时序预测平台", version="1.0.0")

# ── 启动时自动建表（幂等，表已存在则跳过）──
# 如需版本化迁移，请改用 Alembic：
#   alembic revision --autogenerate -m "init"  &&  alembic upgrade head
@app.on_event("startup")
def startup_create_tables():
    Base.metadata.create_all(bind=engine)

# ── 挂载路由 ──
app.include_router(auth_router)         # /api/auth/*
app.include_router(train_mod.router)    # /api/train/*
app.include_router(predict_mod.router)  # /api/predict/* & /api/scripts/*
app.include_router(assistant_router)    # /api/assistant/*
app.include_router(rag_router)          # /api/rag/*

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite 开发服务器地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
