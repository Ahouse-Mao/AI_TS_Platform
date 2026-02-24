from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 实例
app = FastAPI()

# ===== 这里就是配置 CORS 的核心代码 =====
app.add_middleware(
    CORSMiddleware,
    # 允许访问的源列表（写上你前端 Vite 的运行地址）
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    # 允许所有的请求方法（GET, POST, PUT, DELETE 等）
    allow_methods=["*"], 
    # 允许所有的请求头
    # 请求头是浏览器/客户端发送 HTTP 请求时，附带的一些"说明信息"，告诉服务器"我是谁、我要什么、我带了什么格式的数据"。
    allow_headers=["*"], 
)
# ======================================

# 健康检查接口，用于验证后端是否正常启动
# 注意：CORS 是否生效需要从前端（5173端口）发起 fetch 请求才能真正验证
@app.get("/")
def read_root():
    return {"status": "success", "message": "FastAPI 后端已启动！"}

