from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import subprocess

# 创建 FastAPI 实例
app = FastAPI()

# ===== 配置 CORS =====
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
# =====================

# ===== 存储训练状态 =====
trainning_status = {
    "status": "idle",  # idle, running, completed, failed
    "message": "No training in progress.",
    "start_time": None,
    "environment": 'SouthElect_stage2',
}
# =====================

# ===== 后台任务函数 =====
def run_training_task():
    global trainning_status # 声明修改全局变量
    try:
        trainning_status["status"] = "running"
        trainning_status["start_time"] = datetime.now().isoformat()
        trainning_status["message"] = "Training in progress..."

        script_path = os.path.join(os.path.dirname(__file__), 'model_src', 'run_longExp.py')
        env_name = trainning_status["environment"]

        # result.stdout 和 result.stderr 分别捕获标准输出和错误输出
        # result.returncode 捕获命令的返回码, 0 表示成功，非 0 表示失败
        result = subprocess.run(
            f'conda activate {env_name} && python "{script_path}"',
            shell=True, # 使用shell解释命令
            capture_output=True, # 捕获命令的输出内容
            text=True, # 将输出内容作为字符串处理
            cwd=os.path.join(os.path.dirname(__file__), "model_src") # 设置工作目录为 model_src
        )

        if result.returncode == 0:
            trainning_status["status"] = "completed"
            trainning_status["message"] = "Training completed successfully."
        else:
            trainning_status["status"] = "failed"
            trainning_status["message"] = f"Training failed with error: {result.stderr[:200]}" # 只返回错误信息的前200字符，避免过长
    except Exception as e:
        trainning_status["status"] = "failed"
        trainning_status["message"] = f"Training failed with exception: {str(e)}"
# =====================

# ===== API端点: 启动训练 =====

@app.post("/api/train/start")
async def start_training(background_tasks: BackgroundTasks):
# ↑ async 表示这是异步函数，FastAPI 能同时处理多个请求而不互相等待
# ↑ background_tasks 是 FastAPI 内置的特殊参数
#   只要你在参数里写上它，FastAPI 会自动帮你注入一个 BackgroundTasks 对象
#   你不需要自己创建，这叫做"依赖注入"
    global trainning_status
    if trainning_status["status"] == "running":
        return {"success": False, "message": "Training is already in progress, do not start a new one."}
        # 返回字典, fastapi自动序列化为json返回给前端
    background_tasks.add_task(run_training_task)
    # ↑ 把 run_training_task 函数丢进"后台队列"
    # 关键：这行代码立刻返回，不等训练完成
    # run_training_task 会在后台异步执行（可能跑几分钟甚至几小时）
    # 前端不会一直等着，正常用页面

    return {
        "success": True,
        "message": "Training task has been started, check status with status API.",
    }

@ app.get("/api/train/status") # 监听请求用get
def get_training_status(): # 查询请求很快, 不用异步
    return trainning_status # 直接返回全局字典


# # 健康检查接口，用于验证后端是否正常启动
# # 注意：CORS 是否生效需要从前端（5173端口）发起 fetch 请求才能真正验证
# @app.get("/")
# def read_root():
#     return {"status": "success", "message": "FastAPI 后端已启动！"}

