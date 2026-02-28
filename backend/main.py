from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import subprocess
import re
from typing import Optional

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


# ===== 工具函数: 解析 Checkpoint 文件夹名 =====

# 已知模型名列表，用于从文件夹名中定位模型字段的位置
KNOWN_MODELS = ['Autoformer', 'Informer', 'Transformer', 'DLinear',
                'PatchTST', 'Linear', 'NLinear', 'FEDformer', 'Pyraformer']

# 特征模式的中文说明
FEATURES_DESC = {
    'M':  '多变量 → 多变量',
    'S':  '单变量 → 单变量',
    'MS': '多变量 → 单变量',
}

def parse_checkpoint_name(name: str) -> dict:
    """
    将 Checkpoint 文件夹名解析为结构化信息。
    命名格式：
      {model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}
      _dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}
      _eb{embed}_dt{distil}_{des}_{exp_id}
    """
    # 用正则提取 _ft... 之后的所有键值对
    kv_pattern = (
        r'_ft(.+?)_sl(\d+)_ll(\d+)_pl(\d+)'
        r'_dm(\d+)_nh(\d+)_el(\d+)_dl(\d+)_df(\d+)'
        r'_fc(\d+)_eb(.+?)_dt(.+?)_(.+?)_(\d+)$'
    )
    match = re.search(kv_pattern, name)
    if not match:
        return {"folder_name": name, "parse_error": True}

    # match.start() 之前的部分是 "{model_id}_{model}_{data}"
    prefix = name[:match.start()]

    # 在 prefix 里查找已知模型名，从而切分出 model_id 和 data
    model, model_id, dataset = None, prefix, ''
    for m in KNOWN_MODELS:
        if f'_{m}_' in prefix:
            parts = prefix.split(f'_{m}_', 1)  # 只切第一刀
            model_id = parts[0]
            dataset  = parts[1]
            model    = m
            break

    features_raw = match.group(1)
    return {
        "folder_name": name,
        # ── 核心信息 ──
        "model_id":  model_id,
        "model":     model or "Unknown",
        "dataset":   dataset,
        # ── 序列长度 ──
        "features":      features_raw,
        "features_desc": FEATURES_DESC.get(features_raw, features_raw),
        "seq_len":   int(match.group(2)),   # 输入步长
        "label_len": int(match.group(3)),   # 标签步长
        "pred_len":  int(match.group(4)),   # 预测步长
        # ── 模型超参 ──
        "d_model":   int(match.group(5)),
        "n_heads":   int(match.group(6)),
        "e_layers":  int(match.group(7)),
        "d_layers":  int(match.group(8)),
        "d_ff":      int(match.group(9)),
        "factor":    int(match.group(10)),
        "embed":     match.group(11),
        "distil":    match.group(12),
        # ── 实验标识 ──
        "des":       match.group(13),
        "exp_id":    int(match.group(14)),
    }
# =====================

# ===== API端点: 获取 Checkpoint 列表 =====

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'model_src', 'checkpoints')

@app.get("/api/predict/checkpoints")
def list_checkpoints():
    """
    扫描 checkpoints/ 目录，返回每个子文件夹的解析信息，
    并附带该文件夹下可用的模型文件（.pth / .onnx）。
    """
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []

    results = []
    for folder in sorted(os.listdir(CHECKPOINTS_DIR)):
        folder_path = os.path.join(CHECKPOINTS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        info = parse_checkpoint_name(folder)

        # 检查可用文件
        info["has_pth"]  = os.path.isfile(os.path.join(folder_path, 'checkpoint.pth'))
        info["has_onnx"] = os.path.isfile(os.path.join(folder_path, 'model.onnx'))

        results.append(info)

    return results
# =====================

