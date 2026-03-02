# ===================== 训练路由 =====================
# 负责模型训练的启动、状态查询和日志增量拉取。
# 全局状态 training_status / train_logs 仅在本模块内维护。

import os
import subprocess
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/api/train", tags=["train"])

# ── 路径常量（由 main.py 在启动时注入）──
MODEL_SRC_DIR: str = ''


# ── 训练请求体 ──
class TrainRequest(BaseModel):
    # 基本参数
    model:         str   = 'DLinear'
    model_id:      str   = 'ETTh1_336_96'
    data:          str   = 'ETTh1'
    data_path:     str   = 'ETTh1.csv'
    features:      str   = 'M'
    seq_len:       int   = 96
    label_len:     int   = 48
    pred_len:      int   = 96
    train_epochs:  int   = 50
    patience:      int   = 10
    batch_size:    int   = 64
    learning_rate: float = 0.005
    use_gpu:       bool  = True
    # Transformer 系列参数
    enc_in:        Optional[int]   = None
    dec_in:        Optional[int]   = None
    c_out:         Optional[int]   = None
    d_model:       Optional[int]   = None
    n_heads:       Optional[int]   = None
    e_layers:      Optional[int]   = None
    d_layers:      Optional[int]   = None
    d_ff:          Optional[int]   = None
    factor:        Optional[int]   = None
    dropout:       Optional[float] = None
    embed:         Optional[str]   = None
    activation:    Optional[str]   = None
    moving_avg:    Optional[int]   = None
    # PatchTST 专有参数
    fc_dropout:    Optional[float] = None
    head_dropout:  Optional[float] = None
    patch_len:     Optional[int]   = None
    stride:        Optional[int]   = None
    padding_patch: Optional[str]   = None
    revin:         Optional[int]   = None
    affine:        Optional[int]   = None
    subtract_last: Optional[int]   = None
    decomposition: Optional[int]   = None
    kernel_size:   Optional[int]   = None
    individual:    Optional[int]   = None


# ── 模块级可变状态 ──
training_status: dict = {
    "status":    "idle",   # idle | running | completed | failed
    "message":   "No training in progress.",
    "start_time": None,
    "conda_env": "SouthElect_stage2",
}
train_logs: list[str] = []


# ── 后台任务 ──
def run_training_task(config: dict) -> None:
    global training_status, train_logs
    try:
        training_status["status"]     = "running"
        training_status["start_time"] = datetime.now().isoformat()
        training_status["message"]    = "Training in progress..."
        train_logs.clear()
        train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started...")

        script_path = os.path.join(MODEL_SRC_DIR, 'run_longExp.py')
        env_name    = training_status["conda_env"]

        # bool 值转为字符串，其余直接拼接
        args_str = ' '.join(
            f'--{k} {str(v)}' if isinstance(v, bool) else f'--{k} {v}'
            for k, v in config.items()
        )

        proc = subprocess.Popen(
            f'conda activate {env_name} && python -u "{script_path}" {args_str}',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # stderr 合并到 stdout，统一日志流
            encoding='utf-8',
            errors='replace',           # 无法解码的字符替换为 ?
            bufsize=1,                  # 行缓冲（需配合 -u 使用）
            cwd=MODEL_SRC_DIR,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'},
        )

        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip('\n')
            if line:
                train_logs.append(line)

        proc.wait()

        if proc.returncode == 0:
            training_status["status"]  = "completed"
            training_status["message"] = "Training completed successfully."
            train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Training completed successfully.")
        else:
            last_err = train_logs[-1] if train_logs else 'unknown error'
            training_status["status"]  = "failed"
            training_status["message"] = f"Training failed. Last output: {last_err[:200]}"
            train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Training failed (exit code {proc.returncode}).")

    except Exception as e:
        training_status["status"]  = "failed"
        training_status["message"] = f"Training failed with exception: {str(e)}"
        train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Exception: {str(e)}")


# ── 路由 ──

@router.post("/start")
async def start_training(background_tasks: BackgroundTasks, req: TrainRequest = TrainRequest()):
    """启动训练任务（后台异步执行，立即返回）。"""
    global training_status
    if training_status["status"] == "running":
        return {"success": False, "message": "Training is already in progress."}
    config = {k: v for k, v in req.model_dump().items() if v is not None}
    background_tasks.add_task(run_training_task, config)
    return {"success": True, "message": "Training task has been started."}


@router.get("/status")
def get_training_status():
    """返回当前训练状态。"""
    return training_status


@router.get("/logs")
def get_train_logs(since: int = 0):
    """增量返回训练日志行。since: 客户端已接收的行数。"""
    return {"lines": train_logs[since:], "total": len(train_logs)}
