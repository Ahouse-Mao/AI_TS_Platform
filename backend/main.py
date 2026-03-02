from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import os
import sys
import subprocess
import re
import json
from typing import Optional, List

# ---- model_src 加入 Python 搜索路径（用于推理时导入模型/数据集） ----
MODEL_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_src')
if MODEL_SRC_DIR not in sys.path:
    sys.path.insert(0, MODEL_SRC_DIR)

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

# ===== 训练请求体 =====
class TrainRequest(BaseModel):
    # 基本参数
    model:          str   = 'DLinear'
    model_id:       str   = 'ETTh1_336_96'
    data:           str   = 'ETTh1'
    data_path:      str   = 'ETTh1.csv'
    features:       str   = 'M'
    seq_len:        int   = 96
    label_len:      int   = 48
    pred_len:       int   = 96
    train_epochs:   int   = 50
    patience:       int   = 10
    batch_size:     int   = 64
    learning_rate:  float = 0.005
    use_gpu:        bool  = True
    # Transformer 系列参数
    enc_in:         Optional[int]   = None
    dec_in:         Optional[int]   = None
    c_out:          Optional[int]   = None
    d_model:        Optional[int]   = None
    n_heads:        Optional[int]   = None
    e_layers:       Optional[int]   = None
    d_layers:       Optional[int]   = None
    d_ff:           Optional[int]   = None
    factor:         Optional[int]   = None
    dropout:        Optional[float] = None
    embed:          Optional[str]   = None
    activation:     Optional[str]   = None
    moving_avg:     Optional[int]   = None
    # PatchTST 专有参数
    fc_dropout:     Optional[float] = None
    head_dropout:   Optional[float] = None
    patch_len:      Optional[int]   = None
    stride:         Optional[int]   = None
    padding_patch:  Optional[str]   = None
    revin:          Optional[int]   = None
    affine:         Optional[int]   = None
    subtract_last:  Optional[int]   = None
    decomposition:  Optional[int]   = None
    kernel_size:    Optional[int]   = None
    individual:     Optional[int]   = None
# ======================


# ===== 存储训练状态 =====
trainning_status = {
    "status": "idle",  # idle, running, completed, failed
    "message": "No training in progress.",
    "start_time": None,
    "environment": 'SouthElect_stage2',
}
# =====================

# ===== 训练日志缓冲区 =====
# 每次启动新训练时清空，前端通过 /api/train/logs?since=N 增量拉取
train_logs: list[str] = []
# =====================

# ===== 后台任务函数 =====
def run_training_task(config: dict):
    global trainning_status, train_logs
    try:
        trainning_status["status"] = "running"
        trainning_status["start_time"] = datetime.now().isoformat()
        trainning_status["message"] = "Training in progress..."
        train_logs.clear()  # 每次新训练前清空日志
        train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started...")

        script_path = os.path.join(os.path.dirname(__file__), 'model_src', 'run_longExp.py')
        env_name = trainning_status["environment"]

        # 将配置字典转为 CLI 参数，bool 值转为 True/False 字符串
        args_parts = []
        for key, value in config.items():
            if isinstance(value, bool):
                args_parts.append(f'--{key} {str(value)}')
            else:
                args_parts.append(f'--{key} {value}')
        args_str = ' '.join(args_parts)

        # 使用 Popen 逐行实时读取输出，写入日志缓冲区
        # encoding + errors 防止 Windows 上 GBK/UTF-8 混合编码导致读取崩溃
        proc = subprocess.Popen(
            f'conda activate {env_name} && python -u "{script_path}" {args_str}',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
            encoding='utf-8',
            errors='replace',          # 无法解码的字符替换为 ?，防止异常
            bufsize=1,                 # 行缓冲（需配合 -u 使用）
            cwd=os.path.join(os.path.dirname(__file__), "model_src"),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        )

        # 逐行读取并追加到日志缓冲区
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip('\n')
            if line:
                train_logs.append(line)

        proc.wait()  # 等待进程结束，获取返回码

        if proc.returncode == 0:
            trainning_status["status"] = "completed"
            trainning_status["message"] = "Training completed successfully."
            train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Training completed successfully.")
        else:
            trainning_status["status"] = "failed"
            last_err = train_logs[-1] if train_logs else 'unknown error'
            trainning_status["message"] = f"Training failed. Last output: {last_err[:200]}"
            train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Training failed (exit code {proc.returncode}).")
    except Exception as e:
        trainning_status["status"] = "failed"
        trainning_status["message"] = f"Training failed with exception: {str(e)}"
        train_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Exception: {str(e)}")
# =====================

# ===== API端点: 启动训练 =====

@app.post("/api/train/start")
async def start_training(background_tasks: BackgroundTasks, req: TrainRequest = TrainRequest()):
# ↑ async 表示这是异步函数，FastAPI 能同时处理多个请求而不互相等待
# ↑ background_tasks 是 FastAPI 内置的特殊参数
#   只要你在参数里写上它，FastAPI 会自动帮你注入一个 BackgroundTasks 对象
    global trainning_status
    if trainning_status["status"] == "running":
        return {"success": False, "message": "Training is already in progress, do not start a new one."}
        # 返回字典, fastapi自动序列化为json返回给前端
    # 将配置转为字典（移除 None 由脚本自身默言处理）
    config = {k: v for k, v in req.model_dump().items() if v is not None}
    background_tasks.add_task(run_training_task, config)
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


@app.get("/api/train/logs")
def get_train_logs(since: int = 0):
    """
    增量返回训练日志行。
    since: 客户端已接收的行数，返回 since 之后的新行，避免重复传输。
    """
    return {
        "lines": train_logs[since:],
        "total": len(train_logs),
    }


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
        info["has_pt"]   = os.path.isfile(os.path.join(folder_path, 'model.pt'))
        # 可用于快速推理的格式（onnx 或 torchscript 均可）
        info["has_exportable"] = info["has_onnx"] or info["has_pt"]

        results.append(info)

    return results
# =====================


# ===== 请求体模型 =====
class InferenceRequest(BaseModel):
    folder_name: str
    n_samples:   int  = 200    # 推理样本数，默认 200，实际上限由 n_test 决定
    use_onnx:    bool = False  # True 使用 ONNX Runtime，False 使用 PyTorch PTH
# =====================


# ===== API端点: 对选中 Checkpoint 做测试集推理 =====

# 数据集文件映射表（dataset字段 → csv文件名, 时间频率, ETTh/ETTm类型）
DATASET_META = {
    'ETTh1': ('ETTh1.csv', 'h', 'h'),
    'ETTh2': ('ETTh2.csv', 'h', 'h'),
    'ETTm1': ('ETTm1.csv', 't', 'm'),
    'ETTm2': ('ETTm2.csv', 't', 'm'),
}

@app.post("/api/predict/run")
def run_inference(req: InferenceRequest):
    """
    对测试集做推理，返回指标 + 可视化数据。
    支持两种推理后端：
      - PTH：PyTorch 原生，自动选择 CUDA / CPU
      - ONNX：ONNX Runtime，自动选择 CUDA / CPU
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # ---- 1. 解析 checkpoint 文件夹名 ----
    folder_name = req.folder_name
    info = parse_checkpoint_name(folder_name)
    if info.get('parse_error'):
        return {'error': f'无法解析 checkpoint 名称: {folder_name}'}

    ckpt_dir = os.path.join(CHECKPOINTS_DIR, folder_name)
    pth_path  = os.path.join(ckpt_dir, 'checkpoint.pth')
    onnx_path = os.path.join(ckpt_dir, 'model.onnx')
    pt_path   = os.path.join(ckpt_dir, 'model.pt')

    if req.use_onnx:
        has_onnx = os.path.isfile(onnx_path)
        has_pt   = os.path.isfile(pt_path)
        if not has_onnx and not has_pt:
            return {'error': 'model.onnx 和 model.pt 均不存在，请先完成训练'}
        # 优先用 ONNX，如果不存在则回退到 TorchScript
        use_torchscript = (not has_onnx and has_pt)
    else:
        use_torchscript = False
        if not os.path.isfile(pth_path):
            return {'error': 'checkpoint.pth 不存在，请先完成训练'}

    # ---- 2. 数据集信息 ----
    dataset_name = info['dataset']
    if dataset_name not in DATASET_META:
        return {'error': f'暂不支持的数据集: {dataset_name}'}

    csv_file, freq, ett_type = DATASET_META[dataset_name]
    dataset_path = os.path.join(MODEL_SRC_DIR, 'dataset', csv_file)
    if not os.path.isfile(dataset_path):
        return {'error': f'数据集文件不存在: {csv_file}'}

    seq_len   = info['seq_len']
    label_len = info['label_len']
    pred_len  = info['pred_len']
    features  = info['features']   # 'M' | 'S' | 'MS'

    # ---- 3. 读取 CSV 并按 DataLoader 相同切分逻辑分出测试集 ----
    df_raw = pd.read_csv(dataset_path)
    n_cols = len(df_raw.columns) - 1   # 排除 date 列

    if features in ('M', 'MS'):
        df_data = df_raw[df_raw.columns[1:]]
        enc_in  = n_cols
    else:  # S
        df_data = df_raw[['OT']]
        enc_in  = 1
    c_out = 1 if features == 'MS' else enc_in

    unit = 30 * 24 if ett_type == 'h' else 30 * 24 * 4
    border1s = [0,        12*unit - seq_len,          12*unit + 4*unit - seq_len]
    border2s = [12*unit,  12*unit + 4*unit,            12*unit + 8*unit]

    scaler = StandardScaler()
    scaler.fit(df_data.values[border1s[0]:border2s[0]])
    data = scaler.transform(df_data.values)

    data_test = data[border1s[2]:border2s[2]]
    n_test = len(data_test) - seq_len - pred_len + 1
    if n_test <= 0:
        return {'error': '测试集样本数不足，请检查 seq_len / pred_len 设置'}

    n_infer = min(max(1, req.n_samples), n_test)
    f_dim   = -1 if features == 'MS' else 0

    inputs_list: list[list[float]] = []
    preds_list:  list[list[float]] = []
    trues_list:  list[list[float]] = []
    active_device = ''

    # ==============================================================
    # 推理路径 A：ONNX Runtime
    # ==============================================================
    if req.use_onnx and not use_torchscript:
        import onnxruntime as ort

        # 屏蔽 CUDA Provider 加载失败的冗余警告（Error 126 / cuDNN 版本不匹配）
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # 0=Verbose 1=Info 2=Warning 3=Error 4=Fatal

        # GPU → CPU 自动回退
        for providers in (['CUDAExecutionProvider', 'CPUExecutionProvider'],
                          ['CPUExecutionProvider']):
            try:
                sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
                active_device = sess.get_providers()[0]  # 实际生效的 provider
                break
            except Exception:
                continue
        else:
            return {'error': 'ONNX Runtime 初始化失败'}

        # 获取模型输入名称列表（由导出时确定）
        input_names = [inp.name for inp in sess.get_inputs()]

        for i in range(n_infer):
            bx = data_test[i : i + seq_len][np.newaxis].astype(np.float32)          # (1, seq_len, ch)
            yw = data_test[i + seq_len - label_len : i + seq_len + pred_len].astype(np.float32)
            by = yw[np.newaxis]                                                       # (1, label+pred, ch)

            # 构造 decoder 输入：将预测部分置零
            dec_inp = np.concatenate(
                [by[:, :label_len, :],
                 np.zeros((1, pred_len, by.shape[-1]), dtype=np.float32)], axis=1
            )

            # 按位置顺序将输入填入 feed dict
            # 常见导出模式：[batch_x] 或 [batch_x, x_mark_enc, batch_y, x_mark_dec]
            feed: dict[str, np.ndarray] = {}
            placeholder_shape = lambda t: (1, t.shape[1], 4)  # 时间特征维度占位符
            for j, name in enumerate(input_names):
                if j == 0:
                    feed[name] = bx
                elif j == 1:
                    feed[name] = np.zeros(placeholder_shape(bx), dtype=np.float32)
                elif j == 2:
                    feed[name] = dec_inp
                elif j == 3:
                    feed[name] = np.zeros(placeholder_shape(dec_inp), dtype=np.float32)

            out = sess.run(None, feed)[0]                    # (1, pred_len, c_out) or (1, pred_len, 1)
            pred_vec  = out[0, :, -1]                        # (pred_len,)
            true_vec  = by[0, -pred_len:, -1]
            input_vec = bx[0, :, -1]

            inputs_list.append(input_vec.tolist())
            preds_list.append(pred_vec.tolist())
            trues_list.append(true_vec.tolist())

    # ==============================================================
    # 推理路径 A2：TorchScript（.pt格式回退）
    # ==============================================================
    elif req.use_onnx and use_torchscript:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        active_device = str(device)
        is_linear_type = ('Linear' in info['model']) or ('TST' in info['model'])

        traced_model = torch.jit.load(pt_path, map_location=device)
        traced_model.eval()
        print(f'[INFO] 使用 TorchScript 模型进行推理: {pt_path}')

        with torch.no_grad():
            for i in range(n_infer):
                batch_x = torch.tensor(data_test[i : i + seq_len]).float().unsqueeze(0).to(device)
                y_window = data_test[i + seq_len - label_len : i + seq_len + pred_len]
                batch_y  = torch.tensor(y_window).float().unsqueeze(0).to(device)

                if is_linear_type:
                    output = traced_model(batch_x)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
                    output  = traced_model(batch_x, None, dec_inp, None)

                pred_vec  = output[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1]
                true_vec  = batch_y[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1]
                input_vec = batch_x[0, :, -1].cpu().numpy()

                inputs_list.append(input_vec.tolist())
                preds_list.append(pred_vec.tolist())
                trues_list.append(true_vec.tolist())

    # ==============================================================
    # 推理路径 B：PyTorch PTH
    # ==============================================================
    else:
        import torch
        from types import SimpleNamespace
        from models import DLinear, Linear, NLinear, PatchTST, Autoformer, Informer, Transformer  # type: ignore[import]

        # GPU → CPU 自动回退
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        active_device = str(device)

        model_cls_map = {
            'DLinear': DLinear, 'Linear': Linear, 'NLinear': NLinear,
            'PatchTST': PatchTST, 'Autoformer': Autoformer,
            'Informer': Informer, 'Transformer': Transformer,
        }
        model_name = info['model']
        if model_name not in model_cls_map:
            return {'error': f'暂不支持的模型: {model_name}'}

        args = SimpleNamespace(
            model=model_name,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            features=features,
            enc_in=enc_in, dec_in=enc_in, c_out=c_out,
            d_model=info['d_model'],   n_heads=info['n_heads'],
            e_layers=info['e_layers'], d_layers=info['d_layers'],
            d_ff=info['d_ff'],         factor=info['factor'],
            embed=info['embed'],
            distil=(str(info['distil']).lower() == 'true'),
            dropout=0.05, activation='gelu',
            output_attention=False, use_amp=False,
            fc_dropout=0.05, head_dropout=0.0,
            patch_len=16, stride=8, padding_patch='end',
            revin=1, affine=0, subtract_last=0,
            decomposition=0, kernel_size=25, individual=0,
            embed_type=0, moving_avg=25,
        )

        model = model_cls_map[model_name].Model(args).float().to(device)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.eval()

        is_linear_type = ('Linear' in model_name) or ('TST' in model_name)

        with torch.no_grad():
            for i in range(n_infer):
                batch_x = torch.tensor(data_test[i : i + seq_len]).float().unsqueeze(0).to(device)
                y_window = data_test[i + seq_len - label_len : i + seq_len + pred_len]
                batch_y  = torch.tensor(y_window).float().unsqueeze(0).to(device)

                if is_linear_type:
                    output = model(batch_x)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
                    output  = model(batch_x, None, dec_inp, None)

                pred_vec  = output[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1]
                true_vec  = batch_y[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1]
                input_vec = batch_x[0, :, -1].cpu().numpy()

                inputs_list.append(input_vec.tolist())
                preds_list.append(pred_vec.tolist())
                trues_list.append(true_vec.tolist())

    # ---- 计算 MSE / MAE ----
    preds_arr = np.array(preds_list)
    trues_arr = np.array(trues_list)
    mse = float(np.mean((preds_arr - trues_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - trues_arr)))

    return {
        'folder_name':   folder_name,
        'backend':       'torchscript' if (req.use_onnx and use_torchscript) else ('onnx' if req.use_onnx else 'pth'),
        'active_device': active_device,   # 实际运行设备
        'n_total':   n_test,
        'n_samples': n_infer,
        'seq_len':   seq_len,
        'pred_len':  pred_len,
        'metrics': {
            'mse': round(mse, 6),
            'mae': round(mae, 6),
        },
        'inputs': inputs_list,
        'preds':  preds_list,
        'trues':  trues_list,
    }
# =====================


# ===========================
# 脚本参考 API
# ===========================
SCRIPTS_DIR = os.path.join(MODEL_SRC_DIR, 'scripts')

# 模型 → 脚本子文件夹映射（不在表中的模型返回空列表）
MODEL_TO_SCRIPT_FOLDER: dict[str, str] = {
    'DLinear':  'Linear',
    'NLinear':  'Linear',
    'Linear':   'Linear',
    'PatchTST': 'PatchTST',
}

def _resolve_sh_variables(content: str) -> dict[str, str]:
    """提取 shell 脚本顶部的变量赋值，如 seq_len=336。"""
    variables: dict[str, str] = {}
    for m in re.finditer(r'^([A-Za-z_]\w*)=([^\n#]*)', content, re.MULTILINE):
        k, v = m.group(1), m.group(2).strip().strip('"\'')
        variables[k] = v
    return variables

def _parse_script_first_block(content: str) -> dict:
    """从 .sh 文件的第一个 python run_longExp.py 调用中提取参数。"""
    variables = _resolve_sh_variables(content)

    start = content.find('python -u run_longExp.py')
    if start == -1:
        return {}
    block = content[start:]

    # 截到重定向符号（该调用结束）
    redirect = re.search(r'>[ \t]*\S', block)
    if redirect:
        block = block[:redirect.start()]

    # 合并续行
    block = block.replace('\\\n', ' ')

    # 跳过不需要回填到 UI 的参数
    skipped = {'is_training', 'des', 'itr', 'root_path', 'random_seed'}
    raw: dict[str, str] = {}
    for m in re.finditer(r'--(\w+)\s+(\S+)', block):
        key, val = m.group(1), m.group(2)
        if key in skipped:
            continue
        if val.startswith('$'):
            var_name = val[1:].strip("'\"")
            val = variables.get(var_name, val)
        raw[key] = val

    # 字段类型转换
    int_fields   = {'seq_len', 'pred_len', 'label_len', 'enc_in', 'dec_in', 'c_out',
                    'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'factor',
                    'batch_size', 'train_epochs', 'patch_len', 'stride', 'moving_avg',
                    'kernel_size', 'revin', 'affine', 'individual', 'subtract_last', 'decomposition'}
    float_fields = {'learning_rate', 'dropout', 'fc_dropout', 'head_dropout'}

    result: dict = {}
    for k, v in raw.items():
        if k in int_fields:
            try:    result[k] = int(float(v))
            except: pass
        elif k in float_fields:
            try:    result[k] = float(v)
            except: pass
        else:
            result[k] = v

    # data_path 补充（PatchTST 脚本用变量名存储）
    if 'data_path' not in result and 'data_path_name' in variables:
        result['data_path'] = variables['data_path_name']

    return result


@app.get('/api/scripts/list')
def get_script_list(model: str = ''):
    """返回指定模型的可用参考脚本名列表（不含 .sh 后缀）。"""
    folder_name = MODEL_TO_SCRIPT_FOLDER.get(model)
    if not folder_name:
        return {'scripts': []}
    folder_path = os.path.join(SCRIPTS_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return {'scripts': []}
    scripts = sorted(
        f[:-3] for f in os.listdir(folder_path)
        if f.endswith('.sh') and os.path.isfile(os.path.join(folder_path, f))
    )
    return {'scripts': scripts}


@app.get('/api/scripts/params')
def get_script_params(model: str = '', script: str = ''):
    """解析指定脚本的第一个训练调用，返回可回填 UI 的参数字典。"""
    folder_name = MODEL_TO_SCRIPT_FOLDER.get(model)
    if not folder_name or not script:
        return {'params': {}}
    sh_path = os.path.join(SCRIPTS_DIR, folder_name, script + '.sh')
    if not os.path.isfile(sh_path):
        return {'params': {}}
    with open(sh_path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    return {'params': _parse_script_first_block(content)}
# ===========================


# ===========================
# AI 助手 —— 流式聊天代理
# ===========================
class _AssistantMsg(BaseModel):
    role:    str
    content: str

class AssistantChatRequest(BaseModel):
    messages: List[_AssistantMsg]
    model:    str = 'gpt-4o'
    api_key:  str = ''
    base_url: str = 'https://api.openai.com/v1'


@app.get('/api/assistant/models')
def get_assistant_models(api_key: str = '', base_url: str = 'https://api.openai.com/v1'):
    """拉取指定 API 可用的模型列表。"""
    if not api_key.strip():
        return {'models': [], 'error': '请先配置 API Key'}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        page   = client.models.list()
        ids    = sorted(m.id for m in page.data)
        return {'models': ids}
    except Exception as e:
        return {'models': [], 'error': str(e)}


@app.post('/api/assistant/chat')
async def assistant_chat(req: AssistantChatRequest):
    """流式代理到 OpenAI 兼容 API，以 SSE 形式返回给前端。"""

    # 缺少 API Key
    if not req.api_key.strip():
        async def _no_key():
            yield f"data: {json.dumps({'error': '请先在设置中配置 API Key'})}\n\n"
        return StreamingResponse(_no_key(), media_type='text/event-stream')

    # 动态导入 openai
    try:
        from openai import OpenAI
    except ImportError:
        async def _no_lib():
            yield f"data: {json.dumps({'error': 'openai 库未安装，请运行 pip install openai'})}\n\n"
        return StreamingResponse(_no_lib(), media_type='text/event-stream')

    client = OpenAI(api_key=req.api_key, base_url=req.base_url)

    def _stream():
        try:
            resp = client.chat.completions.create(
                model=req.model,
                messages=[{'role': m.role, 'content': m.content} for m in req.messages],
                stream=True,
            )
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield 'data: [DONE]\n\n'
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_stream(), media_type='text/event-stream')
# ===========================
