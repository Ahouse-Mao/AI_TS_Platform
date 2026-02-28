from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import sys
import subprocess
import re
from typing import Optional

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

    if req.use_onnx:
        if not os.path.isfile(onnx_path):
            return {'error': 'model.onnx 不存在，请先导出 ONNX 模型'}
    else:
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
    if req.use_onnx:
        import onnxruntime as ort

        # GPU → CPU 自动回退
        for providers in (['CUDAExecutionProvider', 'CPUExecutionProvider'],
                          ['CPUExecutionProvider']):
            try:
                sess = ort.InferenceSession(onnx_path, providers=providers)
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
        'backend':       'onnx' if req.use_onnx else 'pth',
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
