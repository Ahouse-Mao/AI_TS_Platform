# ===================== 预测 & 推理路由 =====================
# 负责 Checkpoint 列表查询、测试集推理，以及参考脚本解析。

import os
import re

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["predict"])

# ── 路径常量（由 main.py 在启动时注入）──
MODEL_SRC_DIR:   str = ''
CHECKPOINTS_DIR: str = ''
SCRIPTS_DIR:     str = ''

# ── 已知模型列表（用于解析 Checkpoint 文件夹名）──
KNOWN_MODELS = [
    'Autoformer', 'Informer', 'Transformer', 'DLinear',
    'PatchTST', 'Linear', 'NLinear', 'FEDformer', 'Pyraformer',
]

FEATURES_DESC: dict[str, str] = {
    'M':  '多变量 → 多变量',
    'S':  '单变量 → 单变量',
    'MS': '多变量 → 单变量',
}

# ── 数据集元信息（dataset字段 → csv文件名, 时间频率, ETT类型）──
DATASET_META: dict[str, tuple[str, str, str]] = {
    'ETTh1': ('ETTh1.csv', 'h', 'h'),
    'ETTh2': ('ETTh2.csv', 'h', 'h'),
    'ETTm1': ('ETTm1.csv', 't', 'm'),
    'ETTm2': ('ETTm2.csv', 't', 'm'),
}

# ── 模型 → 脚本子文件夹映射 ──
MODEL_TO_SCRIPT_FOLDER: dict[str, str] = {
    'DLinear':  'Linear',
    'NLinear':  'Linear',
    'Linear':   'Linear',
    'PatchTST': 'PatchTST',
}


# ── 推理请求体 ──
class InferenceRequest(BaseModel):
    folder_name: str
    n_samples:   int  = 200     # 推理样本数，实际上限由测试集大小决定
    use_onnx:    bool = False   # True → ONNX Runtime；False → PyTorch PTH


# ─────────────────────────────────────────────
# 工具函数：解析 Checkpoint 文件夹名
# ─────────────────────────────────────────────

def parse_checkpoint_name(name: str) -> dict:
    """
    将 Checkpoint 文件夹名解析为结构化字典。
    命名格式：{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}
              _dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}
              _eb{embed}_dt{distil}_{des}_{exp_id}
    """
    kv_pattern = (
        r'_ft(.+?)_sl(\d+)_ll(\d+)_pl(\d+)'
        r'_dm(\d+)_nh(\d+)_el(\d+)_dl(\d+)_df(\d+)'
        r'_fc(\d+)_eb(.+?)_dt(.+?)_(.+?)_(\d+)$'
    )
    match = re.search(kv_pattern, name)
    if not match:
        return {"folder_name": name, "parse_error": True}

    prefix = name[:match.start()]
    model, model_id, dataset = None, prefix, ''
    for m in KNOWN_MODELS:
        if f'_{m}_' in prefix:
            parts    = prefix.split(f'_{m}_', 1)
            model_id = parts[0]
            dataset  = parts[1]
            model    = m
            break

    features_raw = match.group(1)
    return {
        "folder_name":   name,
        "model_id":      model_id,
        "model":         model or "Unknown",
        "dataset":       dataset,
        "features":      features_raw,
        "features_desc": FEATURES_DESC.get(features_raw, features_raw),
        "seq_len":       int(match.group(2)),
        "label_len":     int(match.group(3)),
        "pred_len":      int(match.group(4)),
        "d_model":       int(match.group(5)),
        "n_heads":       int(match.group(6)),
        "e_layers":      int(match.group(7)),
        "d_layers":      int(match.group(8)),
        "d_ff":          int(match.group(9)),
        "factor":        int(match.group(10)),
        "embed":         match.group(11),
        "distil":        match.group(12),
        "des":           match.group(13),
        "exp_id":        int(match.group(14)),
    }


# ─────────────────────────────────────────────
# Checkpoint 列表 & 最新 Checkpoint
# ─────────────────────────────────────────────

@router.get("/predict/checkpoints")
def list_checkpoints():
    """扫描 checkpoints/ 目录，返回每个子文件夹的解析信息及可用文件标志。"""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []

    results = []
    for folder in sorted(os.listdir(CHECKPOINTS_DIR)):
        folder_path = os.path.join(CHECKPOINTS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        info = parse_checkpoint_name(folder)
        info["has_pth"]        = os.path.isfile(os.path.join(folder_path, 'checkpoint.pth'))
        info["has_onnx"]       = os.path.isfile(os.path.join(folder_path, 'model.onnx'))
        info["has_pt"]         = os.path.isfile(os.path.join(folder_path, 'model.pt'))
        info["has_exportable"] = info["has_onnx"] or info["has_pt"]
        results.append(info)

    return results


@router.get("/predict/latest_checkpoint")
def get_latest_checkpoint():
    """返回最近修改时间的 Checkpoint 文件夹名称（供 AI 助手中 folder_name='latest' 使用）。"""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return {"folder_name": None}

    folders = [
        f for f in os.listdir(CHECKPOINTS_DIR)
        if os.path.isdir(os.path.join(CHECKPOINTS_DIR, f))
    ]
    if not folders:
        return {"folder_name": None}

    latest = max(folders, key=lambda f: os.path.getmtime(os.path.join(CHECKPOINTS_DIR, f)))
    return {"folder_name": latest}


# ─────────────────────────────────────────────
# 推理端点
# ─────────────────────────────────────────────

@router.post("/predict/run")
def run_inference(req: InferenceRequest):
    """
    对测试集做推理，返回指标（MSE/MAE）和可视化所需数据。
    支持三种推理后端：PTH（PyTorch 原生）/ ONNX / TorchScript（.pt 回退）。
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # ── 1. 解析 Checkpoint 元信息 ──
    info = parse_checkpoint_name(req.folder_name)
    if info.get('parse_error'):
        return {'error': f'无法解析 checkpoint 名称: {req.folder_name}'}

    ckpt_dir  = os.path.join(CHECKPOINTS_DIR, req.folder_name)
    pth_path  = os.path.join(ckpt_dir, 'checkpoint.pth')
    onnx_path = os.path.join(ckpt_dir, 'model.onnx')
    pt_path   = os.path.join(ckpt_dir, 'model.pt')

    if req.use_onnx:
        has_onnx = os.path.isfile(onnx_path)
        has_pt   = os.path.isfile(pt_path)
        if not has_onnx and not has_pt:
            return {'error': 'model.onnx 和 model.pt 均不存在，请先完成训练'}
        use_torchscript = (not has_onnx and has_pt)
    else:
        use_torchscript = False
        if not os.path.isfile(pth_path):
            return {'error': 'checkpoint.pth 不存在，请先完成训练'}

    # ── 2. 加载数据集 ──
    dataset_name = info['dataset']
    if dataset_name not in DATASET_META:
        return {'error': f'暂不支持的数据集: {dataset_name}'}

    csv_file, _, ett_type = DATASET_META[dataset_name]
    dataset_path = os.path.join(MODEL_SRC_DIR, 'dataset', csv_file)
    if not os.path.isfile(dataset_path):
        return {'error': f'数据集文件不存在: {csv_file}'}

    seq_len  = info['seq_len']
    label_len = info['label_len']
    pred_len  = info['pred_len']
    features  = info['features']

    df_raw = pd.read_csv(dataset_path)
    n_cols = len(df_raw.columns) - 1   # 排除 date 列

    if features in ('M', 'MS'):
        df_data = df_raw[df_raw.columns[1:]]
        enc_in  = n_cols
    else:
        df_data = df_raw[['OT']]
        enc_in  = 1
    c_out = 1 if features == 'MS' else enc_in

    unit     = 30 * 24 if ett_type == 'h' else 30 * 24 * 4
    border1s = [0,        12*unit - seq_len,          12*unit + 4*unit - seq_len]
    border2s = [12*unit,  12*unit + 4*unit,            12*unit + 8*unit]

    scaler = StandardScaler()
    scaler.fit(df_data.values[border1s[0]:border2s[0]])
    data      = scaler.transform(df_data.values)
    data_test = data[border1s[2]:border2s[2]]
    n_test    = len(data_test) - seq_len - pred_len + 1

    if n_test <= 0:
        return {'error': '测试集样本数不足，请检查 seq_len / pred_len 设置'}

    n_infer = min(max(1, req.n_samples), n_test)
    f_dim   = -1 if features == 'MS' else 0

    inputs_list: list[list[float]] = []
    preds_list:  list[list[float]] = []
    trues_list:  list[list[float]] = []
    active_device = ''

    # ── 推理路径 A：ONNX Runtime ──
    if req.use_onnx and not use_torchscript:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # 仅输出 Error 及以上，屏蔽冗余 CUDA 警告

        for providers in (['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']):
            try:
                sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
                active_device = sess.get_providers()[0]
                break
            except Exception:
                continue
        else:
            return {'error': 'ONNX Runtime 初始化失败'}

        input_names = [inp.name for inp in sess.get_inputs()]

        for i in range(n_infer):
            bx  = data_test[i : i + seq_len][np.newaxis].astype(np.float32)
            yw  = data_test[i + seq_len - label_len : i + seq_len + pred_len].astype(np.float32)
            by  = yw[np.newaxis]
            dec = np.concatenate(
                [by[:, :label_len, :], np.zeros((1, pred_len, by.shape[-1]), dtype=np.float32)],
                axis=1,
            )
            # 按位置依次填入：encoder_input, enc_time_mark, decoder_input, dec_time_mark
            feed: dict[str, np.ndarray] = {}
            for j, name in enumerate(input_names):
                if   j == 0: feed[name] = bx
                elif j == 1: feed[name] = np.zeros((1, bx.shape[1],  4), dtype=np.float32)
                elif j == 2: feed[name] = dec
                elif j == 3: feed[name] = np.zeros((1, dec.shape[1], 4), dtype=np.float32)

            out = sess.run(None, feed)[0]
            inputs_list.append(bx[0, :, -1].tolist())
            preds_list.append(out[0, :, -1].tolist())
            trues_list.append(by[0, -pred_len:, -1].tolist())

    # ── 推理路径 A2：TorchScript（.pt 格式回退）──
    elif req.use_onnx and use_torchscript:
        import torch

        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        active_device = str(device)
        is_linear = ('Linear' in info['model']) or ('TST' in info['model'])

        traced = torch.jit.load(pt_path, map_location=device)
        traced.eval()

        with torch.no_grad():
            for i in range(n_infer):
                bx = torch.tensor(data_test[i : i + seq_len]).float().unsqueeze(0).to(device)
                by = torch.tensor(
                    data_test[i + seq_len - label_len : i + seq_len + pred_len]
                ).float().unsqueeze(0).to(device)

                if is_linear:
                    out = traced(bx)
                else:
                    dec = torch.cat([by[:, :label_len, :], torch.zeros_like(by[:, -pred_len:, :])], dim=1)
                    out = traced(bx, None, dec, None)

                inputs_list.append(bx[0, :, -1].cpu().numpy().tolist())
                preds_list.append(out[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1].tolist())
                trues_list.append(by[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1].tolist())

    # ── 推理路径 B：PyTorch PTH ──
    else:
        import torch
        from types import SimpleNamespace
        from models import DLinear, Linear, NLinear, PatchTST, Autoformer, Informer, Transformer  # type: ignore[import]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        active_device = str(device)

        model_cls_map = {
            'DLinear':     DLinear,
            'Linear':      Linear,
            'NLinear':     NLinear,
            'PatchTST':    PatchTST,
            'Autoformer':  Autoformer,
            'Informer':    Informer,
            'Transformer': Transformer,
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
        is_linear = ('Linear' in model_name) or ('TST' in model_name)

        with torch.no_grad():
            for i in range(n_infer):
                bx = torch.tensor(data_test[i : i + seq_len]).float().unsqueeze(0).to(device)
                by = torch.tensor(
                    data_test[i + seq_len - label_len : i + seq_len + pred_len]
                ).float().unsqueeze(0).to(device)

                if is_linear:
                    out = model(bx)
                else:
                    dec = torch.cat([by[:, :label_len, :], torch.zeros_like(by[:, -pred_len:, :])], dim=1)
                    out = model(bx, None, dec, None)

                inputs_list.append(bx[0, :, -1].cpu().numpy().tolist())
                preds_list.append(out[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1].tolist())
                trues_list.append(by[:, -pred_len:, f_dim:].cpu().numpy()[0, :, -1].tolist())

    # ── 计算 MSE / MAE ──
    preds_arr = np.array(preds_list)
    trues_arr = np.array(trues_list)
    mse = float(np.mean((preds_arr - trues_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - trues_arr)))

    return {
        'folder_name':   req.folder_name,
        'backend':       'torchscript' if (req.use_onnx and use_torchscript) else ('onnx' if req.use_onnx else 'pth'),
        'active_device': active_device,
        'n_total':       n_test,
        'n_samples':     n_infer,
        'seq_len':       seq_len,
        'pred_len':      pred_len,
        'metrics': {'mse': round(mse, 6), 'mae': round(mae, 6)},
        'inputs':  inputs_list,
        'preds':   preds_list,
        'trues':   trues_list,
    }


# ─────────────────────────────────────────────
# 脚本参考 API：解析 .sh 文件回填训练参数
# ─────────────────────────────────────────────

def _resolve_sh_variables(content: str) -> dict[str, str]:
    """提取 shell 脚本顶部的变量赋值，如 seq_len=336。"""
    variables: dict[str, str] = {}
    for m in re.finditer(r'^([A-Za-z_]\w*)=([^\n#]*)', content, re.MULTILINE):
        k, v = m.group(1), m.group(2).strip().strip('"\'')
        variables[k] = v
    return variables


def _parse_script_first_block(content: str) -> dict:
    """从 .sh 文件的第一个 python run_longExp.py 调用中提取可回填到 UI 的参数。"""
    variables = _resolve_sh_variables(content)
    start = content.find('python -u run_longExp.py')
    if start == -1:
        return {}

    block = content[start:]
    redirect = re.search(r'>[ \t]*\S', block)
    if redirect:
        block = block[:redirect.start()]
    block = block.replace('\\\n', ' ')

    skipped = {'is_training', 'des', 'itr', 'root_path', 'random_seed'}
    raw: dict[str, str] = {}
    for m in re.finditer(r'--(\w+)\s+(\S+)', block):
        key, val = m.group(1), m.group(2)
        if key in skipped:
            continue
        if val.startswith('$'):
            val = variables.get(val[1:].strip("'\""), val)
        raw[key] = val

    int_fields   = {'seq_len', 'pred_len', 'label_len', 'enc_in', 'dec_in', 'c_out',
                    'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'factor',
                    'batch_size', 'train_epochs', 'patch_len', 'stride', 'moving_avg',
                    'kernel_size', 'revin', 'affine', 'individual', 'subtract_last', 'decomposition'}
    float_fields = {'learning_rate', 'dropout', 'fc_dropout', 'head_dropout'}

    result: dict = {}
    for k, v in raw.items():
        if k in int_fields:
            try:    result[k] = int(float(v))
            except: pass            # noqa: E722
        elif k in float_fields:
            try:    result[k] = float(v)
            except: pass            # noqa: E722
        else:
            result[k] = v

    if 'data_path' not in result and 'data_path_name' in variables:
        result['data_path'] = variables['data_path_name']
    return result


@router.get('/scripts/list')
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


@router.get('/scripts/params')
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
