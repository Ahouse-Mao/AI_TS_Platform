// ===================== 全局共用类型定义 =====================
// 放在这里的类型会被多个页面共享，避免重复定义

// 训练配置参数（对应 run_longExp.py 的所有 argparse 参数）
export interface TrainConfig {
  // ── 基本参数 ──
  model:          string
  model_id:       string
  data:           string
  data_path:      string
  features:       'M' | 'MS' | 'S'
  seq_len:        number
  label_len:      number
  pred_len:       number
  train_epochs:   number
  patience:       number
  batch_size:     number
  learning_rate:  number
  use_gpu:        boolean
  // ── Transformer 系列参数 ──
  enc_in?:        number
  dec_in?:        number
  c_out?:         number
  d_model?:       number
  n_heads?:       number
  e_layers?:      number
  d_layers?:      number
  d_ff?:          number
  factor?:        number
  dropout?:       number
  embed?:         string
  activation?:    string
  moving_avg?:    number
  // ── PatchTST 专有参数 ──
  fc_dropout?:    number
  head_dropout?:  number
  patch_len?:     number
  stride?:        number
  padding_patch?: string
  revin?:         number
  affine?:        number
  subtract_last?: number
  decomposition?: number
  kernel_size?:   number
  individual?:    number
}

// 训练状态
export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  message: string
  start_time: string | null
  conda_env: string
}

// Checkpoint 信息（从文件夹名解析而来）
export interface CheckpointInfo {
  folder_name:    string
  parse_error?:   boolean
  model_id?:      string
  model?:         string
  dataset?:       string
  features?:      string
  features_desc?: string
  seq_len?:       number
  label_len?:     number
  pred_len?:      number
  d_model?:       number
  n_heads?:       number
  e_layers?:      number
  d_layers?:      number
  d_ff?:          number
  factor?:        number
  embed?:         string
  distil?:        string
  des?:           string
  exp_id?:        number
  has_pth?:       boolean
  has_onnx?:      boolean
}
