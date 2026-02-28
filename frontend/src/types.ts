// ===================== 全局共用类型定义 =====================
// 放在这里的类型会被多个页面共享，避免重复定义

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
