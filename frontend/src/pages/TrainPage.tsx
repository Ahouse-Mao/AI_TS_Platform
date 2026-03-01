import { useState, useEffect, useRef } from 'react'
import type { TrainingStatus, TrainConfig } from '../types'

const API = 'http://localhost:8000'

// ===================== 模型定义 =====================
type ModelFamily = 'linear' | 'transformer' | 'patchtst' | 'stat'

interface ModelDef {
  name:   string
  family: ModelFamily
  desc:   string
}

const MODELS: ModelDef[] = [
  { name: 'DLinear',     family: 'linear',      desc: '分解线性模型，趋势+季节性双线性' },
  { name: 'NLinear',     family: 'linear',      desc: '归一化线性，应对分布偏移' },
  { name: 'Linear',      family: 'linear',      desc: '单层全连接线性基线' },
  { name: 'PatchTST',    family: 'patchtst',    desc: 'Patch 分段自注意力 Transformer' },
  { name: 'Autoformer',  family: 'transformer', desc: '自相关机制替代注意力' },
  { name: 'Informer',    family: 'transformer', desc: 'ProbSparse 稀疏注意力长序列' },
  { name: 'Transformer', family: 'transformer', desc: '标准 Transformer 基线模型' },
  { name: 'Stat_models', family: 'stat',        desc: '经典统计模型（ARIMA 等）' },
]

const FAMILY_STYLE: Record<ModelFamily, { border: string; badge: string; badgeText: string; tag: string }> = {
  linear:      { border: '#2d9e6b', badge: '#1a4a36', badgeText: '#6fcfa0', tag: 'Linear'      },
  patchtst:    { border: '#7b5ea7', badge: '#352058', badgeText: '#c0a0f0', tag: 'PatchTST'    },
  transformer: { border: '#4a6fa5', badge: '#1c3158', badgeText: '#90b4e8', tag: 'Transformer' },
  stat:        { border: '#666',    badge: '#2a2a2a', badgeText: '#aaa',    tag: 'Statistical' },
}

// ===================== 数据集映射 =====================
const DATASET_OPTIONS = [
  { data: 'ETTh1', data_path: 'ETTh1.csv' },
  { data: 'ETTh2', data_path: 'ETTh2.csv' },
  { data: 'ETTm1', data_path: 'ETTm1.csv' },
  { data: 'ETTm2', data_path: 'ETTm2.csv' },
]

// ===================== 默认值（来自 run_longExp.py）=====================
const DEFAULT_BASIC = {
  model_id:      'ETTh1_336_96',
  data:          'ETTh1',
  data_path:     'ETTh1.csv',
  features:      'M' as 'M' | 'MS' | 'S',
  seq_len:       96,
  label_len:     48,
  pred_len:      96,
  train_epochs:  50,
  patience:      10,
  batch_size:    64,
  learning_rate: 0.005,
  use_gpu:       true,
}

const DEFAULT_TRANSFORMER = {
  enc_in:     7,
  dec_in:     7,
  c_out:      7,
  d_model:    512,
  n_heads:    8,
  e_layers:   2,
  d_layers:   1,
  d_ff:       2048,
  factor:     1,
  dropout:    0.05,
  embed:      'timeF',
  activation: 'gelu',
  moving_avg: 25,
}

const DEFAULT_PATCHTST = {
  enc_in:        7,
  fc_dropout:    0.05,
  head_dropout:  0.0,
  patch_len:     16,
  stride:        8,
  padding_patch: 'end',
  revin:         1,
  affine:        0,
  subtract_last: 0,
  decomposition: 0,
  kernel_size:   25,
  individual:    0,
}

const DEFAULT_LINEAR = {
  individual: 0,
}

// ===================== Props 类型 =====================
// TrainPage 的 props：状态和回调从父组件 App 传入
export interface TrainPageProps {
  trainingStatus: TrainingStatus | null
  isLoading:      boolean
  onStartTraining:(config: TrainConfig) => void
  trainLogs:      string[]
  onClearLogs:    () => void
}

// ===================== 主组件 =====================
export function TrainPage({ trainingStatus, isLoading, onStartTraining, trainLogs, onClearLogs }: TrainPageProps) {
  const [selectedModel,     setSelectedModel]     = useState<string>('DLinear')
  const [basicParams,       setBasicParams]       = useState({ ...DEFAULT_BASIC })
  const [transformerParams, setTransformerParams] = useState({ ...DEFAULT_TRANSFORMER })
  const [patchTSTParams,    setPatchTSTParams]    = useState({ ...DEFAULT_PATCHTST })
  const [linearParams,      setLinearParams]      = useState({ ...DEFAULT_LINEAR })

  // 日志控制台（数据来自 App 全局状态，本地仅保留滚动锚点）
  const logEndRef = useRef<HTMLDivElement>(null)

  // 参考脚本
  const [scriptList,     setScriptList]     = useState<string[]>([])
  const [selectedScript, setSelectedScript] = useState<string>('')
  const [scriptLoading,  setScriptLoading]  = useState(false)

  const currentFamily = MODELS.find(m => m.name === selectedModel)?.family ?? 'linear'

  const resetBasic       = () => setBasicParams({ ...DEFAULT_BASIC })
  const resetModelParams = () => {
    if (currentFamily === 'transformer') setTransformerParams({ ...DEFAULT_TRANSFORMER })
    if (currentFamily === 'patchtst')    setPatchTSTParams({ ...DEFAULT_PATCHTST })
    if (currentFamily === 'linear')      setLinearParams({ ...DEFAULT_LINEAR })
  }

  // 模型切换时：重置脚本选择并重新拉取脚本列表
  useEffect(() => {
    setSelectedScript('')
    fetch(`${API}/api/scripts/list?model=${selectedModel}`)
      .then(r => r.json())
      .then(d => setScriptList(d.scripts ?? []))
      .catch(() => setScriptList([]))
  }, [selectedModel])

  // 选择脚本时：拉取参数并回填
  async function handleScriptChange(script: string) {
    setSelectedScript(script)
    if (!script) return
    setScriptLoading(true)
    try {
      const res  = await fetch(`${API}/api/scripts/params?model=${selectedModel}&script=${script}`)
      const data = await res.json() as { params: Record<string, unknown> }
      const p    = data.params
      if (!p || Object.keys(p).length === 0) return

      // 回填基本参数
      setBasicParams(prev => ({
        ...prev,
        ...(typeof p.seq_len       === 'number' ? { seq_len:       p.seq_len       } : {}),
        ...(typeof p.pred_len      === 'number' ? { pred_len:      p.pred_len      } : {}),
        ...(typeof p.label_len     === 'number' ? { label_len:     p.label_len     } : {}),
        ...(typeof p.batch_size    === 'number' ? { batch_size:    p.batch_size    } : {}),
        ...(typeof p.learning_rate === 'number' ? { learning_rate: p.learning_rate } : {}),
        ...(typeof p.train_epochs  === 'number' ? { train_epochs:  p.train_epochs  } : {}),
        ...(typeof p.features      === 'string' && ['M','MS','S'].includes(p.features as string)
              ? { features: p.features as 'M'|'MS'|'S' } : {}),
        ...(typeof p.data          === 'string' ? { data:      p.data as string      } : {}),
        ...(typeof p.data_path     === 'string' ? { data_path: p.data_path as string } : {}),
      }))

      // 回填 Transformer 系参数
      if (currentFamily === 'transformer') {
        setTransformerParams(prev => ({
          ...prev,
          ...(typeof p.enc_in     === 'number' ? { enc_in:     p.enc_in     } : {}),
          ...(typeof p.dec_in     === 'number' ? { dec_in:     p.dec_in     } : {}),
          ...(typeof p.c_out      === 'number' ? { c_out:      p.c_out      } : {}),
          ...(typeof p.d_model    === 'number' ? { d_model:    p.d_model    } : {}),
          ...(typeof p.n_heads    === 'number' ? { n_heads:    p.n_heads    } : {}),
          ...(typeof p.e_layers   === 'number' ? { e_layers:   p.e_layers   } : {}),
          ...(typeof p.d_layers   === 'number' ? { d_layers:   p.d_layers   } : {}),
          ...(typeof p.d_ff       === 'number' ? { d_ff:       p.d_ff       } : {}),
          ...(typeof p.factor     === 'number' ? { factor:     p.factor     } : {}),
          ...(typeof p.dropout    === 'number' ? { dropout:    p.dropout    } : {}),
          ...(typeof p.moving_avg === 'number' ? { moving_avg: p.moving_avg } : {}),
          ...(typeof p.embed      === 'string' ? { embed:      p.embed as string } : {}),
          ...(typeof p.activation === 'string' ? { activation: p.activation as string } : {}),
        }))
      }

      // 回填 PatchTST 参数
      if (currentFamily === 'patchtst') {
        setPatchTSTParams(prev => ({
          ...prev,
          ...(typeof p.enc_in        === 'number' ? { enc_in:        p.enc_in        } : {}),
          ...(typeof p.patch_len     === 'number' ? { patch_len:     p.patch_len     } : {}),
          ...(typeof p.stride        === 'number' ? { stride:        p.stride        } : {}),
          ...(typeof p.fc_dropout    === 'number' ? { fc_dropout:    p.fc_dropout    } : {}),
          ...(typeof p.head_dropout  === 'number' ? { head_dropout:  p.head_dropout  } : {}),
          ...(typeof p.kernel_size   === 'number' ? { kernel_size:   p.kernel_size   } : {}),
          ...(typeof p.revin         === 'number' ? { revin:         p.revin         } : {}),
          ...(typeof p.affine        === 'number' ? { affine:        p.affine        } : {}),
          ...(typeof p.subtract_last === 'number' ? { subtract_last: p.subtract_last } : {}),
          ...(typeof p.decomposition === 'number' ? { decomposition: p.decomposition } : {}),
          ...(typeof p.individual    === 'number' ? { individual:    p.individual    } : {}),
          ...(typeof p.padding_patch === 'string' ? { padding_patch: p.padding_patch as string } : {}),
        }))
      }

      // 回填 Linear 参数
      if (currentFamily === 'linear') {
        setLinearParams(prev => ({
          ...prev,
          ...(typeof p.individual === 'number' ? { individual: p.individual } : {}),
        }))
      }
    } catch {
      // 静默失败
    } finally {
      setScriptLoading(false)
    }
  }

  // 日志新增时自动滚动到底部
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trainLogs])

  function handleStartTraining() {
    const config: TrainConfig = {
      model: selectedModel,
      ...basicParams,
      ...(currentFamily === 'transformer' ? transformerParams : {}),
      ...(currentFamily === 'patchtst'    ? patchTSTParams    : {}),
      ...(currentFamily === 'linear'      ? linearParams      : {}),
    }
    onStartTraining(config)
  }

  const isRunning = trainingStatus?.status === 'running'
  const disabled  = isLoading || isRunning

  return (
    <div>

      {/* ── 模型选择 ── */}
      <SectionHeader title="模型选择" />
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: '12px',
        marginBottom: '32px',
      }}>
        {MODELS.map(m => {
          const fs     = FAMILY_STYLE[m.family]
          const active = selectedModel === m.name
          return (
            <div
              key={m.name}
              onClick={() => setSelectedModel(m.name)}
              style={{
                background:   active ? '#1e2540' : '#252535',
                border:       `1.5px solid ${active ? fs.border : '#3a3a55'}`,
                borderRadius: '10px',
                padding:      '14px 16px',
                cursor:       'pointer',
                transition:   'all 0.15s',
                boxShadow:    active ? `0 0 0 1px ${fs.border}33` : 'none',
              }}
              onMouseEnter={e => {
                if (!active) (e.currentTarget as HTMLDivElement).style.borderColor = fs.border + '88'
              }}
              onMouseLeave={e => {
                if (!active) (e.currentTarget as HTMLDivElement).style.borderColor = '#3a3a55'
              }}
            >
              {/* 家族标签行 */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <span style={{
                  background:   fs.badge,
                  color:        fs.badgeText,
                  borderRadius: '4px',
                  padding:      '1px 8px',
                  fontSize:     '11px',
                  fontStyle:    'italic',
                }}>{fs.tag}</span>
                {active && (
                  <span style={{
                    marginLeft:   'auto',
                    width: '8px', height: '8px',
                    borderRadius: '50%',
                    background:   fs.border,
                    display:      'block',
                    boxShadow:    `0 0 6px ${fs.border}`,
                  }} />
                )}
              </div>
              <div style={{ fontWeight: 'bold', fontSize: '15px', marginBottom: '4px', color: active ? '#e8eaf8' : '#ccc' }}>
                {m.name}
              </div>
              <div style={{ fontSize: '12px', color: '#666', lineHeight: '1.4' }}>{m.desc}</div>
            </div>
          )
        })}
      </div>

      {/* ── 参考脚本 ── */}
      <div style={{
        display:      'flex',
        alignItems:   'center',
        gap:          '10px',
        marginBottom: '12px',
        padding:      '9px 14px',
        background:   '#252535',
        borderRadius: '8px',
        border:       '1px solid #3a3a55',
      }}>
        <span style={{ fontSize: '12px', color: '#888', whiteSpace: 'nowrap' }}>参考脚本</span>
        <select
          value={selectedScript}
          onChange={e => handleScriptChange(e.target.value)}
          disabled={scriptLoading}
          style={{ ...selectStyle, flex: 1, opacity: scriptLoading ? 0.5 : 1 }}
        >
          <option value="">{scriptList.length === 0 ? 'No available scripts' : 'Without using scripts'}</option>
          {scriptList.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        {selectedScript && (
          <span style={{ fontSize: '11px', color: '#6fcfa0', whiteSpace: 'nowrap' }}>
            {scriptLoading ? '加载中…' : '✓ 参数已应用'}
          </span>
        )}
      </div>

      {/* ── 基本参数 ── */}
      <div style={{
        background:   '#252535',
        border:       '1px solid #3a3a55',
        borderRadius: '12px',
        padding:      '20px 24px',
        marginBottom: '20px',
      }}>
        <SectionHeader title="基本参数" onReset={resetBasic} />

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '14px 20px' }}>

          <FormItem label="数据集">
            <select
              value={basicParams.data}
              onChange={e => {
                const opt = DATASET_OPTIONS.find(o => o.data === e.target.value)!
                setBasicParams(p => ({ ...p, data: opt.data, data_path: opt.data_path }))
              }}
              style={selectStyle}
            >
              {DATASET_OPTIONS.map(o => <option key={o.data} value={o.data}>{o.data}</option>)}
            </select>
          </FormItem>

          <FormItem label="预测模式">
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['M', 'MS', 'S'] as const).map(f => (
                <button key={f} onClick={() => setBasicParams(p => ({ ...p, features: f }))}
                  style={toggleBtnStyle(basicParams.features === f)}>{f}</button>
              ))}
            </div>
          </FormItem>

          <FormItem label="输入步长 (seq_len)">
            <NumInput value={basicParams.seq_len} min={1} max={2000}
              onChange={v => setBasicParams(p => ({ ...p, seq_len: v }))} />
          </FormItem>

          <FormItem label="预测步长 (pred_len)">
            <NumInput value={basicParams.pred_len} min={1} max={2000}
              onChange={v => setBasicParams(p => ({ ...p, pred_len: v }))} />
          </FormItem>

          <FormItem label="标签步长 (label_len)">
            <NumInput value={basicParams.label_len} min={0} max={2000}
              onChange={v => setBasicParams(p => ({ ...p, label_len: v }))} />
          </FormItem>

          <FormItem label="训练轮数 (epochs)">
            <NumInput value={basicParams.train_epochs} min={1} max={500}
              onChange={v => setBasicParams(p => ({ ...p, train_epochs: v }))} />
          </FormItem>

          <FormItem label="早停轮数 (patience)">
            <NumInput value={basicParams.patience} min={1} max={100}
              onChange={v => setBasicParams(p => ({ ...p, patience: v }))} />
          </FormItem>

          <FormItem label="批量大小 (batch_size)">
            <NumInput value={basicParams.batch_size} min={1} max={512}
              onChange={v => setBasicParams(p => ({ ...p, batch_size: v }))} />
          </FormItem>

          <FormItem label="学习率 (lr)">
            <input
              type="number"
              value={basicParams.learning_rate}
              min={0.00001} max={1} step={0.001}
              onChange={e => setBasicParams(p => ({ ...p, learning_rate: parseFloat(e.target.value) || 0.005 }))}
              style={inputStyle}
            />
          </FormItem>

          <FormItem label="使用 GPU">
            <div style={{ display: 'flex', gap: '6px' }}>
              {([true, false] as const).map(v => (
                <button key={String(v)} onClick={() => setBasicParams(p => ({ ...p, use_gpu: v }))}
                  style={toggleBtnStyle(basicParams.use_gpu === v)}>{v ? '是' : '否'}</button>
              ))}
            </div>
          </FormItem>

        </div>
      </div>

      {/* ── 模型特有参数 ── */}
      {currentFamily !== 'stat' && (
        <div style={{
          background:   '#252535',
          border:       '1px solid #3a3a55',
          borderRadius: '12px',
          padding:      '20px 24px',
          marginBottom: '28px',
        }}>
          <SectionHeader title={`模型参数（${selectedModel}）`} onReset={resetModelParams} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '14px 20px' }}>

            {/* Transformer 系列 */}
            {currentFamily === 'transformer' && (<>
              <FormItem label="enc_in（编码器输入维度）">
                <NumInput value={transformerParams.enc_in} min={1} max={512}
                  onChange={v => setTransformerParams(p => ({ ...p, enc_in: v }))} />
              </FormItem>
              <FormItem label="dec_in（解码器输入维度）">
                <NumInput value={transformerParams.dec_in} min={1} max={512}
                  onChange={v => setTransformerParams(p => ({ ...p, dec_in: v }))} />
              </FormItem>
              <FormItem label="c_out（输出维度）">
                <NumInput value={transformerParams.c_out} min={1} max={512}
                  onChange={v => setTransformerParams(p => ({ ...p, c_out: v }))} />
              </FormItem>
              <FormItem label="d_model（嵌入维度）">
                <NumInput value={transformerParams.d_model} min={16} max={1024} step={16}
                  onChange={v => setTransformerParams(p => ({ ...p, d_model: v }))} />
              </FormItem>
              <FormItem label="n_heads（注意力头数）">
                <NumInput value={transformerParams.n_heads} min={1} max={32}
                  onChange={v => setTransformerParams(p => ({ ...p, n_heads: v }))} />
              </FormItem>
              <FormItem label="e_layers（编码器层数）">
                <NumInput value={transformerParams.e_layers} min={1} max={8}
                  onChange={v => setTransformerParams(p => ({ ...p, e_layers: v }))} />
              </FormItem>
              <FormItem label="d_layers（解码器层数）">
                <NumInput value={transformerParams.d_layers} min={1} max={8}
                  onChange={v => setTransformerParams(p => ({ ...p, d_layers: v }))} />
              </FormItem>
              <FormItem label="d_ff（FFN 维度）">
                <NumInput value={transformerParams.d_ff} min={64} max={8192} step={64}
                  onChange={v => setTransformerParams(p => ({ ...p, d_ff: v }))} />
              </FormItem>
              <FormItem label="factor（稀疏注意力因子）">
                <NumInput value={transformerParams.factor} min={1} max={10}
                  onChange={v => setTransformerParams(p => ({ ...p, factor: v }))} />
              </FormItem>
              <FormItem label="dropout">
                <input type="number" value={transformerParams.dropout} min={0} max={0.9} step={0.01}
                  onChange={e => setTransformerParams(p => ({ ...p, dropout: parseFloat(e.target.value) || 0.05 }))}
                  style={inputStyle} />
              </FormItem>
              <FormItem label="embed（时间编码方式）">
                <select value={transformerParams.embed}
                  onChange={e => setTransformerParams(p => ({ ...p, embed: e.target.value }))}
                  style={selectStyle}>
                  <option value="timeF">timeF（原始特征）</option>
                  <option value="fixed">fixed（固定编码）</option>
                  <option value="learned">learned（可学习）</option>
                </select>
              </FormItem>
              <FormItem label="activation（激活函数）">
                <select value={transformerParams.activation}
                  onChange={e => setTransformerParams(p => ({ ...p, activation: e.target.value }))}
                  style={selectStyle}>
                  <option value="gelu">gelu</option>
                  <option value="relu">relu</option>
                </select>
              </FormItem>
              {selectedModel === 'Autoformer' && (
                <FormItem label="moving_avg（移动平均窗口）">
                  <NumInput value={transformerParams.moving_avg} min={1} max={100}
                    onChange={v => setTransformerParams(p => ({ ...p, moving_avg: v }))} />
                </FormItem>
              )}
            </>)}

            {/* PatchTST */}
            {currentFamily === 'patchtst' && (<>
              <FormItem label="enc_in（输入通道数）">
                <NumInput value={patchTSTParams.enc_in} min={1} max={512}
                  onChange={v => setPatchTSTParams(p => ({ ...p, enc_in: v }))} />
              </FormItem>
              <FormItem label="patch_len（Patch 大小）">
                <NumInput value={patchTSTParams.patch_len} min={1} max={128}
                  onChange={v => setPatchTSTParams(p => ({ ...p, patch_len: v }))} />
              </FormItem>
              <FormItem label="stride（步幅）">
                <NumInput value={patchTSTParams.stride} min={1} max={128}
                  onChange={v => setPatchTSTParams(p => ({ ...p, stride: v }))} />
              </FormItem>
              <FormItem label="fc_dropout">
                <input type="number" value={patchTSTParams.fc_dropout} min={0} max={0.9} step={0.01}
                  onChange={e => setPatchTSTParams(p => ({ ...p, fc_dropout: parseFloat(e.target.value) || 0 }))}
                  style={inputStyle} />
              </FormItem>
              <FormItem label="head_dropout">
                <input type="number" value={patchTSTParams.head_dropout} min={0} max={0.9} step={0.01}
                  onChange={e => setPatchTSTParams(p => ({ ...p, head_dropout: parseFloat(e.target.value) || 0 }))}
                  style={inputStyle} />
              </FormItem>
              <FormItem label="padding_patch">
                <select value={patchTSTParams.padding_patch}
                  onChange={e => setPatchTSTParams(p => ({ ...p, padding_patch: e.target.value }))}
                  style={selectStyle}>
                  <option value="end">end（末尾填充）</option>
                  <option value="none">none（不填充）</option>
                </select>
              </FormItem>
              <FormItem label="kernel_size（分解核大小）">
                <NumInput value={patchTSTParams.kernel_size} min={1} max={200}
                  onChange={v => setPatchTSTParams(p => ({ ...p, kernel_size: v }))} />
              </FormItem>
              <FormItem label="RevIN 归一化">
                <BoolToggle value={patchTSTParams.revin}
                  onChange={v => setPatchTSTParams(p => ({ ...p, revin: v }))} />
              </FormItem>
              <FormItem label="affine（RevIN 仿射）">
                <BoolToggle value={patchTSTParams.affine}
                  onChange={v => setPatchTSTParams(p => ({ ...p, affine: v }))} />
              </FormItem>
              <FormItem label="subtract_last（减去最后值）">
                <BoolToggle value={patchTSTParams.subtract_last}
                  onChange={v => setPatchTSTParams(p => ({ ...p, subtract_last: v }))} />
              </FormItem>
              <FormItem label="decomposition（趋势分解）">
                <BoolToggle value={patchTSTParams.decomposition}
                  onChange={v => setPatchTSTParams(p => ({ ...p, decomposition: v }))} />
              </FormItem>
              <FormItem label="individual（独立通道头）">
                <BoolToggle value={patchTSTParams.individual}
                  onChange={v => setPatchTSTParams(p => ({ ...p, individual: v }))} />
              </FormItem>
            </>)}

            {/* Linear 系列 */}
            {currentFamily === 'linear' && (
              <FormItem label="individual（每通道独立线性层）">
                <BoolToggle value={linearParams.individual}
                  onChange={v => setLinearParams(p => ({ ...p, individual: v }))} />
              </FormItem>
            )}

          </div>
        </div>
      )}

      {/* ── 开始训练 ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', marginBottom: '20px' }}>
        <button
          onClick={handleStartTraining}
          disabled={disabled}
          style={{
            background:   disabled ? '#2a2a3e' : '#4a6fa5',
            color:        disabled ? '#666'    : '#fff',
            border:       'none',
            borderRadius: '8px',
            padding:      '10px 28px',
            fontSize:     '14px',
            fontWeight:   'bold',
            cursor:       disabled ? 'not-allowed' : 'pointer',
            transition:   'background 0.2s',
          }}
          onMouseEnter={e => { if (!disabled) (e.currentTarget as HTMLButtonElement).style.background = '#3a5f95' }}
          onMouseLeave={e => { if (!disabled) (e.currentTarget as HTMLButtonElement).style.background = '#4a6fa5' }}
        >
          {isRunning ? '⏳ 训练中…' : '▶ 开始训练'}
        </button>
        <span style={{ fontSize: '12px', color: '#555' }}>
          {isRunning
            ? `正在使用 ${selectedModel} 训练，请稍候…`
            : `将使用 ${selectedModel} · 数据集 ${basicParams.data} · 预测 ${basicParams.pred_len} 步`}
        </span>
      </div>

      {/* ── 控制台输出 ── */}
      <div style={{
        background:   '#0d0d0d',
        border:       '1px solid #2a2a2a',
        borderRadius: '10px',
        overflow:     'hidden',
        marginBottom: '24px',
      }}>
        {/* 标题栏 */}
        <div style={{
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'space-between',
          padding:        '8px 14px',
          background:     '#1a1a1a',
          borderBottom:   '1px solid #2a2a2a',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {/* 仿 macOS 三色圆点 */}
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ff5f57', display: 'inline-block' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#febc2e', display: 'inline-block' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#28c840', display: 'inline-block' }} />
            <span style={{ fontSize: '12px', color: '#555', marginLeft: '8px', fontFamily: 'monospace' }}>training output</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {isRunning && (
              <span style={{ fontSize: '11px', color: '#7b8cde', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{
                  display: 'inline-block', width: '6px', height: '6px',
                  borderRadius: '50%', background: '#7b8cde',
                }} />
                LIVE
              </span>
            )}
            <span style={{ fontSize: '11px', color: '#444', fontFamily: 'monospace' }}>
              {trainLogs.length} lines
            </span>
            <button
              onClick={() => onClearLogs()}
              style={{
                background: 'transparent', border: '1px solid #2a2a2a',
                color: '#555', borderRadius: '4px', padding: '2px 8px',
                fontSize: '11px', cursor: 'pointer',
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#aaa' }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
            >clear</button>
          </div>
        </div>
        {/* 日志内容 */}
        <div style={{
          height:     '300px',
          overflowY:  'auto',
          padding:    '12px 16px',
          fontFamily: '"Cascadia Code", "Fira Code", Consolas, monospace',
          fontSize:   '12px',
          lineHeight: '1.7',
          color:      '#c8c8c8',
        }}>
          {trainLogs.length === 0 ? (
            <span style={{ color: '#333' }}>// 等待训练开始…</span>
          ) : (
            trainLogs.map((line, i) => (
              <div key={i} style={{
                color: (line.includes('✓') || line.includes('completed') || line.includes('successfully'))
                         ? '#50fa7b'
                       : (line.includes('✗') || line.toLowerCase().includes('error') || line.toLowerCase().includes('failed'))
                         ? '#ff5555'
                       : line.startsWith('[') && line.includes(']')
                         ? '#6272a4'
                       : (line.includes('>>>') || line.includes('<<<'))
                         ? '#ffb86c'
                       : (line.toLowerCase().includes('epoch') || line.toLowerCase().includes('train loss'))
                         ? '#8be9fd'
                       : '#c8c8c8',
                whiteSpace: 'pre-wrap',
                wordBreak:  'break-all',
              }}>{line}</div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      </div>

      {/* ── 训练状态 ── */}
      {trainingStatus && (
        <div style={{
          padding:      '14px 20px',
          background:
            trainingStatus.status === 'completed' ? '#1a3a2a' :
            trainingStatus.status === 'failed'    ? '#3a1a1a' :
            trainingStatus.status === 'running'   ? '#1a2a3a' : '#252535',
          borderRadius: '10px',
          border: `1px solid ${
            trainingStatus.status === 'completed' ? '#2d9e6b' :
            trainingStatus.status === 'failed'    ? '#e05050' :
            trainingStatus.status === 'running'   ? '#4a6fa5' : '#3a3a55'
          }`,
          fontSize:   '13px',
          color:      '#ccc',
          lineHeight: '1.8',
        }}>
          <div>
            <span style={{ color: '#888' }}>状态：</span>
            <StatusBadge status={trainingStatus.status} />
          </div>
          <div><span style={{ color: '#888' }}>消息：</span>{trainingStatus.message}</div>
          {trainingStatus.start_time && (
            <div><span style={{ color: '#888' }}>开始时间：</span>{trainingStatus.start_time}</div>
          )}
        </div>
      )}
    </div>
  )
}

// ===================== 子组件 =====================

function SectionHeader({ title, onReset }: { title: string; onReset?: () => void }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
      <h3 style={{ margin: 0, fontSize: '13px', fontWeight: 'bold', color: '#a0a8d0', letterSpacing: '0.8px', textTransform: 'uppercase' }}>
        {title}
      </h3>
      {onReset && (
        <button
          onClick={onReset}
          style={{
            background:   'transparent',
            border:       '1px solid #3a3a55',
            color:        '#777',
            borderRadius: '6px',
            padding:      '3px 12px',
            fontSize:     '12px',
            cursor:       'pointer',
            transition:   'all 0.15s',
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLButtonElement).style.color        = '#bbb'
            ;(e.currentTarget as HTMLButtonElement).style.borderColor = '#7b8cde'
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.color        = '#777'
            ;(e.currentTarget as HTMLButtonElement).style.borderColor = '#3a3a55'
          }}
        >
          重置参数
        </button>
      )}
    </div>
  )
}

function FormItem({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <span style={{ fontSize: '12px', color: '#888' }}>{label}</span>
      {children}
    </label>
  )
}

function NumInput({ value, min, max, step = 1, onChange }: {
  value: number; min: number; max: number; step?: number; onChange: (v: number) => void
}) {
  return (
    <input
      type="number"
      value={value}
      min={min} max={max} step={step}
      onChange={e => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))}
      style={inputStyle}
    />
  )
}

function BoolToggle({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div style={{ display: 'flex', gap: '6px' }}>
      {([1, 0] as const).map(v => (
        <button key={v} onClick={() => onChange(v)} style={toggleBtnStyle(value === v)}>
          {v === 1 ? '是' : '否'}
        </button>
      ))}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const MAP: Record<string, { label: string; color: string }> = {
    idle:      { label: '空闲',   color: '#888'    },
    running:   { label: '训练中', color: '#7b8cde' },
    completed: { label: '已完成', color: '#2d9e6b' },
    failed:    { label: '失败',   color: '#e05050' },
  }
  const s = MAP[status] ?? { label: status, color: '#888' }
  return (
    <span style={{
      display:      'inline-block',
      marginLeft:   '6px',
      padding:      '1px 10px',
      borderRadius: '4px',
      fontSize:     '12px',
      fontWeight:   'bold',
      background:   s.color + '22',
      color:        s.color,
      border:       `1px solid ${s.color}55`,
    }}>{s.label}</span>
  )
}

// ===================== 共用样式 =====================
const inputStyle: React.CSSProperties = {
  width:        '100%',
  background:   '#1a1a2e',
  border:       '1px solid #3a3a55',
  borderRadius: '6px',
  color:        '#ccc',
  padding:      '5px 10px',
  fontSize:     '13px',
  boxSizing:    'border-box',
}

const selectStyle: React.CSSProperties = { ...inputStyle, cursor: 'pointer' }

function toggleBtnStyle(active: boolean): React.CSSProperties {
  return {
    padding:      '4px 14px',
    borderRadius: '6px',
    fontSize:     '12px',
    fontWeight:   active ? 'bold' : 'normal',
    cursor:       'pointer',
    border:       active ? '1px solid #7b8cde' : '1px solid #3a3a55',
    background:   active ? '#2a2a4e' : 'transparent',
    color:        active ? '#c0c8f0' : '#888',
    transition:   'all 0.15s',
  }
}
