import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import type { CheckpointInfo, InferenceResult } from '../types'
import { InfoItem } from '../components/InfoItem'
import { ssGet, ssSet } from '../utils/storage'
import { API_BASE } from '../config'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

// ===================== 类型 =====================
// InferenceResult 已移至 src/types.ts 统一管理

// sessionStorage 辅助：按 folderName 维度读写 JSON
// 已提取到 src/utils/storage.ts

// ===================== 主组件 =====================
export function InferencePage() {
  const { folderName } = useParams<{ folderName: string }>()
  const navigate        = useNavigate()

  const [ckpt,          setCkpt]          = useState<CheckpointInfo | null>(null)
  const [ckptLoading,   setCkptLoading]   = useState<boolean>(true)
  const [isLeaving,     setIsLeaving]     = useState<boolean>(false)

  const ssNKey = `infer_nsamples_${folderName}`
  const ssOKey = `infer_onnx_${folderName}`
  const ssRKey = `infer_result_${folderName}`

  const [nSamples,      setNSamples]      = useState<number>(() => ssGet(ssNKey, 200))
  const [useOnnx,       setUseOnnx]       = useState<boolean>(() => ssGet(ssOKey, false))

  const [inferLoading,  setInferLoading]  = useState<boolean>(false)
  const [inferError,    setInferError]    = useState<string | null>(null)
  const [inferResult,   setInferResult]   = useState<InferenceResult | null>(() => ssGet<InferenceResult | null>(ssRKey, null))

  // 持久化设置参数到 sessionStorage
  useEffect(() => { ssSet(ssNKey, nSamples) }, [nSamples])    // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { ssSet(ssOKey, useOnnx)  }, [useOnnx])     // eslint-disable-line react-hooks/exhaustive-deps

  // 拉取 checkpoint 列表，找到当前项
  useEffect(() => {
    if (!folderName) return
    axios.get(`${API_BASE}/api/predict/checkpoints`)
      .then(res => {
        const found = (res.data as CheckpointInfo[]).find(c => c.folder_name === folderName)
        setCkpt(found ?? null)
      })
      .finally(() => setCkptLoading(false))
  }, [folderName])

  // 点击"开始推理"
  function handleStartInference() {
    if (!folderName || inferLoading) return
    setInferResult(null)
    setInferError(null)
    setInferLoading(true)

    // TODO: 默认方案改为 ONNX 轻量推理，当前暂用 PTH
    axios.post(`${API_BASE}/api/predict/run`, {
      folder_name: folderName,
      n_samples:   nSamples,
      use_onnx:    useOnnx,
    })
      .then(res => {
        if (res.data.error) setInferError(res.data.error)
        else {
          const result = res.data as InferenceResult
          setInferResult(result)
          ssSet(ssRKey, result)   // 持久化到 sessionStorage
        }
      })
      .catch(err => setInferError('推理请求失败：' + err.message))
      .finally(() => setInferLoading(false))
  }

  // 点击返回：先播放向右滑出动画（350ms），再执行路由跳转
  function handleGoBack() {
    setIsLeaving(true)
    setTimeout(() => navigate('/predict'), 320)
  }

  return (
    // isLeaving 为 true 时切换为退出动画类
    <div className={isLeaving ? 'page-slide-out' : 'page-slide-in'}>

      {/* ── 返回按钮（内容区左上角，不在顶栏内） ── */}
      <button
        onClick={handleGoBack}
        style={{
          display:        'flex',
          alignItems:     'center',
          gap:            '6px',
          background:     'transparent',
          border:         '1px solid #3a3a55',
          color:          '#aaa',
          borderRadius:   '8px',
          padding:        '6px 14px',
          fontSize:       '13px',
          cursor:         'pointer',
          marginBottom:   '24px',
          transition:     'color 0.15s, border-color 0.15s',
        }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLButtonElement).style.color        = '#fff'
          ;(e.currentTarget as HTMLButtonElement).style.borderColor = '#7b8cde'
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLButtonElement).style.color        = '#aaa'
          ;(e.currentTarget as HTMLButtonElement).style.borderColor = '#3a3a55'
        }}
      >
        {/* SVG 左箭头 */}
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M9 2L4 7L9 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        返回
      </button>

      {/* ── Checkpoint 信息卡片 ── */}
      {ckptLoading && <p style={{ color: '#888' }}>加载中...</p>}

      {!ckptLoading && !ckpt && (
        <p style={{ color: '#f08080' }}>未找到 Checkpoint：{folderName}</p>
      )}

      {ckpt && !ckpt.parse_error && (
        <div style={{
          background:    '#252535',
          border:        '1px solid #3a3a55',
          borderRadius:  '12px',
          padding:       '20px 24px',
          marginBottom:  '24px',
        }}>
          {/* 标题行 */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{
                background: '#4a6fa5', color: '#fff',
                borderRadius: '6px', padding: '2px 10px',
                fontWeight: 'bold', fontSize: '14px',
              }}>{ckpt.model}</span>
              <span style={{ fontWeight: 'bold', fontSize: '16px' }}>{ckpt.model_id}</span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              {ckpt.has_pth  && <span style={{ background: '#2d6a4f', color: '#b7e4c7', borderRadius: '4px', padding: '2px 8px', fontSize: '12px' }}>checkpoint.pth</span>}
              {ckpt.has_onnx && <span style={{ background: '#1b4965', color: '#90e0ef', borderRadius: '4px', padding: '2px 8px', fontSize: '12px' }}>model.onnx</span>}
            </div>
          </div>

          {/* 参数网格 */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
            gap: '8px 16px', fontSize: '13px', color: '#ccc',
          }}>
            <InfoItem label="数据集"   value={ckpt.dataset!} />
            <InfoItem label="特征模式" value={`${ckpt.features} (${ckpt.features_desc})`} />
            <InfoItem label="输入步长" value={String(ckpt.seq_len)} />
            <InfoItem label="预测步长" value={String(ckpt.pred_len)} />
            <InfoItem label="标签步长" value={String(ckpt.label_len)} />
            <InfoItem label="d_model"  value={String(ckpt.d_model)} />
            <InfoItem label="注意力头" value={String(ckpt.n_heads)} />
            <InfoItem label="编码层数" value={String(ckpt.e_layers)} />
            <InfoItem label="解码层数" value={String(ckpt.d_layers)} />
            <InfoItem label="FFN维度"  value={String(ckpt.d_ff)} />
            <InfoItem label="时间编码" value={ckpt.embed!} />
            <InfoItem label="Distil"   value={ckpt.distil!} />
          </div>
        </div>
      )}

      {/* ── 推理设置 + 开始推理按钮 ── */}
      {ckpt && !ckpt.parse_error && (
        <div style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>

          {/* 推理后端选择 */}
          <div style={{ display: 'flex', gap: '6px' }}>
            {(['pth', 'onnx'] as const).map(mode => {
              const disabled = mode === 'onnx' && !ckpt.has_onnx
              const active   = (mode === 'onnx') === useOnnx
              return (
                <button
                  key={mode}
                  disabled={disabled || inferLoading}
                  onClick={() => setUseOnnx(mode === 'onnx')}
                  style={{
                    padding:      '5px 14px',
                    borderRadius: '6px',
                    fontSize:     '12px',
                    fontWeight:   active ? 'bold' : 'normal',
                    cursor:       disabled ? 'not-allowed' : 'pointer',
                    opacity:      disabled ? 0.35 : 1,
                    border:       active ? '1px solid #7b8cde' : '1px solid #3a3a55',
                    background:   active ? '#2a2a4e' : 'transparent',
                    color:        active ? '#c0c8f0' : '#888',
                    transition:   'all 0.15s',
                  }}
                >
                  {mode.toUpperCase()}
                </button>
              )
            })}
          </div>

          {/* 样本数输入 */}
          <label style={{ fontSize: '13px', color: '#aaa', display: 'flex', alignItems: 'center', gap: '8px' }}>
            推理样本数
            <input
              type="number"
              value={nSamples}
              min={1}
              max={5000}
              step={50}
              onChange={e => setNSamples(Math.max(1, Number(e.target.value)))}
              disabled={inferLoading}
              style={{
                width: '80px', background: '#1a1a2e', border: '1px solid #3a3a55',
                borderRadius: '6px', color: '#ccc', padding: '4px 8px', fontSize: '13px',
              }}
            />
          </label>

          <button
            onClick={handleStartInference}
            disabled={inferLoading || !(ckpt.has_pth || ckpt.has_onnx)}
            style={{
              background:   inferLoading ? '#2a2a3e' : '#4a6fa5',
              color:        inferLoading ? '#666'    : '#fff',
              border:       'none',
              borderRadius: '8px',
              padding:      '10px 28px',
              fontSize:     '14px',
              fontWeight:   'bold',
              cursor:       inferLoading ? 'not-allowed' : 'pointer',
              transition:   'background 0.2s',
            }}
          >
            {inferLoading ? '⏳ 推理中…' : '▶ 开始推理'}
          </button>
          <span style={{ fontSize: '12px', color: '#555' }}>
            {ckpt.has_onnx ? '当前使用 ONNX 推理（轻量）' : 'PTH 推理（暂未导出 ONNX）'}
          </span>
        </div>
      )}

      {/* ── 推理结果 ── */}
      {inferError && (
        <p style={{ color: '#f08080', background: '#2a1a1a', padding: '10px 16px', borderRadius: '8px' }}>
          {inferError}
        </p>
      )}

      {inferResult && !inferLoading && (
        <InferenceResultPanel result={inferResult} />
      )}

    </div>
  )
}

// ===================== 结果展示面板 =====================
function InferenceResultPanel({ result }: { result: InferenceResult }) {
  const [sampleIdx, setSampleIdx] = useState<number>(0)

  const pred   = result.preds[sampleIdx]
  const true_  = result.trues[sampleIdx]
  const input_ = result.inputs[sampleIdx]
  const seqLen = result.seq_len

  // 输入区：x = 0 .. seqLen-1，只有 input 值
  // 预测区：x = seqLen .. seqLen+predLen-1，有 pred 和 true
  const chartData = [
    ...input_.map((v, i) => ({
      x:     i,
      input: parseFloat(v.toFixed(4)),
    })),
    ...pred.map((p, i) => ({
      x:    seqLen + i,
      pred: parseFloat(p.toFixed(4)),
      true: parseFloat(true_[i].toFixed(4)),
    })),
  ]

  return (
    <div style={{
      background:   '#1c1c2e',
      border:       '1px solid #3a3a55',
      borderRadius: '12px',
      padding:      '24px',
    }}>
      {/* 指标 */}
      <div style={{ display: 'flex', gap: '24px', marginBottom: '24px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <MetricBadge label="MSE" value={result.metrics.mse} color="#7b8cde" />
        <MetricBadge label="MAE" value={result.metrics.mae} color="#7be4c8" />
        <div style={{ fontSize: '12px', color: '#555', lineHeight: '1.6' }}>
          共推理 {result.n_samples} 个样本（测试集共 {result.n_total} 个可用）<br />
          每样本预测步长 {result.pred_len} 步<br />
          指标在标准化空间计算（与训练度量一致）<br />
          推理后端：<span style={{ color: result.backend === 'onnx' ? '#90e0ef' : '#b7e4c7' }}>
            {result.backend.toUpperCase()}
          </span>
          &nbsp;设备：<span style={{ color: result.active_device.includes('CUDA') || result.active_device === 'cuda' ? '#ffd166' : '#888' }}>
            {result.active_device}
          </span>
        </div>
      </div>

      {/* 样本选择拖动条 */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <span style={{ fontSize: '12px', color: '#888' }}>
            样本索引：<span style={{ color: '#bbb', fontFamily: 'monospace' }}>{sampleIdx}</span>
            <span style={{ color: '#555' }}> / {result.n_samples - 1}</span>
          </span>
          <span style={{ fontSize: '11px', color: '#555' }}>
            相对测试集起点偏移 +{sampleIdx} 步
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={result.n_samples - 1}
          value={sampleIdx}
          onChange={e => setSampleIdx(Number(e.target.value))}
          style={{ width: '100%', accentColor: '#7b8cde' }}
        />
      </div>

      {/* 折线图 */}
      <div style={{ color: '#888', fontSize: '12px', marginBottom: '8px' }}>
        样本 #{sampleIdx}：预测值 vs 真实值（目标变量 OT，标准化空间）
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, seqLen + result.pred_len - 1]}
            tick={{ fill: '#555', fontSize: 11 }}
            tickLine={false}
            label={{ value: '时间步', position: 'insideBottomRight', offset: -4, fill: '#444', fontSize: 11 }}
          />
          <YAxis tick={{ fill: '#555', fontSize: 11 }} tickLine={false} width={48} />
          <Tooltip
            contentStyle={{ background: '#1e1e30', border: '1px solid #3a3a55', borderRadius: '8px', fontSize: '12px' }}
            labelStyle={{ color: '#777' }}
            labelFormatter={v => `时间步 ${v}${Number(v) >= seqLen ? '（预测区）' : '（输入区）'}`}
          />
          <Legend wrapperStyle={{ fontSize: '12px', color: '#777' }} />
          {/* 输入区分隔线 */}
          <Line type="monotone" dataKey="input" stroke="#a0a0c0" dot={false} strokeWidth={1.2} name="输入序列" connectNulls={false} />
          <Line type="monotone" dataKey="true"  stroke="#7be4c8" dot={false} strokeWidth={1.5} name="真实值" connectNulls={false} />
          <Line type="monotone" dataKey="pred"  stroke="#7b8cde" dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="预测值" connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ===================== 指标徽章 =====================
function MetricBadge({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{
      background: '#252535', border: `1px solid ${color}44`,
      borderRadius: '10px', padding: '10px 20px',
      minWidth: '100px', textAlign: 'center',
    }}>
      <div style={{ color: '#666', fontSize: '12px', marginBottom: '4px' }}>{label}</div>
      <div style={{ color, fontSize: '20px', fontWeight: 'bold', fontFamily: 'monospace' }}>
        {value.toFixed(4)}
      </div>
    </div>
  )
}
