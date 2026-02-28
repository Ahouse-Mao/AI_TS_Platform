import { useState, useEffect } from 'react'
import axios from 'axios'
import type { CheckpointInfo } from '../types'
import { InfoItem } from '../components/InfoItem'

// 模型预测页面 —— 展示可用 Checkpoint
export function PredictPage() {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]) // 存储从后端获取的 Checkpoint 列表
  const [loading, setLoading]         = useState<boolean>(true)        // 是否在请求中
  const [error, setError]             = useState<string | null>(null)  // 错误信息

  // 组件挂载时自动拉取列表（[] 表示只在首次渲染时执行一次）
  useEffect(() => {
    axios.get('http://localhost:8000/api/predict/checkpoints')
      .then(res => {
        setCheckpoints(res.data)
        setLoading(false)
      })
      .catch(err => {
        setError('获取 Checkpoint 列表失败：' + err.message)
        setLoading(false)
      })
  }, [])

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>模型预测 —— 可用 Checkpoints</h3>

      {/* 加载中 */}
      {loading && <p style={{ color: '#aaa' }}>加载中...</p>}

      {/* 错误提示 */}
      {error && (
        <p style={{ color: '#f08080', background: '#2a1a1a', padding: '10px 16px', borderRadius: '8px' }}>
          {error}
        </p>
      )}

      {/* 无数据提示 */}
      {!loading && !error && checkpoints.length === 0 && (
        <p style={{ color: '#888' }}>checkpoints 目录为空，请先训练模型。</p>
      )}

      {/* Checkpoint 卡片列表 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {checkpoints.map((ckpt, idx) => (
          <div
            key={idx}
            style={{
              background: '#252535',
              border: '1px solid #3a3a55',
              borderRadius: '12px',
              padding: '20px 24px',
            }}
          >
            {/* 解析失败：显示原始文件夹名 */}
            {ckpt.parse_error ? (
              <div>
                <span style={{ color: '#f08080', fontWeight: 'bold' }}>⚠ 无法解析名称：</span>
                <code style={{ color: '#ccc', fontSize: '13px', marginLeft: '8px' }}>{ckpt.folder_name}</code>
              </div>
            ) : (
              <>
                {/* 标题行：模型徽标 + model_id，右侧文件标签 */}
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
                    {ckpt.has_pth && (
                      <span style={{ background: '#2d6a4f', color: '#b7e4c7', borderRadius: '4px', padding: '2px 8px', fontSize: '12px' }}>
                        checkpoint.pth
                      </span>
                    )}
                    {ckpt.has_onnx && (
                      <span style={{ background: '#1b4965', color: '#90e0ef', borderRadius: '4px', padding: '2px 8px', fontSize: '12px' }}>
                        model.onnx
                      </span>
                    )}
                  </div>
                </div>

                {/* 参数网格：自动响应式列数 */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                  gap: '8px 16px',
                  fontSize: '13px',
                  color: '#ccc',
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
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
