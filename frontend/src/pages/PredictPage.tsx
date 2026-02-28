import { useNavigate } from 'react-router-dom'
import type { CheckpointInfo } from '../types'
import { InfoItem } from '../components/InfoItem'

// ===================== Props =====================
interface PredictPageProps {
  checkpoints: CheckpointInfo[]
  loading:     boolean
  error:       string | null
  onRefresh:   () => void
}

// ===================== 主组件 =====================
export function PredictPage({ checkpoints, loading, error, onRefresh }: PredictPageProps) {
  const navigate = useNavigate()

  return (
    <div>
      {/* 标题行 + 刷新按钮 */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
        <h3 style={{ margin: 0 }}>模型预测 —— 可用 Checkpoints</h3>
        <button
          onClick={onRefresh}
          disabled={loading}
          title="刷新列表"
          style={{
            display:      'flex',
            alignItems:   'center',
            gap:          '5px',
            background:   'transparent',
            border:       '1px solid #3a3a55',
            borderRadius: '7px',
            color:        loading ? '#444' : '#aaa',
            padding:      '5px 12px',
            fontSize:     '12px',
            cursor:       loading ? 'not-allowed' : 'pointer',
            transition:   'color 0.15s, border-color 0.15s',
          }}
          onMouseEnter={e => { if (!loading) { (e.currentTarget as HTMLButtonElement).style.color = '#fff'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#7b8cde' } }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#aaa'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#3a3a55' }}
        >
          {/* 圆形刷新图标 */}
          <svg
            width="13" height="13" viewBox="0 0 13 13" fill="none"
            style={{ animation: loading ? 'spin 0.8s linear infinite' : 'none' }}
          >
            <path
              d="M11 6.5A4.5 4.5 0 1 1 6.5 2a4.5 4.5 0 0 1 3.18 1.32M9.68 1v2.32H12"
              stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
            />
          </svg>
          刷新
        </button>
      </div>
      <p style={{ color: '#888', fontSize: '13px', marginTop: '6px', marginBottom: '16px' }}>
        点击卡片进入推理页面
      </p>

      {loading && <p style={{ color: '#aaa' }}>加载中...</p>}

      {error && (
        <p style={{ color: '#f08080', background: '#2a1a1a', padding: '10px 16px', borderRadius: '8px' }}>
          {error}
        </p>
      )}

      {!loading && !error && checkpoints.length === 0 && (
        <p style={{ color: '#888' }}>checkpoints 目录为空，请先训练模型。</p>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {checkpoints.map((ckpt, idx) => {
          const canClick = !ckpt.parse_error && (ckpt.has_pth || ckpt.has_onnx)

          return (
            <div
              key={idx}
              onClick={() => canClick && navigate(`/predict/${encodeURIComponent(ckpt.folder_name)}`)}
              // encodeURIComponent 处理特殊字符，避免 URL 错误
              style={{
                background:   '#252535',
                border:       '1px solid #3a3a55',
                borderRadius: '12px',
                padding:      '20px 24px',
                cursor:       canClick ? 'pointer' : 'default',
                transition:   'border-color 0.15s, box-shadow 0.15s',
                opacity:      canClick ? 1 : 0.6,
                userSelect:   'none',
              }}
              onMouseEnter={e => { if (canClick) (e.currentTarget as HTMLDivElement).style.borderColor = '#7b8cde' }}
              onMouseLeave={e => { if (canClick) (e.currentTarget as HTMLDivElement).style.borderColor = '#3a3a55' }}
              // e.currentTarget 是事件绑定的元素，保证我们修改的是卡片本身的样式, e.target 可能是内部的某个子元素，导致样式修改失效
              // as HTMLDivElement 是类型断言，告诉 TypeScript 我们知道 currentTarget 是一个 div 元素，这样就可以访问 style 属性了
            >
              {ckpt.parse_error ? (  /* 三元语法, 处理异常情况 */
                <div>
                  <span style={{ color: '#f08080', fontWeight: 'bold' }}>⚠ 无法解析名称：</span>
                  <code style={{ color: '#ccc', fontSize: '13px', marginLeft: '8px' }}>{ckpt.folder_name}</code>
                </div>
              ) : (
                <>
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
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
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
                </>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
