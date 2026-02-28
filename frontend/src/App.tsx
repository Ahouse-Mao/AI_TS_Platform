import { useState, useEffect } from 'react' // React 的 useState/useEffect Hook
import axios from 'axios' // HTTP 请求库，比原生 fetch 更简洁, 用于和后端通信

// 训练状态的类型定义
interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  message: string
  start_time: string | null
  conda_env: string
}

// 导航 Tab 类型
type TabKey = 'train' | 'predict' | 'assistant'

// ===================== 子页面组件 =====================

// 模型训练与配置页面
function TrainPage() {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  // 存储训练状态对象，初始值 null（还没查询过）
  const [isLoading, setIsLoading] = useState<boolean>(false)
  // 存储按钮是否在请求中的标志, 防止重复提交

  // 启动训练函数
  const startTraining = async () => {
    try {
      setIsLoading(true)
      const response = await axios.post('http://localhost:8000/api/train/start')
      if (response.data.success) {
        startPollingStatus()
      }
    } catch (error) {
      console.error('Error starting training:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // 轮询状态函数：每2秒向后端查询一次训练状态
  const startPollingStatus = () => {
    const interval = setInterval(async () => {
      const response = await axios.get('http://localhost:8000/api/train/status')
      setTrainingStatus(response.data)
      if (response.data.status === 'completed' || response.data.status === 'failed') {
        clearInterval(interval) // 训练结束，停止轮询
      }
    }, 2000)
  }

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>模型训练</h3>
      <button
        onClick={startTraining}
        disabled={isLoading || trainingStatus?.status === 'running'}
        style={{
          padding: '12px 24px',
          fontSize: '16px',
          cursor: trainingStatus?.status === 'running' ? 'not-allowed' : 'pointer',
          backgroundColor: trainingStatus?.status === 'running' ? '#d61b1b' : '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '10px',
        }}
      >
        {trainingStatus?.status === 'running' ? '训练中...' : '开始训练'}
      </button>

      {/* 训练状态显示：只有 trainingStatus 不为 null 时才渲染 */}
      {trainingStatus && (
        <div style={{
          marginTop: '20px',
          padding: '10px 20px',
          background:
            trainingStatus.status === 'completed' ? '#d4edda' :
            trainingStatus.status === 'failed'    ? '#f8d7da' : '#fff3cd',
          borderRadius: '10px',
          color: '#333',
          border: `5px solid ${
            trainingStatus.status === 'completed' ? '#a8d5b5' :
            trainingStatus.status === 'failed'    ? '#f0a8ae' : '#ffe08a'
          }`,
        }}>
          <strong>训练状态：</strong> {trainingStatus.status} <br />
          <strong>消息：</strong> {trainingStatus.message} <br />
          {trainingStatus.start_time && (
            <><strong>开始时间：</strong> {trainingStatus.start_time}</>
          )}
        </div>
      )}
    </div>
  )
}

// ── Checkpoint 信息类型 ──
interface CheckpointInfo {
  folder_name:   string
  parse_error?:  boolean
  model_id?:     string
  model?:        string
  dataset?:      string
  features?:     string
  features_desc?: string
  seq_len?:      number
  label_len?:    number
  pred_len?:     number
  d_model?:      number
  n_heads?:      number
  e_layers?:     number
  d_layers?:     number
  d_ff?:         number
  factor?:       number
  embed?:        string
  distil?:       string
  des?:          string
  exp_id?:       number
  has_pth?:      boolean
  has_onnx?:     boolean
}

// 模型预测页面 —— 展示可用 Checkpoint
function PredictPage() {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [loading, setLoading]         = useState<boolean>(true)
  const [error, setError]             = useState<string | null>(null)

  // 组件挂载时自动拉取列表
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
  }, []) // [] 表示只在组件首次渲染时执行一次

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
            {/* 解析失败提示 */}
            {ckpt.parse_error ? (
              <div>
                <span style={{ color: '#f08080', fontWeight: 'bold' }}>⚠ 无法解析名称：</span>
                <code style={{ color: '#ccc', fontSize: '13px', marginLeft: '8px' }}>{ckpt.folder_name}</code>
              </div>
            ) : (
              <>
                {/* 标题行：模型 + 实验ID */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{
                      background: '#4a6fa5', color: '#fff',
                      borderRadius: '6px', padding: '2px 10px',
                      fontWeight: 'bold', fontSize: '14px'
                    }}>{ckpt.model}</span>
                    <span style={{ fontWeight: 'bold', fontSize: '16px' }}>{ckpt.model_id}</span>
                  </div>
                  {/* 文件标签 */}
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {ckpt.has_pth && (
                      <span style={{
                        background: '#2d6a4f', color: '#b7e4c7',
                        borderRadius: '4px', padding: '2px 8px', fontSize: '12px'
                      }}>checkpoint.pth</span>
                    )}
                    {ckpt.has_onnx && (
                      <span style={{
                        background: '#1b4965', color: '#90e0ef',
                        borderRadius: '4px', padding: '2px 8px', fontSize: '12px'
                      }}>model.onnx</span>
                    )}
                  </div>
                </div>

                {/* 参数网格 */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                  gap: '8px 16px',
                  fontSize: '13px',
                  color: '#ccc',
                }}>
                  <InfoItem label="数据集"    value={ckpt.dataset!} />
                  <InfoItem label="特征模式"  value={`${ckpt.features} (${ckpt.features_desc})`} />
                  <InfoItem label="输入步长"  value={String(ckpt.seq_len)} />
                  <InfoItem label="预测步长"  value={String(ckpt.pred_len)} />
                  <InfoItem label="标签步长"  value={String(ckpt.label_len)} />
                  <InfoItem label="d_model"   value={String(ckpt.d_model)} />
                  <InfoItem label="注意力头"  value={String(ckpt.n_heads)} />
                  <InfoItem label="编码层数"  value={String(ckpt.e_layers)} />
                  <InfoItem label="解码层数"  value={String(ckpt.d_layers)} />
                  <InfoItem label="FFN维度"   value={String(ckpt.d_ff)} />
                  <InfoItem label="时间编码"  value={ckpt.embed!} />
                  <InfoItem label="Distil"    value={ckpt.distil!} />
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// 单行参数展示小组件
function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', gap: '6px' }}>
      <span style={{ color: '#888', whiteSpace: 'nowrap' }}>{label}:</span>
      <span style={{ color: '#e0e0e0', fontWeight: 500 }}>{value}</span>
    </div>
  )
}

// AI 助手页面（占位）
function AssistantPage() {
  return (
    <div>
      <h3 style={{ marginTop: 0 }}>AI 助手</h3>
      <p style={{ color: '#888' }}>AI 助手功能正在开发中...</p>
    </div>
  )
}

// ===================== 主页面 =====================

// 导航栏配置
const TABS: { key: TabKey; label: string }[] = [
  { key: 'train',     label: '模型训练与配置' },
  { key: 'predict',   label: '模型预测' },
  { key: 'assistant', label: 'AI 助手' },
]

// 页面背景色
const PAGE_BG = '#1e1e2e'
// 导航栏背景色（比页面略深，用于非激活标签）
const NAV_BG = '#16161f'

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('train')
  // 当前激活的 Tab，初始值为"模型训练与配置"

  return (
    <div style={{ minHeight: '100vh', backgroundColor: PAGE_BG, fontFamily: 'sans-serif', color: '#e0e0e0' }}>
      {/* 最小高度   撑满整个屏幕 */}

      {/* ===== 顶部导航栏 ===== */}
      <div style={{ backgroundColor: NAV_BG, display: 'flex', alignItems: 'flex-end', padding: '0 40px' }}>
        {/* flex: 横向排布元素 *, flex-end: 子元素底部对齐/}

        {/* 左侧标题 */}
        <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '18px', padding: '16px 24px 16px 0', marginRight: '16px' }}>
          时序预测平台
        </div>

        {/* Tab 按钮列表 */}
        {TABS.map(tab => {
          {/* .map()遍历数组, 检查每个tab是否处于激活状态 */}
          {/* 类似python列表推导式[render(tab) for tab in TABS] */}
          const isActive = activeTab === tab.key
          // 判断这个 Tab 是否是当前激活的
          // 例如: activeTab='train', tab.key='train' → isActive=true
          return (
            // 函数自己的return, 负责渲染这个 Tab 的 JSX
            <div
              key={tab.key}
              // ↑ React 要求列表中每个元素有唯一的 key，用于性能优化

              onClick={() => setActiveTab(tab.key)}
              
              // ↑ 点击时把 activeTab 改为这个 Tab 的 key
              // → 触发重新渲染 → 这个 Tab 变为激活状态
              style={{
                padding: '12px 24px',
                cursor: 'pointer',
                fontSize: '15px',
                fontWeight: isActive ? 'bold' : 'normal',
                color: isActive ? '#fff' : '#aaa',
                backgroundColor: isActive ? PAGE_BG : 'transparent',
                // 激活 → 背景色与页面相同（视觉上融为一体）
                // 未激活 → 透明背景
                borderRadius: isActive ? '8px 8px 0 0' : '8px 8px 0 0',
                borderTop:    isActive ? '1px solid #444' : '1px solid transparent',
                borderLeft:   isActive ? '1px solid #444' : '1px solid transparent',
                borderRight:  isActive ? '1px solid #444' : '1px solid transparent',
                borderBottom: 'none',
                // 激活 → 三边有细边框，底边没有（与页面内容连通）
                // 未激活 → 透明边框（占位，防止布局跳动）
                transition: 'all 0.3s',
                userSelect: 'none',
              }}
            >
              {tab.label}
            </div>
          )
        })}
      </div>

      {/* ===== 页面内容区域 ===== */}
      {/* 根据activeTab的值决定显示下面的子页面 */}
      <div style={{ padding: '40px 50px' }}>
        {activeTab === 'train'     && <TrainPage />}
        {activeTab === 'predict'   && <PredictPage />}
        {activeTab === 'assistant' && <AssistantPage />}
      </div>

    </div>
  )
}

export default App