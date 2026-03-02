// ===================== 应用骨架 =====================
// 路由配置、导航栏、页面布局以及跨页共享等全局性功能

import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import type { TrainingStatus, CheckpointInfo, TrainConfig } from './types'
import { TrainPage }       from './pages/TrainPage'
import { PredictPage }     from './pages/PredictPage'
import { AssistantPage }   from './pages/AssistantPage'
import { InferencePage }   from './pages/InferencePage'
import { LoginPage }       from './pages/LoginPage'
import { RegisterPage }    from './pages/RegisterPage'
import { ProtectedRoute }  from './components/ProtectedRoute'
import { useAuth, setupAxiosAuth } from './hooks/useAuth'

// ── 全局 axios 拦截器：所有请求自动附加 Bearer Token ──
setupAxiosAuth(() => localStorage.getItem('ai_ts_token'))

// 导航栏配置：每个 Tab 对应一个路由路径
const TABS = [
  { path: '/train',     label: '模型训练与配置' },
  { path: '/predict',   label: '模型预测' },
  { path: '/assistant', label: 'AI 助手' },
]

// 页面背景色
const PAGE_BG = '#1e1e2e'
// 导航栏背景色（比页面略深，用于非激活标签）
const NAV_BG = '#16161f'

function App() {
  // ── 认证状态 ──
  const { isAuthed, username, login, register, logout } = useAuth()

  // ── 训练状态提升到 App ──
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [isTrainLoading, setIsTrainLoading] = useState<boolean>(false)

  // ── 训练日志提升到 App，路由切换不丢失 ──
  const [trainLogs,  setTrainLogs]  = useState<string[]>([])
  const logPollRef                  = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Checkpoint 列表提升到 App（仅首次加载，切换路由不重复请求） ──
  const [checkpoints,        setCheckpoints]        = useState<CheckpointInfo[]>([])
  const [checkpointsLoading, setCheckpointsLoading] = useState<boolean>(true)
  const [checkpointsError,   setCheckpointsError]   = useState<string | null>(null)

  useEffect(() => {
    axios.get('http://localhost:8000/api/predict/checkpoints')
      .then(res  => setCheckpoints(res.data))
      .catch(err => setCheckpointsError('获取 Checkpoint 列表失败：' + err.message))
      .finally(() => setCheckpointsLoading(false))
  }, [])  // 空依赖 → 只在 App 首次挂载时执行一次

  // 主动刷新 Checkpoint 列表
  const refreshCheckpoints = () => {
    setCheckpointsLoading(true)
    setCheckpointsError(null)
    axios.get('http://localhost:8000/api/predict/checkpoints')
      .then(res  => setCheckpoints(res.data))
      .catch(err => setCheckpointsError('获取 Checkpoint 列表失败：' + err.message))
      .finally(() => setCheckpointsLoading(false))
  }

  // 轮询状态 + 日志，两者合并为一个 interval
  const startPollingStatus = () => {
    if (logPollRef.current) clearInterval(logPollRef.current)
    let logOffset = 0
    logPollRef.current = setInterval(async () => {
      try {
        const [statusRes, logRes] = await Promise.all([
          axios.get('http://localhost:8000/api/train/status'),
          axios.get(`http://localhost:8000/api/train/logs?since=${logOffset}`),
        ])
        setTrainingStatus(statusRes.data)
        const { lines, total } = logRes.data as { lines: string[]; total: number }
        if (lines.length > 0) {
          setTrainLogs(prev => [...prev, ...lines])
          logOffset = total
        }
        if (statusRes.data.status === 'completed' || statusRes.data.status === 'failed') {
          clearInterval(logPollRef.current!)
          logPollRef.current = null
        }
      } catch { /* 网络抖动静默跳过 */ }
    }, 1500)
  }

  // 启动训练函数
  const startTraining = async (config: TrainConfig) => {
    try {
      setIsTrainLoading(true)
      setTrainLogs([])
      const response = await axios.post('http://localhost:8000/api/train/start', config)
      if (response.data.success) {
        setTimeout(startPollingStatus, 800)
      }
    } catch (error) {
      console.error('Error starting training:', error)
    } finally {
      setIsTrainLoading(false)
    }
  }

  return (
    <BrowserRouter>
      <div style={{ minHeight: '100vh', backgroundColor: PAGE_BG, fontFamily: 'sans-serif', color: '#e0e0e0' }}>

        {/* ===== 顶部导航栏（仅登录状态下显示 Tab）===== */}
        {isAuthed && (
          <div style={{ backgroundColor: NAV_BG, display: 'flex', alignItems: 'flex-end', padding: '0 40px' }}>

            {/* 左侧标题 */}
            <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '18px', padding: '16px 24px 16px 0', marginRight: '16px' }}>
              时序预测平台
            </div>

            {/* Tab 导航 */}
            {TABS.map(tab => (
              <NavLink
                key={tab.path}
                to={tab.path}
                style={({ isActive }) => ({
                  display: 'block',
                  padding: '12px 24px',
                  cursor: 'pointer',
                  fontSize: '15px',
                  fontWeight: isActive ? 'bold' : 'normal',
                  color: isActive ? '#fff' : '#aaa',
                  backgroundColor: isActive ? PAGE_BG : 'transparent',
                  borderRadius: '8px 8px 0 0',
                  borderTop:    isActive ? '1px solid #444' : '1px solid transparent',
                  borderLeft:   isActive ? '1px solid #444' : '1px solid transparent',
                  borderRight:  isActive ? '1px solid #444' : '1px solid transparent',
                  borderBottom: 'none',
                  transition: 'all 0.3s',
                  userSelect: 'none',
                  textDecoration: 'none',
                })}
              >
                {tab.label}
              </NavLink>
            ))}

            {/* 右侧：用户名 + 退出 */}
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px', paddingBottom: '10px' }}>
              <span style={{ fontSize: '13px', color: '#888' }}>
                👤 {username}
              </span>
              <button
                onClick={logout}
                style={{
                  background: 'transparent', border: '1px solid #3a3a55',
                  color: '#777', borderRadius: '6px', padding: '4px 12px',
                  fontSize: '12px', cursor: 'pointer', transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = '#e05050'; e.currentTarget.style.color = '#e08080' }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = '#3a3a55'; e.currentTarget.style.color = '#777' }}
              >退出</button>
            </div>
          </div>
        )}

        {/* ===== 页面内容区域 ===== */}
        <div style={isAuthed ? { padding: '40px 50px' } : {}}>
          <Routes>
            {/* ── 公开路由 ── */}
            <Route path="/login"    element={isAuthed ? <Navigate to="/" replace /> : <LoginPage   onLogin={login}       />} />
            <Route path="/register" element={isAuthed ? <Navigate to="/" replace /> : <RegisterPage onRegister={register} />} />

            {/* ── 受保护路由（未登录跳转 /login）── */}
            <Route element={<ProtectedRoute isAuthed={isAuthed} />}>
              <Route path="/" element={<Navigate to="/train" replace />} />
              <Route path="/train"
                element={<TrainPage trainingStatus={trainingStatus} isLoading={isTrainLoading}
                            onStartTraining={startTraining} trainLogs={trainLogs} onClearLogs={() => setTrainLogs([])} />}
              />
              <Route path="/predict"            element={<PredictPage checkpoints={checkpoints} loading={checkpointsLoading} error={checkpointsError} onRefresh={refreshCheckpoints} />} />
              <Route path="/predict/:folderName" element={<InferencePage />} />
              <Route path="/assistant"           element={<AssistantPage />} />
            </Route>

            {/* ── 兜底重定向 ── */}
            <Route path="*" element={<Navigate to={isAuthed ? '/train' : '/login'} replace />} />
          </Routes>
        </div>

      </div>
    </BrowserRouter>
  )
}

export default App