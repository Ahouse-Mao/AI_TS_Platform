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
// BrowserRouter : 路由容器，监听浏览器地址栏变化
// Routes        : 路由规则集合，匹配当前路径并渲染对应组件
// Route         : 单条路由规则，path → element
// NavLink       : 带激活状态的链接，比普通 <a> 多一个 isActive 回调
// Navigate      : 重定向组件，访问 / 时自动跳转到 /train

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
  // ── 训练状态提升到 App ──
  // App 组件在整个应用生命周期内只挂载一次，路由切换只影响 <Routes> 内部的子组件
  // 所以放在 App 里的 state 不会因为切换 Tab 而丢失
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
        // 并发拉取状态和增量日志
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

  // 启动训练函数（接受训练配置参数，传递给后端）
  const startTraining = async (config: TrainConfig) => {
    try {
      setIsTrainLoading(true)
      setTrainLogs([])   // 清空上次日志
      const response = await axios.post('http://localhost:8000/api/train/start', config)
      if (response.data.success) {
        setTimeout(startPollingStatus, 800)  // 延迟 800ms 等后端写入第一条日志
      }
    } catch (error) {
      console.error('Error starting training:', error)
    } finally {
      setIsTrainLoading(false)
    }
  }

  return (
    <BrowserRouter>
      {/* BrowserRouter 包裹整个应用，使内部组件能访问路由上下文 */}
      <div style={{ minHeight: '100vh', backgroundColor: PAGE_BG, fontFamily: 'sans-serif', color: '#e0e0e0' }}>

        {/* ===== 顶部导航栏 ===== */}
        <div style={{ backgroundColor: NAV_BG, display: 'flex', alignItems: 'flex-end', padding: '0 40px' }}>

          {/* 左侧标题 */}
          <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '18px', padding: '16px 24px 16px 0', marginRight: '16px' }}>
            时序预测平台
          </div>

          {/* Tab 导航链接列表 */}
          {TABS.map(tab => (
            <NavLink
              key={tab.path}
              to={tab.path}
              // NavLink 的 style 接收一个函数，参数由 react-router 自动注入：
              // isActive: 当前 URL 是否匹配 to 路径，true/false
              // 以前需要手动维护 activeTab state，现在路由自动判断
              style={({ isActive }) => ({
                display: 'block',         // NavLink 本质是 <a>，默认 inline，改成 block 让 padding 生效
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
                textDecoration: 'none',   // 去掉 <a> 标签默认的下划线
              })}
            >
              {tab.label}
            </NavLink>
          ))}
        </div>

        {/* ===== 页面内容区域 ===== */}
        <div style={{ padding: '40px 50px' }}>
          <Routes>
            {/* 访问根路径 / 时自动重定向到 /train */}
            {/* replace: 重定向不留历史记录，避免按返回键又跳回 / */}
            <Route path="/" element={<Navigate to="/train" replace />} />
            <Route path="/train"     element={<TrainPage trainingStatus={trainingStatus} isLoading={isTrainLoading} onStartTraining={startTraining} trainLogs={trainLogs} onClearLogs={() => setTrainLogs([])} />} />
            <Route path="/predict"               element={<PredictPage checkpoints={checkpoints} loading={checkpointsLoading} error={checkpointsError} onRefresh={refreshCheckpoints} />} />
            <Route path="/predict/:folderName"    element={<InferencePage />} />
            <Route path="/assistant" element={<AssistantPage />} />
          </Routes>
        </div>

      </div>
    </BrowserRouter>
  )
}

export default App