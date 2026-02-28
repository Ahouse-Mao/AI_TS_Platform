import { useState } from 'react'
import axios from 'axios'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import type { TrainingStatus } from './types'
import { TrainPage }      from './pages/TrainPage'
import { PredictPage }    from './pages/PredictPage'
import { AssistantPage }  from './pages/AssistantPage'
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

  // 启动训练函数
  const startTraining = async () => {
    try {
      setIsTrainLoading(true)
      const response = await axios.post('http://localhost:8000/api/train/start')
      if (response.data.success) {
        startPollingStatus()
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
            <Route path="/train"     element={<TrainPage trainingStatus={trainingStatus} isLoading={isTrainLoading} onStartTraining={startTraining} />} />
            <Route path="/predict"   element={<PredictPage />} />
            <Route path="/assistant" element={<AssistantPage />} />
          </Routes>
        </div>

      </div>
    </BrowserRouter>
  )
}

export default App