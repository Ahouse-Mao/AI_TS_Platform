import { useState } from 'react' // React 的 useState Hook, 用于在函数组件中管理状态
import axios from 'axios' // HTTP 请求库，比原生 fetch 更简洁, 用于和后端通信

// 训练状态的类型定义
interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  message: string
  start_time: string | null
  conda_env: string
}

// 状态变量
function App() {
  const [message, setMessage] = useState<string>('等待后端相应...')
  // 定义一个状态变量 message, 存储后端返回的文字消息, 初始值为 '等待后端响应...'
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  // 存储训练状态对象
  // { status: "running", message: "训练中...", start_time: "..." }
  // 初始值：null（还没查询过）
  const [isLoading, setIsLoading] = useState<boolean>(false)
  // 存储按钮是否在请求中的标志, 防止用户疯狂点击提交按钮重复提交

  // 原有的测试函数
  const fetchData = async () => {
    try {
      setMessage('正在请求...')
      const response = await axios.get('http://localhost:8000/')
      setMessage(response.data.message)
    } catch (error) {
      setMessage('请求失败，请检查后端是否启动或跨域配置！')
      console.error(error)
    }
  }

  // 启动训练函数
  const startTraining = async () => {
    try {
      setIsLoading(true) // 开始请求，设置加载状态为 true

      const response = await axios.post('http://localhost:8000/api/train/start')
      // 向后端发 POST 请求
      // await：等待后端响应，期间页面不卡死
      // response.data 就是后端 return 的那个字典：
      // { "success": true, "message": "训练已启动..." }

      if (response.data.success) {
        setMessage(response.data.message) // 更新 message 状态，显示后端返回的消息
        startPollingStatus() // 启动轮询查询训练状态
      } else {
        setMessage(response.data.message) // 显示错误消息
      }
    } catch (error) {
      setMessage('Failed to start training:' + error) // 请求失败，显示错误消息
      console.error('Error starting training:', error)
    } finally {
      setIsLoading(false) // 无论成功失败，请求结束后都恢复按钮状态
    }
  }

  // 轮询状态函数
  const startPollingStatus = () => {
    const interval = setInterval(async () => {
      // setInterval 是浏览器内置函数
      // 作用：每隔指定毫秒数，重复执行一次里面的函数
      // 返回值 interval 是这个定时器的"ID"，用于后面停止它

      const response = await axios.get('http://localhost:8000/api/train/status')
      // 每次执行都向后端查询训练状态

      setTrainingStatus(response.data) // 用最新状态更新变量, react自动重新渲染页面

      if (response.data.status === 'completed' || response.data.status === 'failed') {
        clearInterval(interval) // 停止轮询
      }
    }, 2000) // 每隔 2 秒查询一次状态
  }

  // UI
  return (
    <div style={{ padding: '50px', fontFamily: 'sans-serif' }}>
      <h2>时序预测平台</h2>
      {/* 新增：训练按钮 */}
      <h3>模型训练</h3>
      <button 
        onClick={startTraining}
        // 点击按钮时调用 startTraining 函数, startTraining()是函数调用, startTraining 是函数引用
        disabled={isLoading || trainingStatus?.status === 'running'}
        // 禁用条件：如果正在请求中（isLoading）或者训练状态是 'running'，则禁用按钮(变灰)，防止重复提交
        style={{ 
          padding: '12px 24px', // 内边距
          fontSize: '16px', 
          cursor: trainingStatus?.status === 'running' ? 'not-allowed' : 'pointer',
          // 指针形状是否在训练中, 是则显示禁止符号, 否则显示手型
          backgroundColor: trainingStatus?.status === 'running' ? '#d61b1b' : '#4CAF50',
          color: 'white', // 字体颜色
          border: 'none', // 无边框
          borderRadius: '10px' // 圆角
        }}
      // 按钮文本根据训练状态动态显示, 如果正在训练则显示 '训练中...', 否则显示 '开始训练'
      >
        {trainingStatus?.status === 'running' ? '训练中...' : '开始训练'} 
      </button>

      {/* 训练状态显示 */}
      {trainingStatus && (
        <div style={{ 
          marginTop: '20px', 
          padding: '10px 20px', 
          background: trainingStatus.status === 'completed' ? '#d4edda' : 
                     trainingStatus.status === 'failed' ? '#f8d7da' : '#fff3cd',
          borderRadius: '10px',
          color: '#333',
          border: `5px solid ${
            trainingStatus.status === 'completed' ? '#a8d5b5' :
            trainingStatus.status === 'failed'    ? '#f0a8ae' : '#ffe08a'
          }`
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

export default App