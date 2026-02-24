import { useState } from 'react'
import axios from 'axios'

function App() {
  // 就像 Python 里的变量，这里用来存后端返回的消息
  // <string> 就是我们之前说的 TS 泛型，明确这个状态存的是字符串
  const [message, setMessage] = useState<string>('等待后端响应...')
  // message 是当前的状态值，setMessage 是用来更新这个状态的函数

  // 这是一个异步函数，用来向后端发请求
  const fetchData = async () => {
    try {
      setMessage('正在请求...')
      // 注意：这里的 8000 是你 FastAPI 的端口
      const response = await axios.get('http://localhost:8000/')
      //等待后端响应, 期间页面不卡死
      //respone的结构{ data: { status: "success", message: "..." }, status: 200, ... }
      
      // 拿到后端返回的 JSON 数据里的 message 字段
      setMessage(response.data.message)
    } catch (error) {
      // 如果上述请求失败了（比如后端没启动，或者跨域问题），就会执行这里的代码
      setMessage('请求失败，请检查后端是否启动或跨域配置！')
      console.error(error)
    }
  }

  return ( // UI返回react, react再渲染到页面上
    <div style={{ padding: '50px', fontFamily: 'sans-serif' }}>
      <h2>我的时序预测平台 - 前后端通信测试</h2>
      
      {/* 画一个按钮，点击就触发 fetchData 函数 */}
      <button 
        onClick={fetchData} 
        style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
      >
        向 FastAPI 发送 GET 请求
      </button>

      {/* 展示后端返回的信息 */}
      <div style={{ marginTop: '20px', padding: '15px', background: '#e71c1c', borderRadius: '5px' }}>
        <strong>后端返回结果：</strong> {message}
      </div>
    </div>
  )
}

export default App