// ===================== 登录页面 =====================
import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import type { UseAuth } from '../hooks/useAuth'
import {
  authPageStyle, authCardStyle, authInputStyle, authSubmitBtnStyle,
} from '../components/authStyles'
import { AuthField, AuthError } from '../components/AuthForm'

interface Props {
  onLogin: UseAuth['login']
}

export function LoginPage({ onLogin }: Props) {
  const navigate              = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!username.trim() || !password) return
    setLoading(true)
    setError('')
    const err = await onLogin(username.trim(), password)
    setLoading(false)
    if (err) { setError(err); return }
    navigate('/', { replace: true })
  }

  return (
    <div style={authPageStyle}>
      <div style={authCardStyle}>

        {/* 标题 */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 'bold', color: '#e0e0f0' }}>
            时序预测平台
          </h1>
          <p style={{ margin: '6px 0 0', fontSize: '13px', color: '#666' }}>登录以继续</p>
        </div>

        {/* 表单 */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <AuthField label="用户名">
            <input
              type="text" autoFocus autoComplete="username"
              value={username} onChange={e => setUsername(e.target.value)}
              placeholder="请输入用户名"
              style={authInputStyle}
            />
          </AuthField>

          <AuthField label="密码">
            <input
              type="password" autoComplete="current-password"
              value={password} onChange={e => setPassword(e.target.value)}
              placeholder="请输入密码"
              style={authInputStyle}
            />
          </AuthField>

          {error && <AuthError message={error} />}

          <button
            type="submit" disabled={loading || !username.trim() || !password}
            style={authSubmitBtnStyle(loading || !username.trim() || !password)}
          >
            {loading ? '登录中…' : '登 录'}
          </button>
        </form>

        {/* 注册链接 */}
        <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '13px', color: '#666' }}>
          还没有账号？{' '}
          <Link to="/register" style={{ color: '#7b8cde', textDecoration: 'none', fontWeight: 'bold' }}>
            立即注册
          </Link>
        </p>
      </div>
    </div>
  )
}


