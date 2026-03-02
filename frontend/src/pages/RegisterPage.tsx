// ===================== 注册页面 =====================
import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import type { UseAuth } from '../hooks/useAuth'
import {
  authPageStyle, authCardStyle, authInputStyle, authSubmitBtnStyle,
} from '../components/authStyles'
import { AuthField, AuthError } from '../components/AuthForm'

interface Props {
  onRegister: UseAuth['register']
}

export function RegisterPage({ onRegister }: Props) {
  const navigate              = useNavigate()
  const [username,  setUsername]  = useState('')
  const [password,  setPassword]  = useState('')
  const [password2, setPassword2] = useState('')
  const [error,     setError]     = useState('')
  const [loading,   setLoading]   = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError('')

    if (username.trim().length < 2) { setError('用户名至少 2 个字符'); return }
    if (password.length < 6)        { setError('密码至少 6 位'); return }
    if (password !== password2)     { setError('两次密码不一致'); return }

    setLoading(true)
    const err = await onRegister(username.trim(), password)
    setLoading(false)
    if (err) { setError(err); return }
    navigate('/', { replace: true })
  }

  const canSubmit = username.trim().length >= 2 && password.length >= 6 && password === password2 && !loading

  return (
    <div style={authPageStyle}>
      <div style={authCardStyle}>

        {/* 标题 */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 'bold', color: '#e0e0f0' }}>
            创建账号
          </h1>
          <p style={{ margin: '6px 0 0', fontSize: '13px', color: '#666' }}>时序预测平台</p>
        </div>

        {/* 表单 */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          <AuthField label="用户名">
            <input
              type="text" autoFocus autoComplete="username"
              value={username} onChange={e => setUsername(e.target.value)}
              placeholder="2-50 个字符"
              style={authInputStyle}
              onFocus={e  => { e.currentTarget.style.borderColor = '#7b8cde' }}
              onBlur={e   => { e.currentTarget.style.borderColor = '#3a3a55' }}
            />
          </AuthField>

          <AuthField label="密码">
            <input
              type="password" autoComplete="new-password"
              value={password} onChange={e => setPassword(e.target.value)}
              placeholder="至少 6 位"
              style={authInputStyle}
              onFocus={e  => { e.currentTarget.style.borderColor = '#7b8cde' }}
              onBlur={e   => { e.currentTarget.style.borderColor = '#3a3a55' }}
            />
          </AuthField>

          <AuthField label="确认密码">
            <input
              type="password" autoComplete="new-password"
              value={password2} onChange={e => setPassword2(e.target.value)}
              placeholder="再次输入密码"
              style={{
                ...authInputStyle,
                borderColor: password2 && password !== password2 ? '#e05050' : '#3a3a55',
              }}
              onFocus={e  => { if (!(password2 && password !== password2)) e.currentTarget.style.borderColor = '#7b8cde' }}
              onBlur={e   => { e.currentTarget.style.borderColor = password2 && password !== password2 ? '#e05050' : '#3a3a55' }}
            />
            {password2 && password !== password2 && (
              <span style={{ fontSize: '11px', color: '#e08080', marginTop: '2px' }}>密码不一致</span>
            )}
          </AuthField>

          {error && <AuthError message={error} />}

          <button type="submit" disabled={!canSubmit} style={authSubmitBtnStyle(!canSubmit, '#2d9e6b')}>
            {loading ? '注册中…' : '注 册'}
          </button>
        </form>

        {/* 登录链接 */}
        <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '13px', color: '#666' }}>
          已有账号？{' '}
          <Link to="/login" style={{ color: '#7b8cde', textDecoration: 'none', fontWeight: 'bold' }}>
            返回登录
          </Link>
        </p>
      </div>
    </div>
  )
}

