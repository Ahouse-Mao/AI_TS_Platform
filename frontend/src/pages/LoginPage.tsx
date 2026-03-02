// ===================== 登录页面 =====================
import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import type { UseAuth } from '../hooks/useAuth'

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
    <div style={pageStyle}>
      <div style={cardStyle}>

        {/* 标题 */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 'bold', color: '#e0e0f0' }}>
            时序预测平台
          </h1>
          <p style={{ margin: '6px 0 0', fontSize: '13px', color: '#666' }}>登录以继续</p>
        </div>

        {/* 表单 */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Field label="用户名">
            <input
              type="text" autoFocus autoComplete="username"
              value={username} onChange={e => setUsername(e.target.value)}
              placeholder="请输入用户名"
              style={inputStyle}
            />
          </Field>

          <Field label="密码">
            <input
              type="password" autoComplete="current-password"
              value={password} onChange={e => setPassword(e.target.value)}
              placeholder="请输入密码"
              style={inputStyle}
            />
          </Field>

          {error && (
            <div style={{
              padding: '9px 14px', borderRadius: '8px',
              background: '#3a1a1a', border: '1px solid #7a3030',
              color: '#e08080', fontSize: '13px',
            }}>{error}</div>
          )}

          <button
            type="submit" disabled={loading || !username.trim() || !password}
            style={submitBtnStyle(loading || !username.trim() || !password)}
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

/* ── 子组件 ── */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <span style={{ fontSize: '12px', color: '#888', fontWeight: 'bold' }}>{label}</span>
      {children}
    </label>
  )
}

/* ── 共用样式 ── */
const pageStyle: React.CSSProperties = {
  minHeight:      '100vh',
  background:     '#1e1e2e',
  display:        'flex',
  alignItems:     'center',
  justifyContent: 'center',
  fontFamily:     'sans-serif',
  padding:        '24px',
}

const cardStyle: React.CSSProperties = {
  width:        '380px',
  maxWidth:     '100%',
  background:   '#252535',
  border:       '1px solid #3a3a55',
  borderRadius: '16px',
  padding:      '40px 36px',
  boxShadow:    '0 20px 60px rgba(0,0,0,0.4)',
}

const inputStyle: React.CSSProperties = {
  width:        '100%',
  background:   '#1a1a2e',
  border:       '1px solid #3a3a55',
  borderRadius: '8px',
  color:        '#e0e0e0',
  padding:      '11px 14px',
  fontSize:     '14px',
  outline:      'none',
  boxSizing:    'border-box',
  transition:   'border-color 0.15s',
}

function submitBtnStyle(disabled: boolean): React.CSSProperties {
  return {
    width:        '100%',
    padding:      '12px',
    marginTop:    '4px',
    borderRadius: '8px',
    fontSize:     '15px',
    fontWeight:   'bold',
    border:       'none',
    cursor:       disabled ? 'not-allowed' : 'pointer',
    background:   disabled ? '#2a2a3e' : '#7b8cde',
    color:        disabled ? '#555' : '#fff',
    transition:   'background 0.15s',
  }
}
