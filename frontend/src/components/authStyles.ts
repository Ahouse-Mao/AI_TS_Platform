// ── 认证页面共用样式常量（与 AuthForm 分离，避免 react-refresh 警告）──
import type React from 'react'

export const authPageStyle: React.CSSProperties = {
  minHeight:      '100vh',
  background:     '#1e1e2e',
  display:        'flex',
  alignItems:     'center',
  justifyContent: 'center',
  fontFamily:     'sans-serif',
  padding:        '24px',
}

export const authCardStyle: React.CSSProperties = {
  width:        '380px',
  maxWidth:     '100%',
  background:   '#252535',
  border:       '1px solid #3a3a55',
  borderRadius: '16px',
  padding:      '40px 36px',
  boxShadow:    '0 20px 60px rgba(0,0,0,0.4)',
}

export const authInputStyle: React.CSSProperties = {
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

/** color: 按钮激活色，登录页传 '#7b8cde'，注册页传 '#2d9e6b' */
export function authSubmitBtnStyle(disabled: boolean, color = '#7b8cde'): React.CSSProperties {
  return {
    width:        '100%',
    padding:      '12px',
    marginTop:    '4px',
    borderRadius: '8px',
    fontSize:     '15px',
    fontWeight:   'bold',
    border:       'none',
    cursor:       disabled ? 'not-allowed' : 'pointer',
    background:   disabled ? '#2a2a3e' : color,
    color:        disabled ? '#555' : '#fff',
    transition:   'background 0.15s',
  }
}
