// ===================== 认证页面共用组件 =====================
// LoginPage 和 RegisterPage 共用同一套表单字段组件。
// 共用样式常量见 authStyles.ts
import type React from 'react'

// ── 表单字段组件（label + input 竖向排列）──
export function AuthField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <span style={{ fontSize: '12px', color: '#888', fontWeight: 'bold' }}>{label}</span>
      {children}
    </label>
  )
}

// ── 错误提示框 ──
export function AuthError({ message }: { message: string }) {
  return (
    <div style={{
      padding: '9px 14px', borderRadius: '8px',
      background: '#3a1a1a', border: '1px solid #7a3030',
      color: '#e08080', fontSize: '13px',
    }}>
      {message}
    </div>
  )
}
