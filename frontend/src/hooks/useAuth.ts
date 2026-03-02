// ===================== 认证状态 Hook =====================
// 封装所有与"当前用户是否登录"相关的状态和操作。
// 使用 localStorage 持久化 Token，页面刷新后自动恢复登录态。

import { useState, useCallback } from 'react'

const API       = 'http://localhost:8000'
const SK_TOKEN  = 'ai_ts_token'
const SK_USER   = 'ai_ts_username'

export interface AuthState {
  token:    string | null
  username: string | null
  isAuthed: boolean
}

export interface UseAuth extends AuthState {
  login:    (username: string, password: string) => Promise<string | null>  // 返回错误信息或 null
  register: (username: string, password: string) => Promise<string | null>
  logout:   () => void
}

export function useAuth(): UseAuth {
  const [token,    setToken]    = useState<string | null>(() => localStorage.getItem(SK_TOKEN))
  const [username, setUsername] = useState<string | null>(() => localStorage.getItem(SK_USER))

  // ── 通用请求封装 ──
  const authRequest = useCallback(async (
    endpoint: string,
    username: string,
    password: string,
  ): Promise<string | null> => {
    try {
      const res  = await fetch(`${API}/api/auth/${endpoint}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ username, password }),
      })
      const data = await res.json()

      if (!res.ok) {
        // FastAPI 的业务错误格式：{ detail: "..." }
        return data.detail ?? `请求失败（${res.status}）`
      }

      // 保存 Token 到 state + localStorage
      const { access_token, username: uname } = data as { access_token: string; username: string }
      setToken(access_token)
      setUsername(uname)
      localStorage.setItem(SK_TOKEN,  access_token)
      localStorage.setItem(SK_USER,   uname)
      return null   // null = 成功，无错误
    } catch (e) {
      return '网络错误：' + (e instanceof Error ? e.message : String(e))
    }
  }, [])

  const login    = useCallback((u: string, p: string) => authRequest('login',    u, p), [authRequest])
  const register = useCallback((u: string, p: string) => authRequest('register', u, p), [authRequest])

  const logout = useCallback(() => {
    setToken(null)
    setUsername(null)
    localStorage.removeItem(SK_TOKEN)
    localStorage.removeItem(SK_USER)
  }, [])

  return {
    token,
    username,
    isAuthed: !!token,
    login,
    register,
    logout,
  }
}

// ── axios 拦截器工具：为所有请求自动附加 Bearer Token ──
// 在 App.tsx 或 main.tsx 中调用一次即可：
//
//   import axios from 'axios'
//   import { setupAxiosAuth } from './hooks/useAuth'
//   setupAxiosAuth(() => localStorage.getItem('ai_ts_token'))
//
export function setupAxiosAuth(getToken: () => string | null) {
  // 动态导入避免循环依赖
  import('axios').then(({ default: axios }) => {
    axios.interceptors.request.use(config => {
      const t = getToken()
      if (t) config.headers.Authorization = `Bearer ${t}`
      return config
    })
  })
}
