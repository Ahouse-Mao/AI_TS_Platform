// ===================== 认证状态 Hook =====================
// 封装所有与"当前用户是否登录"相关的状态和操作。
// 使用 localStorage 持久化 Token，页面刷新后自动恢复登录态。

import { useState, useCallback } from 'react'
import { API_BASE } from '../config'
import { lsGet, lsSet } from '../utils/storage'

const SK_TOKEN  = 'ai_ts_token'
const SK_USER   = 'ai_ts_username'
const REQUEST_TIMEOUT_MS = 20000

let axiosInterceptorInstalled = false

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
  const [token,    setToken]    = useState<string | null>(() => lsGet<string | null>(SK_TOKEN, null))
  const [username, setUsername] = useState<string | null>(() => lsGet<string | null>(SK_USER, null))

  // ── 通用请求封装 ──
  const authRequest = useCallback(async (
    endpoint: string,
    username: string,
    password: string,
  ): Promise<string | null> => {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS)
    try {
      const res  = await fetch(`${API_BASE}/api/auth/${endpoint}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ username, password }),
        signal:  ctrl.signal,
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
      lsSet(SK_TOKEN, access_token)
      lsSet(SK_USER,  uname)
      return null   // null = 成功，无错误
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        return '请求超时，后端可能繁忙，请稍后重试'
      }
      return '网络错误：' + (e instanceof Error ? e.message : String(e))
    } finally {
      clearTimeout(timer)
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
    if (axiosInterceptorInstalled) return
    axiosInterceptorInstalled = true

    axios.defaults.timeout = REQUEST_TIMEOUT_MS

    axios.interceptors.request.use(config => {
      const t = getToken()
      if (t) config.headers.Authorization = `Bearer ${t}`
      return config
    })

    axios.interceptors.response.use(
      response => response,
      error => {
        if (error?.code === 'ECONNABORTED') {
          return Promise.reject(new Error('请求超时，后端可能繁忙，请稍后重试'))
        }

        const status = error?.response?.status
        const detail = error?.response?.data?.detail
        if (status === 503) {
          return Promise.reject(new Error(detail || '后端繁忙，请稍后重试'))
        }

        return Promise.reject(error)
      },
    )
  })
}
