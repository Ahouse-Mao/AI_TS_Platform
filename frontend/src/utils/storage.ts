// ===================== 本地存储工具函数 =====================
// 统一封装 localStorage / sessionStorage 的 JSON 读写，
// 处理 JSON 解析异常和存储配额溢出。

// ── localStorage ──
export function lsGet<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

export function lsSet(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch { /* quota exceeded 静默忽略 */ }
}

// ── sessionStorage ──
export function ssGet<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

export function ssSet(key: string, value: unknown): void {
  try {
    sessionStorage.setItem(key, JSON.stringify(value))
  } catch { /* quota exceeded 静默忽略 */ }
}
