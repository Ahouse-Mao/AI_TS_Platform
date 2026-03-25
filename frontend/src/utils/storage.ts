// ===================== 浏览器本地存储工具函数 =====================
// 统一封装 localStorage / sessionStorage 的 JSON 读写, 
// 处理 JSON 解析异常和存储配额溢出。

// ── localStorage ──
export function lsGet<T>(key: string, fallback: T): T {
    try {
        const raw = localStorage.getItem(key) // 从 localStorage 获取原始字符串, 浏览器API, 返回string | null
        return raw ? (JSON.parse(raw) as T) : fallback // 解析 JSON, 如果 raw 不为 null 则解析为类型 T 并返回, 否则返回 fallback
    } catch {
        return fallback
    }
    }

export function lsSet(key: string, value: unknown): void { // unknown 表示 value 可以是任意类型, 比any更安全, 要求调用者明确类型
  try {
    localStorage.setItem(key, JSON.stringify(value)) // 将 value 转换为 JSON 字符串并存储到 localStorage 中, 浏览器API
  } catch { /* quota exceeded 静默忽略 */ }
}

// ── sessionStorage ──
// sessionStorage 的接口和 localStorage 基本相同, 但数据仅在当前标签页有效, 关闭标签页即丢失
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
