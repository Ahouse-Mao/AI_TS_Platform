// ===================== 路由守卫 =====================
// 未登录时重定向到 /login，登录后放行子路由。
// 用法：
//   <Route element={<ProtectedRoute isAuthed={isAuthed} />}>
//     <Route path="/train" element={<TrainPage ... />} />
//     ...
//   </Route>

import { Navigate, Outlet } from 'react-router-dom'

interface Props {
  isAuthed: boolean
}

export function ProtectedRoute({ isAuthed }: Props) {
  // Outlet 渲染嵌套子路由；未登录时跳转 /login
  // replace: 不留历史记录，防止返回键绕过认证
  return isAuthed ? <Outlet /> : <Navigate to="/login" replace />
}
