// ===================== 复用组件格式 =====================

// 单行参数展示小组件
// label: 左侧灰色标签，value: 右侧白色值
export function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', gap: '6px' }}>
      <span style={{ color: '#888', whiteSpace: 'nowrap' }}>{label}:</span>
      <span style={{ color: '#e0e0e0', fontWeight: 500 }}>{value}</span>
    </div>
  )
}
