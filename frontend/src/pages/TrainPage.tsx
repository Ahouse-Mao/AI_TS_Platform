import type { TrainingStatus } from '../types'

// TrainPage 的 props 类型：状态和回调都从父组件 App 传入
// 这种模式叫"状态提升"(Lifting State Up)：把状态放到不会被路由卸载的父组件里，路由切换时状态不丢失
export interface TrainPageProps {
  trainingStatus: TrainingStatus | null
  isLoading: boolean
  onStartTraining: () => void
}

// 模型训练与配置页面（不再自己持有状态，改为从 props 接收）
export function TrainPage({ trainingStatus, isLoading, onStartTraining }: TrainPageProps) {
  return (
    <div>
      <h3 style={{ marginTop: 0 }}>模型训练</h3>
      <button
        onClick={onStartTraining}
        disabled={isLoading || trainingStatus?.status === 'running'}
        style={{
          padding: '12px 24px',
          fontSize: '16px',
          cursor: trainingStatus?.status === 'running' ? 'not-allowed' : 'pointer',
          backgroundColor: trainingStatus?.status === 'running' ? '#d61b1b' : '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '10px',
        }}
      >
        {trainingStatus?.status === 'running' ? '训练中...' : '开始训练'}
      </button>

      {/* 训练状态显示：只有 trainingStatus 不为 null 时才渲染 */}
      {trainingStatus && (
        <div style={{
          marginTop: '20px',
          padding: '10px 20px',
          background:
            trainingStatus.status === 'completed' ? '#d4edda' :
            trainingStatus.status === 'failed'    ? '#f8d7da' : '#fff3cd',
          borderRadius: '10px',
          color: '#333',
          border: `5px solid ${
            trainingStatus.status === 'completed' ? '#a8d5b5' :
            trainingStatus.status === 'failed'    ? '#f0a8ae' : '#ffe08a'
          }`,
        }}>
          <strong>训练状态：</strong> {trainingStatus.status} <br />
          <strong>消息：</strong> {trainingStatus.message} <br />
          {trainingStatus.start_time && (
            <><strong>开始时间：</strong> {trainingStatus.start_time}</>
          )}
        </div>
      )}
    </div>
  )
}
