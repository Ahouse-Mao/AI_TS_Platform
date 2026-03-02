import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import { API_BASE } from '../config'

const EMBEDDING_MODELS = [
  { value: 'BAAI/bge-small-zh-v1.5', label: 'BGE Small ZH（轻量，512维）' },
  { value: 'BAAI/bge-base-zh-v1.5',  label: 'BGE Base ZH（推荐，768维）' },
  { value: 'BAAI/bge-large-zh-v1.5', label: 'BGE Large ZH（高精度，1024维）' },
]

type BuildStatus = 'idle' | 'running' | 'completed' | 'failed'

interface RAGBuildState {
  status:      BuildStatus
  message:     string
  doc_count?:  number
  start_time?: string
}

const STATUS_LABEL: Record<BuildStatus, string> = {
  idle:      '空闲',
  running:   '构建中',
  completed: '已完成',
  failed:    '失败',
}

const STATUS_COLOR: Record<BuildStatus, string> = {
  idle:      '#888',
  running:   '#c0a0f0',
  completed: '#2d9e6b',
  failed:    '#e05050',
}

export function RAGPage() {
  const [selectedModel, setSelectedModel] = useState<string>('BAAI/bge-base-zh-v1.5')
  const [buildStatus,   setBuildStatus]   = useState<RAGBuildState | null>(null)
  const [isBuilding,    setIsBuilding]    = useState<boolean>(false)
  const [buildLogs,     setBuildLogs]     = useState<string[]>([])
  const [isClearing,    setIsClearing]    = useState<boolean>(false)
  const logEndRef = useRef<HTMLDivElement>(null)

  // useEffect 用于监听 buildLogs 变化，自动滚动日志到底部
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [buildLogs])

  async function handleStartBuild() {
    setIsBuilding(true)
    setBuildLogs([])
    setBuildStatus({ status: 'running', message: '正在构建索引向量...' })

    try {
      await axios.post(`${API_BASE}/api/rag/build`, { embedding_model: selectedModel })

      // 轮询构建状态，任务在后台执行，前端每秒查询一次
      const poll = setInterval(async () => {
        try {
          const { data } = await axios.get<RAGBuildState & { logs: string[] }>(`${API_BASE}/api/rag/status`)
          setBuildLogs(data.logs ?? [])
          setBuildStatus({ status: data.status, message: data.message, doc_count: data.doc_count, start_time: data.start_time })
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(poll)
            setIsBuilding(false)
          }
        } catch {
          clearInterval(poll)
          setIsBuilding(false)
          setBuildStatus({ status: 'failed', message: '轮询状态失败，请检查后端服务' })
        }
      }, 1000)
    } catch {
      setIsBuilding(false)
      setBuildStatus({ status: 'failed', message: '请求失败，请检查后端服务' })
    }
  }

  async function handleClearIndex() {
    if (!window.confirm('确定要清除向量库吗？此操作不可恢复，需要重新构建。')) return
    setIsClearing(true)
    try {
      const { data } = await axios.delete<{ success: boolean; message: string }>(`${API_BASE}/api/rag/index`)
      setBuildLogs(prev => [...prev, `[清除] ${data.message}`])
      setBuildStatus(null)
    } catch {
      setBuildLogs(prev => [...prev, '[清除] 请求失败，请检查后端服务'])
    } finally {
      setIsClearing(false)
    }
  }

  return (
    <div>

      {/* 嵌入模型选择 */}
      <div style={{ background: '#252535', border: '1px solid #3a3a55', borderRadius: '12px', padding: '20px 24px', marginBottom: '20px' }}>
        <h3 style={{ margin: '0 0 16px', fontSize: '13px', color: '#a0a8d0', letterSpacing: '0.8px', textTransform: 'uppercase' }}>
          嵌入模型
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '12px' }}>
          {EMBEDDING_MODELS.map(m => (
            <div
              key={m.value}
              onClick={() => setSelectedModel(m.value)}
              style={{
                background:   selectedModel === m.value ? '#1e2540' : '#1a1a2e',
                border:       `1.5px solid ${selectedModel === m.value ? '#7b5ea7' : '#3a3a55'}`,
                borderRadius: '10px',
                padding:      '12px 16px',
                cursor:       'pointer',
                transition:   'all 0.15s',
              }}
            >
              <div style={{ fontWeight: 'bold', fontSize: '13px', color: '#ccc' }}>{m.label}</div>
              <div style={{ fontSize: '11px', color: '#555', marginTop: '4px', fontFamily: 'monospace' }}>{m.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 操作按钮行 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
        <button
          onClick={handleStartBuild}
          disabled={isBuilding || isClearing}
          style={{
            background:   (isBuilding || isClearing) ? '#2a2a3e' : '#7b5ea7',
            color:        (isBuilding || isClearing) ? '#666'    : '#fff',
            border:       'none',
            borderRadius: '8px',
            padding:      '10px 28px',
            fontSize:     '14px',
            fontWeight:   'bold',
            cursor:       (isBuilding || isClearing) ? 'not-allowed' : 'pointer',
            transition:   'background 0.2s',
          }}
        >
          {isBuilding ? '⏳ 构建中...' : '▶ 开始构建'}
        </button>

        <button
          onClick={handleClearIndex}
          disabled={isBuilding || isClearing}
          style={{
            background:   (isBuilding || isClearing) ? '#2a2a3e' : 'transparent',
            color:        (isBuilding || isClearing) ? '#555'    : '#e05050',
            border:       `1px solid ${(isBuilding || isClearing) ? '#3a3a55' : '#e05050'}`,
            borderRadius: '8px',
            padding:      '10px 20px',
            fontSize:     '14px',
            fontWeight:   'bold',
            cursor:       (isBuilding || isClearing) ? 'not-allowed' : 'pointer',
            transition:   'all 0.2s',
          }}
        >
          {isClearing ? '⏳ 清除中...' : '🗑 清除向量库'}
        </button>

        <span style={{ fontSize: '12px', color: '#555' }}>
          {isBuilding ? `正在使用 ${selectedModel} 构建向量索引...` : `将使用 ${selectedModel}`}
        </span>
      </div>

      {/* 日志区域 */}
      <div style={{ background: '#0d0d0d', border: '1px solid #2a2a2a', borderRadius: '10px', overflow: 'hidden', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 14px', background: '#1a1a1a', borderBottom: '1px solid #2a2a2a' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ff5f57', display: 'inline-block' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#febc2e', display: 'inline-block' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#28c840', display: 'inline-block' }} />
            <span style={{ fontSize: '12px', color: '#555', marginLeft: '8px', fontFamily: 'monospace' }}>rag build output</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {isBuilding && (
              <span style={{ fontSize: '11px', color: '#c0a0f0', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#c0a0f0', display: 'inline-block' }} />
                LIVE
              </span>
            )}
            <span style={{ fontSize: '11px', color: '#444', fontFamily: 'monospace' }}>{buildLogs.length} lines</span>
            <button
              onClick={() => setBuildLogs([])}
              style={{ background: 'transparent', border: '1px solid #2a2a2a', color: '#555', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', cursor: 'pointer' }}
            >
              clear
            </button>
          </div>
        </div>
        <div style={{ height: '260px', overflowY: 'auto', padding: '12px 16px', fontFamily: '"Cascadia Code", Consolas, monospace', fontSize: '12px', lineHeight: '1.7', color: '#c8c8c8' }}>
          {buildLogs.length === 0
            ? <span style={{ color: '#333' }}>// 等待构建开始</span>
            : buildLogs.map((line, i) => (
                <div key={i} style={{
                  color:
                    (line.includes('✅') || line.includes('完成') || line.includes('成功')) ? '#50fa7b'
                    : (line.includes('❌') || line.toLowerCase().includes('error'))         ? '#ff5555'
                    : line.startsWith('[')                                                  ? '#6272a4'
                    : '#c8c8c8',
                  whiteSpace: 'pre-wrap',
                  wordBreak:  'break-all',
                }}>
                  {line}
                </div>
              ))
          }
          <div ref={logEndRef} />
        </div>
      </div>

      {/* 构建状态卡片 */}
      {buildStatus && (
        <div style={{
          padding:      '14px 20px',
          background:   buildStatus.status === 'completed' ? '#1a3a2a' : buildStatus.status === 'failed' ? '#3a1a1a' : buildStatus.status === 'running' ? '#1e1a40' : '#252535',
          borderRadius: '10px',
          border:       `1px solid ${buildStatus.status === 'completed' ? '#2d9e6b' : buildStatus.status === 'failed' ? '#e05050' : buildStatus.status === 'running' ? '#7b5ea7' : '#3a3a55'}`,
          fontSize:     '13px',
          color:        '#ccc',
          lineHeight:   '1.8',
        }}>
          <div>
            <span style={{ color: '#888' }}>状态：</span>
            <span style={{ fontWeight: 'bold', color: STATUS_COLOR[buildStatus.status] }}>
              {STATUS_LABEL[buildStatus.status]}
            </span>
          </div>
          <div><span style={{ color: '#888' }}>消息：</span>{buildStatus.message}</div>
          {buildStatus.doc_count !== undefined && (
            <div><span style={{ color: '#888' }}>文档数量：</span>{buildStatus.doc_count} 条</div>
          )}
          {buildStatus.start_time && (
            <div><span style={{ color: '#888' }}>开始时间：</span>{buildStatus.start_time}</div>
          )}
        </div>
      )}

    </div>
  )
}
