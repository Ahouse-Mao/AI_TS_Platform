import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import { API_BASE } from '../config'

const EMBEDDING_MODELS = [
  { value: 'BAAI/bge-small-zh-v1.5', label: 'BGE Small ZH（轻量，512维）' },
  { value: 'BAAI/bge-base-zh-v1.5',  label: 'BGE Base ZH（推荐，768维）' },
  { value: 'BAAI/bge-large-zh-v1.5', label: 'BGE Large ZH（高精度，1024维）' },
]

const ASSISTANT_MODELS = [
  { value: 'qwen3.5-flash',     label: 'Qwen3.5 Flash' },
  { value: 'qwen3.5-plus',      label: 'Qwen3.5 Plus' },
  { value: 'gpt-4o',            label: 'GPT-4o' },
  { value: 'gpt-4o-mini',       label: 'GPT-4o Mini' },
  { value: 'gpt-3.5-turbo',     label: 'GPT-3.5 Turbo' },
  { value: 'deepseek-chat',     label: 'DeepSeek Chat' },
  { value: 'deepseek-reasoner', label: 'DeepSeek R1' },
]

type BuildStatus = 'idle' | 'running' | 'completed' | 'failed'

type EvalMode = 'classic' | 'struct' | 'both'
type EvalMetricView = 'recall_focused' | 'first_hit_focused'

interface RAGBuildState {
  status:      BuildStatus
  message:     string
  doc_count?:  number
  start_time?: string
  use_struct_rag?: boolean
}

interface EvalSummary {
  samples: number
  macro_precision_at_k: number
  macro_recall_at_k: number
  macro_precision_at_1?: number
  macro_hit_at_1?: number
  macro_mrr?: number
}

interface EvalReport {
  samples_file: string
  embedding_model: string
  mode: EvalMode
  k_override: number | null
  summary: Record<string, EvalSummary>
}

interface RagasReport {
  mode: 'classic' | 'struct'
  samples: number
  top_k: number
  assistant_url: string
  generator_model: string
  judge_model: string
  judge_embedding_model: string
  summary: Record<string, number>
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
  const [useStructRag,  setUseStructRag]  = useState<boolean>(false)
  const [evalMode,      setEvalMode]      = useState<EvalMode>('struct')
  const [evalMetricView, setEvalMetricView] = useState<EvalMetricView>('recall_focused')
  const [evalTopK,      setEvalTopK]      = useState<string>('')
  const [isEvaluating,  setIsEvaluating]  = useState<boolean>(false)
  const [evalError,     setEvalError]     = useState<string>('')
  const [evalReport,    setEvalReport]    = useState<EvalReport | null>(null)
  const [ragasModel, setRagasModel] = useState<string>('qwen3.5-flash')
  const [ragasJudgeModel, setRagasJudgeModel] = useState<string>('qwen3.5-plus')
  const [ragasJudgeEmbeddingModel, setRagasJudgeEmbeddingModel] = useState<string>('text-embedding-v4')
  const [ragasUseStruct, setRagasUseStruct] = useState<boolean>(true)
  const [ragasTopK, setRagasTopK] = useState<number>(3)
  const [ragasSamplesFile, setRagasSamplesFile] = useState<string>('RAG/eval/ragas_eval_samples.json')
  const [ragasOutFile, setRagasOutFile] = useState<string>('RAG/eval/ragas_report.json')
  const [isRagasEvaluating, setIsRagasEvaluating] = useState<boolean>(false)
  const [ragasError, setRagasError] = useState<string>('')
  const [ragasReport, setRagasReport] = useState<RagasReport | null>(null)
  const ragasPollRef = useRef<number | null>(null)
  const logEndRef = useRef<HTMLDivElement>(null)

  const assistantModelOptions = useMemo(() => {
    const defaultOptions = ASSISTANT_MODELS.map(m => ({ value: m.value, label: m.label }))
    try {
      const raw = localStorage.getItem('ai_ts_fetched_models')
      const parsed = raw ? JSON.parse(raw) : []
      if (Array.isArray(parsed) && parsed.length > 0) {
        const merged = [...defaultOptions]
        const exists = new Set(merged.map(m => m.value))
        for (const m of parsed) {
          if (typeof m === 'string' && m && !exists.has(m)) {
            merged.push({ value: m, label: m })
            exists.add(m)
          }
        }
        return merged
      }
    } catch {
      // ignore and fallback to default options
    }
    return defaultOptions
  }, [])

  const formatMetric = (v?: number) => (v ?? 0).toFixed(4)
  const ragasLabelStyle = { color: '#a8b2e6', fontSize: '12px', marginBottom: '4px', lineHeight: '1.35' }
  const ragasHintStyle = { color: '#6973a8', fontSize: '11px', marginTop: '4px', lineHeight: '1.45', minHeight: '16px' }
  const ragasInputStyle = { width: '100%', background: '#17182a', color: '#d9ddff', border: '1px solid #3c3f66', borderRadius: '8px', padding: '8px 10px', fontSize: '12px', boxSizing: 'border-box' as const }
  const ragasFieldStyle = { display: 'flex', flexDirection: 'column' as const, minHeight: '98px' }

  // useEffect 用于监听 buildLogs 变化，自动滚动日志到底部
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [buildLogs])

  // 兼容旧配置：若仍是 OpenAI 默认 embedding，自动切到阿里云可用模型。
  useEffect(() => {
    if (ragasJudgeEmbeddingModel === 'text-embedding-3-small') {
      setRagasJudgeEmbeddingModel('text-embedding-v4')
    }
  }, [ragasJudgeEmbeddingModel])

  useEffect(() => {
    return () => {
      if (ragasPollRef.current !== null) {
        window.clearInterval(ragasPollRef.current)
      }
    }
  }, [])

  async function handleStartBuild() {
    setIsBuilding(true)
    setBuildLogs([])
    setBuildStatus({ status: 'running', message: '正在构建索引向量...' })

    try {
      await axios.post(`${API_BASE}/api/rag/build`, {
        embedding_model: selectedModel,
        use_struct_rag: useStructRag,
      })

      // 轮询构建状态，任务在后台执行，前端每秒查询一次
      const poll = setInterval(async () => {
        try {
          const { data } = await axios.get<RAGBuildState & { logs: string[] }>(`${API_BASE}/api/rag/status`)
          setBuildLogs(data.logs ?? [])
          setBuildStatus({
            status: data.status,
            message: data.message,
            doc_count: data.doc_count,
            start_time: data.start_time,
            use_struct_rag: data.use_struct_rag,
          })
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
    const modeText = useStructRag ? '结构化索引' : '普通索引'
    if (!window.confirm(`确定要清除${modeText}吗？此操作不可恢复，需要重新构建。`)) return
    setIsClearing(true)
    try {
      const { data } = await axios.delete<{ success: boolean; message: string }>(`${API_BASE}/api/rag/index`, {
        params: { use_struct_rag: useStructRag },
      })
      setBuildLogs(prev => [...prev, `[清除-${modeText}] ${data.message}`])
      setBuildStatus({
        status: data.success ? 'idle' : 'failed',
        message: `${modeText}${data.success ? '清除完成' : '清除失败'}：${data.message}`,
        use_struct_rag: useStructRag,
      })
    } catch {
      setBuildLogs(prev => [...prev, '[清除] 请求失败，请检查后端服务'])
      setBuildStatus({
        status: 'failed',
        message: '清除请求失败，请检查后端服务',
        use_struct_rag: useStructRag,
      })
    } finally {
      setIsClearing(false)
    }
  }

  async function handleClearAllIndexes() {
    if (!window.confirm('确定要清除普通索引和结构化索引吗？此操作不可恢复，需要重新构建。')) return
    setIsClearing(true)
    try {
      const [normalRes, structRes] = await Promise.all([
        axios.delete<{ success: boolean; message: string }>(`${API_BASE}/api/rag/index`, { params: { use_struct_rag: false } }),
        axios.delete<{ success: boolean; message: string }>(`${API_BASE}/api/rag/index`, { params: { use_struct_rag: true } }),
      ])

      setBuildLogs(prev => [
        ...prev,
        `[清除-普通索引] ${normalRes.data.message}`,
        `[清除-结构化索引] ${structRes.data.message}`,
      ])

      const allSuccess = normalRes.data.success && structRes.data.success
      setBuildStatus({
        status: allSuccess ? 'idle' : 'failed',
        message: allSuccess ? '普通索引与结构化索引均已清除' : '部分索引清除失败，请查看日志',
      })
    } catch {
      setBuildLogs(prev => [...prev, '[清除-全部] 请求失败，请检查后端服务'])
      setBuildStatus({ status: 'failed', message: '一键清除失败，请检查后端服务' })
    } finally {
      setIsClearing(false)
    }
  }

  async function handleRunEvaluation() {
    setIsEvaluating(true)
    setEvalError('')
    try {
      const { data } = await axios.post<{
        success: boolean
        message: string
        report?: EvalReport
      }>(`${API_BASE}/api/rag/eval/run`, {
        mode: evalMode,
        k_override: evalTopK ? Number(evalTopK) : null,
        embedding_model: selectedModel,
        samples_file: 'RAG/eval/rag_eval_samples_25.json',
        out_file: 'RAG/eval/rag_eval_report_latest.json',
      })

      if (!data.success || !data.report) {
        setEvalError(data.message || '评测失败')
        return
      }

      setEvalReport(data.report)
    } catch (e: unknown) {
      setEvalError('评测请求失败：' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setIsEvaluating(false)
    }
  }

  async function handleRunRagasEvaluation() {
    setIsRagasEvaluating(true)
    setRagasError('')
    setRagasReport(null)
    try {
      let apiKey = ''
      let baseUrl = 'https://api.openai.com/v1'
      try {
        const raw = localStorage.getItem('ai_ts_settings')
        const parsed = raw ? JSON.parse(raw) : {}
        apiKey = parsed?.apiKey ?? ''
        baseUrl = parsed?.baseUrl ?? baseUrl
      } catch {
        // keep defaults
      }

      if (!apiKey.trim()) {
        setRagasError('请先在 AI 助手页面的“设置”里配置 API Key')
        return
      }

      const { data } = await axios.post<{
        success: boolean
        message: string
      }>(`${API_BASE}/api/rag/eval/ragas/run`, {
        assistant_url: `${API_BASE}/api/assistant/chat`,
        api_key: apiKey,
        base_url: baseUrl,
        model: ragasModel,
        judge_model: ragasJudgeModel,
        judge_embedding_model: ragasJudgeEmbeddingModel,
        use_struct_rag: ragasUseStruct,
        top_k: ragasTopK,
        embedding_model: selectedModel,
        samples_file: ragasSamplesFile,
        out_file: ragasOutFile,
      })

      if (!data.success) {
        setRagasError(data.message || 'Ragas 评测失败')
        setIsRagasEvaluating(false)
        return
      }

      if (ragasPollRef.current !== null) {
        window.clearInterval(ragasPollRef.current)
      }

      ragasPollRef.current = window.setInterval(async () => {
        try {
          const { data: status } = await axios.get<{
            status: 'idle' | 'running' | 'completed' | 'failed'
            message: string
            report?: RagasReport | null
          }>(`${API_BASE}/api/rag/eval/ragas/status`)

          if (status.status === 'completed') {
            if (ragasPollRef.current !== null) {
              window.clearInterval(ragasPollRef.current)
              ragasPollRef.current = null
            }
            setIsRagasEvaluating(false)
            if (status.report) {
              setRagasReport(status.report)
            } else {
              setRagasError('评测完成，但未返回报告，请检查后端日志')
            }
          } else if (status.status === 'failed') {
            if (ragasPollRef.current !== null) {
              window.clearInterval(ragasPollRef.current)
              ragasPollRef.current = null
            }
            setIsRagasEvaluating(false)
            setRagasError(status.message || 'Ragas 评测失败')
          }
        } catch (e: unknown) {
          if (ragasPollRef.current !== null) {
            window.clearInterval(ragasPollRef.current)
            ragasPollRef.current = null
          }
          setIsRagasEvaluating(false)
          setRagasError('Ragas 状态轮询失败：' + (e instanceof Error ? e.message : String(e)))
        }
      }, 2000)
    } catch (e: unknown) {
      setRagasError('Ragas 评测请求失败：' + (e instanceof Error ? e.message : String(e)))
      setIsRagasEvaluating(false)
    } finally {
      // 评测启动成功后由轮询来关闭 loading
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
          onClick={() => setUseStructRag(p => !p)}
          disabled={isBuilding || isClearing}
          style={{
            background:   (isBuilding || isClearing) ? '#2a2a3e' : (useStructRag ? '#18332a' : '#252540'),
            color:        (isBuilding || isClearing) ? '#666' : (useStructRag ? '#8fd4b5' : '#c0a0f0'),
            border:       `1px solid ${(isBuilding || isClearing) ? '#3a3a55' : (useStructRag ? '#4f8f6f' : '#7b5ea7')}`,
            borderRadius: '8px',
            padding:      '10px 16px',
            fontSize:     '13px',
            fontWeight:   'bold',
            cursor:       (isBuilding || isClearing) ? 'not-allowed' : 'pointer',
            transition:   'all 0.2s',
          }}
          title={useStructRag ? '当前构建结构化索引，点击切换普通索引' : '当前构建普通索引，点击切换结构化索引'}
        >
          {useStructRag ? '🧩 结构化RAG' : '🔍 普通RAG'}
        </button>

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
          {isClearing ? '⏳ 清除中...' : `🗑 清除${useStructRag ? '结构化' : '普通'}索引`}
        </button>

        <button
          onClick={handleClearAllIndexes}
          disabled={isBuilding || isClearing}
          style={{
            background:   (isBuilding || isClearing) ? '#2a2a3e' : 'transparent',
            color:        (isBuilding || isClearing) ? '#555'    : '#f0b25e',
            border:       `1px solid ${(isBuilding || isClearing) ? '#3a3a55' : '#f0b25e'}`,
            borderRadius: '8px',
            padding:      '10px 20px',
            fontSize:     '14px',
            fontWeight:   'bold',
            cursor:       (isBuilding || isClearing) ? 'not-allowed' : 'pointer',
            transition:   'all 0.2s',
          }}
          title="同时清除普通索引与结构化索引"
        >
          {isClearing ? '⏳ 清除中...' : '🧹 一键清除全部索引'}
        </button>

        <span style={{ fontSize: '12px', color: '#555' }}>
          {isBuilding
            ? `正在使用 ${selectedModel} 构建${useStructRag ? '结构化' : '普通'}向量索引...`
            : `将使用 ${selectedModel} 构建${useStructRag ? '结构化' : '普通'}向量索引`}
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
          <div>
            <span style={{ color: '#888' }}>模式：</span>
            {buildStatus.use_struct_rag ? '结构化RAG' : '普通RAG'}
          </div>
          {buildStatus.doc_count !== undefined && (
            <div><span style={{ color: '#888' }}>文档数量：</span>{buildStatus.doc_count} 条</div>
          )}
          {buildStatus.start_time && (
            <div><span style={{ color: '#888' }}>开始时间：</span>{buildStatus.start_time}</div>
          )}
        </div>
      )}

      {/* 评测面板 */}
      <div style={{
        marginTop: '22px',
        padding: '16px 18px',
        background: '#1f2035',
        border: '1px solid #343657',
        borderRadius: '10px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
          <div>
            <div style={{ color: '#d7dbff', fontSize: '14px', fontWeight: 'bold' }}>RAG 检索评测</div>
            <div style={{ color: '#7077a8', fontSize: '12px', marginTop: '2px' }}>
              {evalMetricView === 'recall_focused'
                ? '基于 25 条样本计算 Precision@k / Recall@k'
                : '基于 25 条样本计算 MRR / Hit@1 / Precision@1'}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <select
              value={evalMode}
              onChange={e => setEvalMode(e.target.value as EvalMode)}
              disabled={isEvaluating}
              style={{
                background: '#17182a',
                color: '#c7cdf7',
                border: '1px solid #3c3f66',
                borderRadius: '8px',
                padding: '8px 10px',
                fontSize: '12px',
                outline: 'none',
              }}
            >
              <option value="struct">仅结构化RAG</option>
              <option value="classic">仅普通RAG</option>
              <option value="both">普通 + 结构化</option>
            </select>

            <select
              value={evalTopK}
              onChange={e => setEvalTopK(e.target.value)}
              disabled={isEvaluating}
              style={{
                background: '#17182a',
                color: '#c7cdf7',
                border: '1px solid #3c3f66',
                borderRadius: '8px',
                padding: '8px 10px',
                fontSize: '12px',
                outline: 'none',
              }}
              title="评测 topk。留空表示按样本文件中的 k"
            >
              <option value="">按样本k</option>
              <option value="1">top1</option>
              <option value="2">top2</option>
              <option value="3">top3</option>
              <option value="5">top5</option>
              <option value="8">top8</option>
              <option value="10">top10</option>
            </select>

            <select
              value={evalMetricView}
              onChange={e => setEvalMetricView(e.target.value as EvalMetricView)}
              disabled={isEvaluating}
              style={{
                background: '#17182a',
                color: '#c7cdf7',
                border: '1px solid #3c3f66',
                borderRadius: '8px',
                padding: '8px 10px',
                fontSize: '12px',
                outline: 'none',
              }}
              title="评测指标视图切换"
            >
              <option value="recall_focused">召回导向（P@k / R@k）</option>
              <option value="first_hit_focused">首条命中导向（MRR / Hit@1 / P@1）</option>
            </select>

            <button
              onClick={handleRunEvaluation}
              disabled={isEvaluating || isBuilding}
              style={{
                background: (isEvaluating || isBuilding) ? '#2a2a3e' : '#4f68c6',
                color: (isEvaluating || isBuilding) ? '#666' : '#fff',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 14px',
                fontSize: '12px',
                fontWeight: 'bold',
                cursor: (isEvaluating || isBuilding) ? 'not-allowed' : 'pointer',
              }}
            >
              {isEvaluating ? '⏳ 评测中...' : '▶ 运行评测'}
            </button>
          </div>
        </div>

        {evalError && (
          <div style={{
            marginTop: '12px',
            padding: '10px 12px',
            background: '#3a1a1a',
            border: '1px solid #7a3030',
            borderRadius: '8px',
            color: '#e08080',
            fontSize: '12px',
          }}>
            {evalError}
          </div>
        )}

        {evalReport && (
          <div style={{ marginTop: '12px' }}>
            <div style={{ color: '#98a2d8', fontSize: '12px', marginBottom: '8px' }}>
              模型：{evalReport.embedding_model} ｜ 样本：{evalReport.samples_file} ｜ topk：{evalReport.k_override ?? '按样本k'}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
              {Object.entries(evalReport.summary).map(([mode, s]) => (
                <div
                  key={mode}
                  style={{
                    background: '#161729',
                    border: '1px solid #2f3155',
                    borderRadius: '8px',
                    padding: '10px 12px',
                  }}
                >
                  <div style={{ color: '#cfd5ff', fontSize: '12px', fontWeight: 'bold', marginBottom: '6px' }}>
                    {mode === 'struct' ? '结构化RAG' : mode === 'classic' ? '普通RAG' : mode}
                  </div>
                  <div style={{ color: '#96a0d5', fontSize: '12px' }}>samples: {s.samples}</div>
                  {evalMetricView === 'recall_focused' ? (
                    <>
                      <div style={{ color: '#96a0d5', fontSize: '12px' }}>
                        precision@k: <span style={{ color: '#8fd4b5' }}>{formatMetric(s.macro_precision_at_k)}</span>
                      </div>
                      <div style={{ color: '#96a0d5', fontSize: '12px' }}>
                        recall@k: <span style={{ color: '#f0d091' }}>{formatMetric(s.macro_recall_at_k)}</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div style={{ color: '#96a0d5', fontSize: '12px' }}>
                        mrr: <span style={{ color: '#7fd3ff' }}>{formatMetric(s.macro_mrr)}</span>
                      </div>
                      <div style={{ color: '#96a0d5', fontSize: '12px' }}>
                        hit@1: <span style={{ color: '#f0d091' }}>{formatMetric(s.macro_hit_at_1)}</span>
                      </div>
                      <div style={{ color: '#96a0d5', fontSize: '12px' }}>
                        precision@1: <span style={{ color: '#8fd4b5' }}>{formatMetric(s.macro_precision_at_1)}</span>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Ragas 评测面板 */}
      <div style={{
        marginTop: '22px',
        padding: '16px 18px',
        background: '#202437',
        border: '1px solid #3a4164',
        borderRadius: '10px',
      }}>
        <div style={{ marginBottom: '10px' }}>
          <div style={{ color: '#d7dbff', fontSize: '14px', fontWeight: 'bold' }}>Ragas 管线评测</div>
          <div style={{ color: '#7f89b8', fontSize: '12px', marginTop: '2px' }}>
            自动复用 AI 助手页面的 API Key / Base URL / Assistant API；仅需选择模型和评测参数
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          columnGap: '12px',
          rowGap: '16px',
          alignItems: 'start',
        }}>
          <div style={ragasFieldStyle}>
            <div style={ragasLabelStyle}>生成模型（Generator）</div>
            <select
              value={ragasModel}
              onChange={e => setRagasModel(e.target.value)}
              style={ragasInputStyle}
            >
              {assistantModelOptions.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <div style={ragasHintStyle}>用于被评测RAG回答生成</div>
          </div>

          <div style={ragasFieldStyle}>
            <div style={ragasLabelStyle}>判分模型（Judge LLM）</div>
            <select
              value={ragasJudgeModel}
              onChange={e => setRagasJudgeModel(e.target.value)}
              style={ragasInputStyle}
            >
              {assistantModelOptions.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <div style={ragasHintStyle}>Ragas 用来判断相关性与忠实度</div>
          </div>

          <div style={ragasFieldStyle}>
            <div style={ragasLabelStyle}>判分Embedding模型</div>
            <input
              value={ragasJudgeEmbeddingModel}
              onChange={e => setRagasJudgeEmbeddingModel(e.target.value)}
              placeholder="例如 text-embedding-v4"
              style={ragasInputStyle}
            />
            <div style={ragasHintStyle}>Ragas 计算语义相似度时使用</div>
          </div>

          <div style={ragasFieldStyle}>
            <div style={ragasLabelStyle}>样本文件路径</div>
            <input
              value={ragasSamplesFile}
              onChange={e => setRagasSamplesFile(e.target.value)}
              placeholder="例如 RAG/eval/ragas_eval_samples.json"
              style={ragasInputStyle}
            />
            <div style={ragasHintStyle}>JSON 数组，至少包含 question 与 ground_truth</div>
          </div>

          <div style={ragasFieldStyle}>
            <div style={ragasLabelStyle}>报告输出路径</div>
            <input
              value={ragasOutFile}
              onChange={e => setRagasOutFile(e.target.value)}
              placeholder="例如 RAG/eval/ragas_report.json"
              style={ragasInputStyle}
            />
            <div style={ragasHintStyle}>评测完成后写入本地报告文件</div>
          </div>

          <div style={ragasFieldStyle}>
            <div style={ragasLabelStyle}>检索模式与TopK</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <select
                value={String(ragasUseStruct)}
                onChange={e => setRagasUseStruct(e.target.value === 'true')}
                style={{ ...ragasInputStyle, width: '100%' }}
              >
                <option value="true">结构化RAG</option>
                <option value="false">普通RAG</option>
              </select>
              <select
                value={String(ragasTopK)}
                onChange={e => setRagasTopK(Number(e.target.value))}
                style={ragasInputStyle}
              >
                <option value="1">top1</option>
                <option value="2">top2</option>
                <option value="3">top3</option>
                <option value="5">top5</option>
                <option value="8">top8</option>
                <option value="10">top10</option>
              </select>
            </div>
            <div style={ragasHintStyle}>TopK 越大召回通常更高，但噪声也可能变多</div>
          </div>
        </div>

        <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={handleRunRagasEvaluation}
            disabled={isRagasEvaluating || isBuilding}
            style={{
              background: (isRagasEvaluating || isBuilding) ? '#2a2a3e' : '#3a7dd8',
              color: (isRagasEvaluating || isBuilding) ? '#666' : '#fff',
              border: 'none',
              borderRadius: '8px',
              padding: '8px 14px',
              fontSize: '12px',
              fontWeight: 'bold',
              cursor: (isRagasEvaluating || isBuilding) ? 'not-allowed' : 'pointer',
            }}
          >
            {isRagasEvaluating ? '⏳ Ragas评测中...' : '▶ 运行Ragas评测'}
          </button>

          <span style={{ color: '#8b93bf', fontSize: '12px' }}>
            检索嵌入模型：{selectedModel}
          </span>
        </div>

        {ragasError && (
          <div style={{ marginTop: '10px', padding: '10px 12px', background: '#3a1a1a', border: '1px solid #7a3030', borderRadius: '8px', color: '#e08080', fontSize: '12px' }}>
            {ragasError}
          </div>
        )}

        {ragasReport && (
          <div style={{ marginTop: '12px' }}>
            <div style={{ color: '#98a2d8', fontSize: '12px', marginBottom: '8px' }}>
              模式：{ragasReport.mode === 'struct' ? '结构化RAG' : '普通RAG'} ｜ 样本：{ragasReport.samples} ｜ topk：{ragasReport.top_k}
            </div>
            <div style={{ background: '#161729', border: '1px solid #2f3155', borderRadius: '8px', padding: '10px 12px' }}>
              {Object.entries(ragasReport.summary).map(([k, v]) => (
                <div key={k} style={{ color: '#96a0d5', fontSize: '12px', lineHeight: '1.8' }}>
                  {k}: <span style={{ color: '#8fd4b5' }}>{Number(v).toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}
