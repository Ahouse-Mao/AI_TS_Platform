// ===================== AI 助手聊天页面 =====================
import { useState, useEffect, useRef, useCallback } from 'react'

const API = 'http://localhost:8000'

/* ================================================================
   类型
   ================================================================ */
interface ChatMessage  { role: 'user' | 'assistant'; content: string }
interface Conversation { id: string; title: string; messages: ChatMessage[]; createdAt: number }
interface Settings     { apiKey: string; baseUrl: string; systemPrompt: string }

/* ================================================================
   常量
   ================================================================ */
const SIDEBAR_W = 260
const NAV_H     = 52        // App.tsx 顶部导航栏高度（近似）

const SK_CONVOS   = 'ai_ts_convos'
const SK_SETTINGS = 'ai_ts_settings'
const SK_MODEL    = 'ai_ts_model'
const SK_FETCHED_MODELS = 'ai_ts_fetched_models'

const MODELS = [
  { value: 'gpt-4o',            label: 'GPT-4o' },
  { value: 'gpt-4o-mini',       label: 'GPT-4o Mini' },
  { value: 'gpt-3.5-turbo',     label: 'GPT-3.5 Turbo' },
  { value: 'deepseek-chat',     label: 'DeepSeek Chat' },
  { value: 'deepseek-reasoner', label: 'DeepSeek R1' },
]

const DEFAULT_SETTINGS: Settings = {
  apiKey:       '',
  baseUrl:      'https://api.openai.com/v1',
  systemPrompt: '你是一个时序预测平台的 AI 助手，帮助用户理解模型、调优参数和分析预测结果。',
}

const QUICK_PROMPTS = [
  'DLinear 和 PatchTST 有什么区别？',
  '如何选择 seq_len 和 pred_len？',
  '解释 ONNX 推理的优势',
]

/* helpers */
const uid   = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n) + '…' : s)
function load<T>(key: string, fallback: T): T {
  try { const r = localStorage.getItem(key); return r ? JSON.parse(r) : fallback }
  catch { return fallback }
}

/* ================================================================
   主组件
   ================================================================ */
export function AssistantPage() {
  /* ── 全局状态 ── */
  const [convos,        setConvos]        = useState<Conversation[]>(() => load(SK_CONVOS, []))
  const [activeId,      setActiveId]      = useState<string | null>(null)
  const [model,         setModel]         = useState<string>(() => load(SK_MODEL, 'gpt-4o'))
  const [settings,      setSettings]      = useState<Settings>(() => load(SK_SETTINGS, DEFAULT_SETTINGS))
  const [fetchedModels, setFetchedModels] = useState<string[]>(() => load(SK_FETCHED_MODELS, []))

  /* ── 输入 & 流式 ── */
  const [input,      setInput]      = useState('')
  const [streaming,  setStreaming]  = useState(false)
  const [streamText, setStreamText] = useState('')

  /* ── UI 状态 ── */
  const [showSettings, setShowSettings] = useState(false)
  const [editingId,    setEditingId]    = useState<string | null>(null)
  const [editTitle,    setEditTitle]    = useState('')
  const [hoveredId,    setHoveredId]    = useState<string | null>(null)

  const activeConvo = convos.find(c => c.id === activeId) ?? null
  const messagesRef = useRef<HTMLDivElement>(null)
  const inputRef    = useRef<HTMLTextAreaElement>(null)
  const abortRef    = useRef<AbortController | null>(null)

  /* ── 持久化到 localStorage ── */
  useEffect(() => { localStorage.setItem(SK_CONVOS,         JSON.stringify(convos)) },        [convos])
  useEffect(() => { localStorage.setItem(SK_MODEL,          JSON.stringify(model)) },         [model])
  useEffect(() => { localStorage.setItem(SK_SETTINGS,       JSON.stringify(settings)) },      [settings])
  useEffect(() => { localStorage.setItem(SK_FETCHED_MODELS, JSON.stringify(fetchedModels)) }, [fetchedModels])

  /* ── 消息变化时自动滚动到底部 ── */
  useEffect(() => {
    const el = messagesRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [streamText, activeConvo?.messages.length])

  /* ── 新建对话 ── */
  const newConvo = useCallback(() => {
    const c: Conversation = { id: uid(), title: '新对话', messages: [], createdAt: Date.now() }
    setConvos(prev => [c, ...prev])
    setActiveId(c.id)
    setInput('')
    setTimeout(() => inputRef.current?.focus(), 60)
  }, [])

  /* ── 删除对话 ── */
  const deleteConvo = useCallback((id: string) => {
    setConvos(prev => prev.filter(c => c.id !== id))
    if (activeId === id) setActiveId(null)
  }, [activeId])

  /* ── 重命名 ── */
  const startRename   = (id: string, title: string) => { setEditingId(id); setEditTitle(title) }
  const confirmRename = () => {
    if (editingId && editTitle.trim()) {
      setConvos(prev => prev.map(c => c.id === editingId ? { ...c, title: editTitle.trim() } : c))
    }
    setEditingId(null)
  }

  /* ── 发送消息（流式） ── */
  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return

    let targetId      = activeId
    let currentConvos = convos

    // 没有活跃对话 → 自动创建
    if (!targetId) {
      const c: Conversation = { id: uid(), title: trunc(text, 30), messages: [], createdAt: Date.now() }
      currentConvos = [c, ...currentConvos]
      targetId      = c.id
      setActiveId(targetId)
    }

    // 追加用户消息
    const userMsg: ChatMessage = { role: 'user', content: text }
    const updatedConvos = currentConvos.map(c =>
      c.id === targetId
        ? { ...c, messages: [...c.messages, userMsg], title: c.messages.length === 0 ? trunc(text, 30) : c.title }
        : c
    )
    setConvos(updatedConvos)
    setInput('')
    setStreaming(true)
    setStreamText('')
    if (inputRef.current) inputRef.current.style.height = 'auto'

    // 构造 API 消息列表
    const target  = updatedConvos.find(c => c.id === targetId)!
    const apiMsgs: { role: string; content: string }[] = []
    if (settings.systemPrompt.trim()) apiMsgs.push({ role: 'system', content: settings.systemPrompt })
    apiMsgs.push(...target.messages.map(m => ({ role: m.role, content: m.content })))

    let full = ''
    try {
      const abort = new AbortController()
      abortRef.current = abort

      const res = await fetch(`${API}/api/assistant/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ messages: apiMsgs, model, api_key: settings.apiKey, base_url: settings.baseUrl }),
        signal:  abort.signal,
      })

      const reader = res.body!.getReader()
      const dec    = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()!
        for (const ln of lines) {
          if (!ln.startsWith('data: ')) continue
          const d = ln.slice(6)
          if (d === '[DONE]') continue
          try {
            const p = JSON.parse(d)
            if (p.error)        full += (full ? '\n' : '') + '⚠️ ' + p.error
            else if (p.content) full += p.content
            setStreamText(full)
          } catch { /* 忽略非 JSON 行 */ }
        }
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === 'AbortError') { /* 用户取消 */ }
      else { full += (full ? '\n' : '') + '⚠️ 请求失败：' + (e instanceof Error ? e.message : String(e)) }
      setStreamText(full)
    }

    // 流式结束 → 将 assistant 回复写入对话
    if (full) {
      const fid = targetId
      setConvos(prev => prev.map(c =>
        c.id === fid ? { ...c, messages: [...c.messages, { role: 'assistant' as const, content: full }] } : c
      ))
    }
    setStreaming(false)
    setStreamText('')
    abortRef.current = null
  }, [input, streaming, activeId, convos, settings, model])

  /* ── 停止生成 ── */
  const stop = useCallback(() => { abortRef.current?.abort() }, [])

  /* ── 输入处理 ── */
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send() }
  }

  /* ── 构建显示消息列表 (对话消息 + 正在流式的 assistant 消息) ── */
  const displayMessages: ChatMessage[] = [
    ...(activeConvo?.messages ?? []),
    ...(streaming && streamText ? [{ role: 'assistant' as const, content: streamText }] : []),
  ]

  /* ================================================================
     渲染
     ================================================================ */
  return (
    <>
      {/* 闪烁光标动画 */}
      <style>{`@keyframes cursorBlink{0%,100%{opacity:1}50%{opacity:0}}`}</style>

      <div style={{
        margin:   '-40px -50px',
        height:   `calc(100vh - ${NAV_H}px)`,
        display:  'flex',
        overflow: 'hidden',
        background: '#1a1a2e',
      }}>

        {/* ════════ 左侧边栏 ════════ */}
        <aside style={{
          width:       SIDEBAR_W,
          minWidth:    SIDEBAR_W,
          background:  '#141422',
          borderRight: '1px solid #2a2a3d',
          display:     'flex',
          flexDirection: 'column',
        }}>
          {/* 新建对话 */}
          <div style={{ padding: '16px 14px 8px' }}>
            <button onClick={newConvo} style={{
              width: '100%', padding: '10px',
              background: '#252540', border: '1px solid #3a3a55', borderRadius: '8px',
              color: '#ccc', fontSize: '13px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              transition: 'all 0.15s',
            }}
              onMouseEnter={e => { e.currentTarget.style.background = '#2d2d50'; e.currentTarget.style.borderColor = '#7b8cde' }}
              onMouseLeave={e => { e.currentTarget.style.background = '#252540'; e.currentTarget.style.borderColor = '#3a3a55' }}
            ><span style={{ fontSize: '16px', fontWeight: 'bold' }}>+</span> 发起新对话</button>
          </div>

          {/* 对话列表 */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px' }}>
            {convos.length === 0 && (
              <div style={{ textAlign: 'center', color: '#444', fontSize: '12px', marginTop: '40px' }}>暂无对话记录</div>
            )}
            {convos.map(c => {
              const isActive  = c.id === activeId
              const isHovered = c.id === hoveredId
              const isEditing = c.id === editingId
              return (
                <div key={c.id}
                  onClick={() => { setActiveId(c.id); setTimeout(() => inputRef.current?.focus(), 60) }}
                  onMouseEnter={() => setHoveredId(c.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  style={{
                    padding: '10px 12px', borderRadius: '8px', cursor: 'pointer',
                    background: isActive ? '#252545' : isHovered ? '#1e1e35' : 'transparent',
                    marginBottom: '2px',
                    display: 'flex', alignItems: 'center', gap: '8px',
                    transition: 'background 0.1s',
                  }}
                >
                  {isEditing ? (
                    <input autoFocus value={editTitle}
                      onChange={e => setEditTitle(e.target.value)}
                      onBlur={confirmRename}
                      onKeyDown={e => { if (e.key === 'Enter') confirmRename(); if (e.key === 'Escape') setEditingId(null) }}
                      onClick={e => e.stopPropagation()}
                      style={{
                        flex: 1, background: '#1a1a2e', border: '1px solid #7b8cde', borderRadius: '4px',
                        color: '#ccc', padding: '2px 6px', fontSize: '13px', outline: 'none',
                      }}
                    />
                  ) : (
                    <>
                      <span style={{
                        flex: 1, fontSize: '13px', color: isActive ? '#e0e0f0' : '#999',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>{c.title}</span>
                      {(isActive || isHovered) && (
                        <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }} onClick={e => e.stopPropagation()}>
                          <MiniBtn tip="重命名" onClick={() => startRename(c.id, c.title)}>✎</MiniBtn>
                          <MiniBtn tip="删除"   onClick={() => deleteConvo(c.id)}>✕</MiniBtn>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )
            })}
          </div>

          {/* 底部设置按钮 */}
          <div style={{ borderTop: '1px solid #2a2a3d', padding: '10px 14px' }}>
            <button onClick={() => setShowSettings(true)} style={{
              width: '100%', padding: '8px',
              background: 'transparent', border: 'none',
              color: '#777', fontSize: '13px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: '8px', borderRadius: '6px',
              transition: 'all 0.15s',
            }}
              onMouseEnter={e => { e.currentTarget.style.background = '#1e1e35'; e.currentTarget.style.color = '#bbb' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#777' }}
            ><span style={{ fontSize: '15px' }}>⚙</span> 设置</button>
          </div>
        </aside>

        {/* ════════ 右侧主区域 ════════ */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>

          {/* ── 顶栏：标题 + 模型选择 ── */}
          <div style={{
            height: '52px', borderBottom: '1px solid #2a2a3d',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0 24px', flexShrink: 0,
          }}>
            <span style={{ color: '#888', fontSize: '14px' }}>
              {activeConvo ? activeConvo.title : 'AI 助手'}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <select value={model} onChange={e => setModel(e.target.value)}
                style={{
                  background: '#1e1e35', border: '1px solid #3a3a55', borderRadius: '6px',
                  color: '#ccc', padding: '5px 10px', fontSize: '13px',
                  cursor: 'pointer', outline: 'none',
                }}
              >
                {fetchedModels.length > 0
                  ? fetchedModels.map(m => <option key={m} value={m}>{m}</option>)
                  : MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
              <button onClick={() => setShowSettings(true)} style={{
                background: 'transparent', border: '1px solid #3a3a55', borderRadius: '6px',
                color: '#888', padding: '4px 10px', fontSize: '14px', cursor: 'pointer',
              }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = '#7b8cde'; e.currentTarget.style.color = '#bbb' }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = '#3a3a55'; e.currentTarget.style.color = '#888' }}
              >⚙</button>
            </div>
          </div>

          {/* ── 聊天消息区 ── */}
          <div ref={messagesRef} style={{ flex: 1, overflowY: 'auto', padding: '24px 0' }}>
            {displayMessages.length === 0 ? (
              /* 欢迎屏 */
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: '#555', gap: '16px',
              }}>
                <div style={{ fontSize: '48px', opacity: 0.3 }}>🤖</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#888' }}>AI 助手</div>
                <div style={{ fontSize: '13px', color: '#555', textAlign: 'center', maxWidth: '420px', lineHeight: '1.7' }}>
                  我可以帮你理解时序预测模型、调优训练参数、<br />分析预测结果，或回答任何技术问题。
                </div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                  {QUICK_PROMPTS.map(q => (
                    <button key={q} onClick={() => { setInput(q); inputRef.current?.focus() }} style={{
                      background: '#252540', border: '1px solid #3a3a55', borderRadius: '20px',
                      color: '#999', padding: '8px 16px', fontSize: '12px', cursor: 'pointer',
                      transition: 'all 0.15s',
                    }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = '#7b8cde'; e.currentTarget.style.color = '#ccc' }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = '#3a3a55'; e.currentTarget.style.color = '#999' }}
                    >{q}</button>
                  ))}
                </div>
              </div>
            ) : (
              /* 消息列表 */
              <div style={{ maxWidth: '760px', margin: '0 auto', padding: '0 24px' }}>
                {displayMessages.map((msg, i) => (
                  <MessageBubble key={i} msg={msg}
                    isStreaming={streaming && i === displayMessages.length - 1 && msg.role === 'assistant'} />
                ))}
              </div>
            )}
          </div>

          {/* ── 输入区域 ── */}
          <div style={{ borderTop: '1px solid #2a2a3d', padding: '16px 24px', flexShrink: 0 }}>
            <div style={{ maxWidth: '760px', margin: '0 auto', display: 'flex', gap: '10px', alignItems: 'flex-end' }}>
              <textarea ref={inputRef} value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                placeholder="输入消息… (Enter 发送，Shift+Enter 换行)"
                rows={1}
                style={{
                  flex: 1,
                  background: '#252540', border: '1px solid #3a3a55', borderRadius: '12px',
                  color: '#e0e0e0', padding: '12px 16px', fontSize: '14px', lineHeight: '1.5',
                  resize: 'none', outline: 'none', fontFamily: 'inherit',
                  maxHeight: '200px', overflowY: 'auto',
                }}
                onFocus={e => { e.currentTarget.style.borderColor = '#7b8cde' }}
                onBlur={e  => { e.currentTarget.style.borderColor = '#3a3a55' }}
              />
              {streaming ? (
                <button onClick={stop} style={{
                  width: '42px', height: '42px',
                  background: '#e05050', border: 'none', borderRadius: '10px',
                  color: '#fff', fontSize: '16px', cursor: 'pointer', flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }} title="停止生成">■</button>
              ) : (
                <button onClick={send} disabled={!input.trim()} style={{
                  width: '42px', height: '42px',
                  background: input.trim() ? '#7b8cde' : '#2a2a3e',
                  border: 'none', borderRadius: '10px',
                  color: input.trim() ? '#fff' : '#555',
                  fontSize: '18px',
                  cursor: input.trim() ? 'pointer' : 'not-allowed',
                  flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'background 0.15s',
                }} title="发送">↑</button>
              )}
            </div>
            <div style={{ textAlign: 'center', fontSize: '11px', color: '#444', marginTop: '8px' }}>
              AI 可能会产生不准确的信息，请注意甄别
            </div>
          </div>
        </main>

        {/* ════════ 设置弹窗 ════════ */}
        {showSettings && (
          <SettingsModal
            settings={settings}
            initialFetchedModels={fetchedModels}
            onSave={(s, models) => { setSettings(s); setFetchedModels(models); setShowSettings(false) }}
            onClose={() => setShowSettings(false)}
          />
        )}
      </div>
    </>
  )
}

/* ================================================================
   子组件
   ================================================================ */

/** 侧栏迷你操作按钮 */
function MiniBtn({ children, tip, onClick }: { children: React.ReactNode; tip: string; onClick: () => void }) {
  return (
    <button title={tip} onClick={onClick} style={{
      background: 'transparent', border: 'none',
      color: '#666', cursor: 'pointer', fontSize: '13px',
      padding: '2px 4px', borderRadius: '4px', lineHeight: 1,
    }}
      onMouseEnter={e => { e.currentTarget.style.color = '#ccc'; e.currentTarget.style.background = '#2a2a40' }}
      onMouseLeave={e => { e.currentTarget.style.color = '#666'; e.currentTarget.style.background = 'transparent' }}
    >{children}</button>
  )
}

/** 聊天气泡 */
function MessageBubble({ msg, isStreaming }: { msg: ChatMessage; isStreaming: boolean }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: '16px',
    }}>
      {/* AI 头像 */}
      {!isUser && (
        <div style={{
          width: '32px', height: '32px', borderRadius: '8px', background: '#2d3455',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '15px', flexShrink: 0, marginRight: '10px', marginTop: '2px',
        }}>🤖</div>
      )}

      {/* 消息体 */}
      <div style={{
        maxWidth: '78%',
        padding: '10px 16px',
        borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
        background: isUser ? '#2d3a5e' : '#252540',
        color: '#e0e0e0',
        fontSize: '14px', lineHeight: '1.65',
        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>
        {msg.content || (isStreaming ? '' : '…')}
        {isStreaming && (
          <span style={{
            display: 'inline-block', width: '6px', height: '16px',
            background: '#7b8cde', marginLeft: '2px', verticalAlign: 'text-bottom',
            animation: 'cursorBlink 1s step-end infinite',
          }} />
        )}
      </div>

      {/* 用户头像 */}
      {isUser && (
        <div style={{
          width: '32px', height: '32px', borderRadius: '8px', background: '#3a4a6e',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '15px', flexShrink: 0, marginLeft: '10px', marginTop: '2px',
        }}>👤</div>
      )}
    </div>
  )
}

/** 设置弹窗 */
function SettingsModal({ settings, initialFetchedModels, onSave, onClose }: {
  settings:             Settings
  initialFetchedModels: string[]
  onSave:  (s: Settings, models: string[]) => void
  onClose: () => void
}) {
  const [temp,          setTemp]          = useState<Settings>({ ...settings })
  const [fetchedModels, setFetchedModels] = useState<string[]>(initialFetchedModels)
  const [fetchLoading,  setFetchLoading]  = useState(false)
  const [fetchError,    setFetchError]    = useState('')

  const fetchModels = async () => {
    if (!temp.apiKey.trim()) { setFetchError('请先填写 API Key'); return }
    setFetchLoading(true)
    setFetchError('')
    try {
      const params = new URLSearchParams({ api_key: temp.apiKey, base_url: temp.baseUrl })
      const res  = await fetch(`${API}/api/assistant/models?${params}`)
      const data = await res.json() as { models: string[]; error?: string }
      if (data.error) { setFetchError(data.error); return }
      setFetchedModels(data.models)
    } catch (e) {
      setFetchError('请求失败：' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setFetchLoading(false)
    }
  }

  return (
    <div style={{
      position: 'fixed', inset: 0,
      background: 'rgba(0,0,0,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000,
    }} onClick={onClose}>
      <div style={{
        background: '#1e1e2e', border: '1px solid #3a3a55', borderRadius: '16px',
        padding: '28px 32px', width: '520px', maxWidth: '90vw',
        boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        maxHeight: '90vh', overflowY: 'auto',
      }} onClick={e => e.stopPropagation()}>

        <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: '#e0e0f0' }}>⚙ 设置</h3>

        <FieldGroup label="API Key" desc="支持 OpenAI、DeepSeek 等兼容接口">
          <input type="password"
            value={temp.apiKey}
            onChange={e => setTemp(p => ({ ...p, apiKey: e.target.value }))}
            placeholder="sk-..."
            style={modalInputStyle}
          />
        </FieldGroup>

        <FieldGroup label="Base URL" desc="API 端点地址，不同服务商地址不同">
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              value={temp.baseUrl}
              onChange={e => setTemp(p => ({ ...p, baseUrl: e.target.value }))}
              placeholder="https://api.openai.com/v1"
              style={{ ...modalInputStyle, flex: 1 }}
            />
            <button onClick={fetchModels} disabled={fetchLoading} style={{
              padding: '0 14px', borderRadius: '8px', fontSize: '12px', fontWeight: 'bold',
              cursor: fetchLoading ? 'not-allowed' : 'pointer',
              border: '1px solid #3a3a55',
              background: fetchLoading ? '#1a1a2e' : '#252545',
              color: fetchLoading ? '#555' : '#9ab0e8',
              whiteSpace: 'nowrap', transition: 'all 0.15s', flexShrink: 0,
            }}
              onMouseEnter={e => { if (!fetchLoading) { e.currentTarget.style.borderColor = '#7b8cde'; e.currentTarget.style.color = '#c0ccff' } }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = '#3a3a55'; e.currentTarget.style.color = '#9ab0e8' }}
            >{fetchLoading ? '获取中…' : '拉取模型'}</button>
          </div>
        </FieldGroup>

        {/* 模型列表 */}
        {(fetchedModels.length > 0 || fetchError) && (
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#ccc', marginBottom: '6px' }}>
              可用模型
              <span style={{ fontSize: '11px', color: '#666', fontWeight: 'normal', marginLeft: '8px' }}>
                {fetchedModels.length > 0 ? `共 ${fetchedModels.length} 个，将替换默认列表` : ''}
              </span>
            </div>
            {fetchError && (
              <div style={{
                padding: '8px 12px', borderRadius: '6px',
                background: '#3a1a1a', border: '1px solid #7a3030',
                color: '#e08080', fontSize: '12px', marginBottom: '8px',
              }}>{fetchError}</div>
            )}
            {fetchedModels.length > 0 && (
              <div style={{
                maxHeight: '160px', overflowY: 'auto',
                border: '1px solid #2a2a3d', borderRadius: '8px',
                padding: '8px 10px',
                display: 'flex', flexWrap: 'wrap', gap: '6px',
              }}>
                {fetchedModels.map(m => (
                  <span key={m} style={{
                    display: 'inline-block',
                    padding: '3px 10px', borderRadius: '12px',
                    background: '#1e2040', border: '1px solid #3a3a60',
                    color: '#a0b0e0', fontSize: '12px',
                  }}>{m}</span>
                ))}
              </div>
            )}
            {fetchedModels.length > 0 && (
              <button onClick={() => setFetchedModels([])} style={{
                marginTop: '6px', background: 'transparent', border: 'none',
                color: '#555', fontSize: '11px', cursor: 'pointer', padding: 0,
              }}
                onMouseEnter={e => { e.currentTarget.style.color = '#e08080' }}
                onMouseLeave={e => { e.currentTarget.style.color = '#555' }}
              >✕ 清除，恢复默认列表</button>
            )}
          </div>
        )}

        <FieldGroup label="系统提示词" desc="用于指导 AI 行为的隐藏消息（可留空）">
          <textarea
            value={temp.systemPrompt}
            onChange={e => setTemp(p => ({ ...p, systemPrompt: e.target.value }))}
            rows={3}
            style={{ ...modalInputStyle, resize: 'vertical', fontFamily: 'inherit', lineHeight: '1.5' }}
          />
        </FieldGroup>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '24px' }}>
          <button onClick={onClose}                          style={modalBtnStyle(false)}>取消</button>
          <button onClick={() => onSave(temp, fetchedModels)} style={modalBtnStyle(true)}>保存</button>
        </div>
      </div>
    </div>
  )
}

/** 设置字段组 */
function FieldGroup({ label, desc, children }: { label: string; desc?: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'block', marginBottom: '16px' }}>
      <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#ccc', marginBottom: '4px' }}>{label}</div>
      {desc && <div style={{ fontSize: '11px', color: '#666', marginBottom: '6px' }}>{desc}</div>}
      {children}
    </label>
  )
}

/* ================================================================
   共用样式
   ================================================================ */
const modalInputStyle: React.CSSProperties = {
  width:        '100%',
  background:   '#141422',
  border:       '1px solid #3a3a55',
  borderRadius: '8px',
  color:        '#e0e0e0',
  padding:      '10px 14px',
  fontSize:     '13px',
  outline:      'none',
  boxSizing:    'border-box',
}

function modalBtnStyle(primary: boolean): React.CSSProperties {
  return {
    padding:      '8px 24px',
    borderRadius: '8px',
    fontSize:     '13px',
    fontWeight:   'bold',
    cursor:       'pointer',
    border:       primary ? 'none' : '1px solid #3a3a55',
    background:   primary ? '#7b8cde' : 'transparent',
    color:        primary ? '#fff' : '#999',
    transition:   'all 0.15s',
  }
}
