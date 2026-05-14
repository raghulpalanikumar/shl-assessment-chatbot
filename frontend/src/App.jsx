import { useMemo, useState } from 'react'
import './App.css'

// In dev, use same-origin + Vite proxy to backend (see vite.config.js). Override with VITE_API_URL if needed.
const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? '' : 'http://localhost:8000')
const MAX_MESSAGES = 8

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [recommendations, setRecommendations] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const canSend =
    input.trim() &&
    !isLoading &&
    messages.length < MAX_MESSAGES

  const visibleHistory = useMemo(
    () => messages.filter((m) => m.role === 'user' || m.role === 'assistant'),
    [messages],
  )

  async function sendMessage(event) {
    event.preventDefault()
    if (!canSend) return

    const prior = messages
    const userMessage = { role: 'user', content: input.trim() }
    const nextMessages = [...messages, userMessage].slice(-MAX_MESSAGES)
    setMessages(nextMessages)
    setInput('')
    setError('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ messages: nextMessages }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail?.[0]?.msg || data.error || 'Request failed.')

      const assistantMessage = { role: 'assistant', content: data.reply }
      const updated = [...nextMessages, assistantMessage].slice(-MAX_MESSAGES)
      setMessages(updated)
      setRecommendations(Array.isArray(data.recommendations) ? data.recommendations : [])
    } catch (requestError) {
      setError(requestError.message)
      setMessages(prior)
    } finally {
      setIsLoading(false)
    }
  }

  function clearChat() {
    setMessages([])
    setRecommendations([])
    setError('')
    setInput('')
  }

  return (
    <main className="shell">
      <div className="topbar-wrapper">
        <header className="topbar">
          <div className="topbar-branding">
            <div className="shl-logo">SHL<span className="shl-logo-accent">.</span></div>
            <div>
              <p className="eyebrow">Assessment Intelligence</p>
              <h1>AI Recommender Agent</h1>
            </div>
          </div>
          <button className="ghost-button" type="button" onClick={clearChat}>
            Clear Chat
          </button>
        </header>
      </div>

      <section className="workspace" aria-label="SHL assessment recommendation assistant">
        <div className="content-grid">
          <section className="chat-panel" aria-label="Conversation">
            <div className="messages">
              {visibleHistory.length === 0 && (
                <article className="message assistant intro-block">
                  <span>Assistant</span>
                  <p>
                    Describe the role, seniority, and what you want to measure (technical, cognitive, personality, or a
                    pasted job description). I only recommend Individual Test Solutions from the SHL catalog—up to eight
                    messages per session.
                  </p>
                </article>
              )}
              {visibleHistory.map((message, index) => (
                <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  <span>{message.role === 'assistant' ? 'Assistant' : 'Recruiter'}</span>
                  <p>{message.content}</p>
                </article>
              ))}
              {isLoading && (
                <article className="message assistant">
                  <span>Assistant</span>
                  <p>Reviewing your conversation against the SHL catalog…</p>
                </article>
              )}
            </div>

            {messages.length >= MAX_MESSAGES && (
              <p className="limit-note">This demo keeps the last {MAX_MESSAGES} messages to match the evaluation limit.</p>
            )}

            {error && <p className="error">{error}</p>}

            <form className="composer" onSubmit={sendMessage}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Example: Mid-level Java developer, 4 years, strong stakeholders and problem solving. Or paste a short job description."
                rows={3}
                disabled={messages.length >= MAX_MESSAGES}
              />
              <button type="submit" disabled={!canSend}>
                Send
              </button>
            </form>
          </section>

          <aside className="side-panel" aria-label="Recommendations">
            <section className="recommendations" aria-label="Recommendations">
              <div className="section-heading">
                <h2>Shortlisted Assessments</h2>
                <span className="badge">{recommendations.length}/10</span>
              </div>
              {recommendations.length === 0 ? (
                <p className="empty-state">
                  When the agent gathers enough context, you will see up to ten catalog-backed assessments here. Empty means
                  the agent is clarifying, comparing, or finishing up.
                </p>
              ) : (
                <ul>
                  {recommendations.map((item) => (
                    <li key={`${item.name}-${item.url}`}>
                      <div className="rec-header">
                        <a href={item.url} target="_blank" rel="noreferrer">
                          {item.name}
                        </a>
                        <span className="type-badge" title="Test Type">{item.test_type}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>
        </div>
      </section>
    </main>
  )
}
