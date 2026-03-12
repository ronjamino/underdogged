'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

export function LoginForm() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const supabase = createClient()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')

    const { error } = await supabase.auth.signInWithPassword({ email, password })

    if (error) {
      setError(error.message)
      setLoading(false)
      return
    }

    router.push('/dashboard')
    router.refresh()
  }

  return (
    <div style={{ width: '100%', maxWidth: '380px' }}>
      {/* Wordmark */}
      <div style={{ marginBottom: '8px' }}>
        <span className="font-display" style={{
          fontSize: '42px',
          fontStyle: 'italic',
          color: 'var(--text)',
          display: 'block',
          lineHeight: 1.1,
        }}>
          underdogged
        </span>
      </div>
      <p style={{
        color: 'var(--text-muted)',
        fontSize: '11px',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        marginBottom: '40px',
      }}>
        match predictions. five leagues.
      </p>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '12px' }}>
          <input
            type="email"
            placeholder="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            autoComplete="email"
            style={{
              width: '100%',
              padding: '12px 14px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
              fontSize: '13px',
            }}
          />
        </div>
        <div style={{ marginBottom: '20px' }}>
          <input
            type="password"
            placeholder="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            autoComplete="current-password"
            style={{
              width: '100%',
              padding: '12px 14px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
              fontSize: '13px',
            }}
          />
        </div>

        {error && (
          <div style={{
            marginBottom: '16px',
            padding: '10px 14px',
            background: 'var(--red-dim)',
            border: '1px solid var(--red)',
            color: 'var(--red)',
            fontSize: '12px',
          }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%',
            padding: '13px 14px',
            background: loading ? 'var(--text-muted)' : 'var(--accent)',
            color: '#0A0A0A',
            fontSize: '13px',
            fontWeight: 600,
            letterSpacing: '0.05em',
            border: 'none',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'background 0.15s',
          }}
        >
          {loading ? 'signing in...' : 'sign in →'}
        </button>
      </form>
    </div>
  )
}
