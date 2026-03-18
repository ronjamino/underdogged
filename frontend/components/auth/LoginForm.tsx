'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

type Mode = 'signin' | 'signup'

export function LoginForm() {
  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [googleLoading, setGoogleLoading] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const router = useRouter()
  const supabase = createClient()

  const redirectTo =
    typeof window !== 'undefined'
      ? `${window.location.origin}/auth/callback`
      : `${process.env.NEXT_PUBLIC_SITE_URL ?? 'https://underdogged.vercel.app'}/auth/callback`

  async function handleGoogle() {
    setGoogleLoading(true)
    setError('')
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo },
    })
    if (error) {
      setError(error.message)
      setGoogleLoading(false)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')

    if (mode === 'signin') {
      const { error } = await supabase.auth.signInWithPassword({ email, password })
      if (error) {
        setError(error.message)
        setLoading(false)
        return
      }
      router.push('/dashboard')
      router.refresh()
    } else {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: { emailRedirectTo: redirectTo },
      })
      if (error) {
        setError(error.message)
        setLoading(false)
        return
      }
      setConfirmed(true)
      setLoading(false)
    }
  }

  function switchMode(next: Mode) {
    setMode(next)
    setError('')
    setConfirmed(false)
    setEmail('')
    setPassword('')
  }

  return (
    <div style={{
      width: '100%',
      maxWidth: '400px',
      position: 'relative',
      zIndex: 10,
    }}>
      {/* Glassmorphic card */}
      <div style={{
        background: 'rgba(12,14,26,0.75)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        border: '1px solid rgba(28,32,64,0.9)',
        borderRadius: '12px',
        padding: '36px 32px',
        boxShadow: '0 24px 64px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.03) inset',
      }}>

        {/* Wordmark */}
        <div style={{ marginBottom: '6px' }}>
          <span className="font-display" style={{
            fontSize: '36px',
            fontStyle: 'italic',
            background: 'linear-gradient(135deg, #F5A623 0%, #FFD07A 60%, #F5A623 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            display: 'block',
            lineHeight: 1.1,
          }}>
            underdogged
          </span>
        </div>
        <p style={{
          color: 'var(--text-muted)',
          fontSize: '10px',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          marginBottom: '28px',
        }}>
          match predictions · five leagues
        </p>

        {/* Divider */}
        <div style={{ height: '1px', background: 'var(--border)', marginBottom: '24px' }} />

        {/* Mode toggle */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: '24px' }}>
          {(['signin', 'signup'] as Mode[]).map(m => (
            <button
              key={m}
              onClick={() => switchMode(m)}
              style={{
                padding: '8px 14px',
                background: 'none',
                border: 'none',
                borderBottom: mode === m ? '2px solid var(--accent)' : '2px solid transparent',
                color: mode === m ? 'var(--accent)' : 'var(--text-muted)',
                textShadow: mode === m ? '0 0 20px var(--accent-glow)' : 'none',
                fontSize: '10px',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                cursor: 'pointer',
                fontFamily: 'JetBrains Mono, monospace',
                marginBottom: '-1px',
                transition: 'color 0.15s',
              }}
            >
              {m === 'signin' ? 'Sign In' : 'Create Account'}
            </button>
          ))}
        </div>

        {/* Confirmed state */}
        {confirmed ? (
          <div style={{
            padding: '14px 16px',
            background: 'var(--green-dim)',
            border: '1px solid rgba(16,217,122,0.3)',
            borderRadius: '6px',
            color: 'var(--green)',
            fontSize: '12px',
            lineHeight: 1.7,
          }}>
            Check your email — we sent a confirmation link to <strong>{email}</strong>.
            Click it to activate your account, then sign in.
          </div>
        ) : (
          <>
            {/* Google OAuth */}
            <button
              onClick={handleGoogle}
              disabled={googleLoading}
              style={{
                width: '100%',
                padding: '11px 14px',
                background: 'rgba(28,32,64,0.5)',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                color: googleLoading ? 'var(--text-muted)' : 'var(--text)',
                fontSize: '12px',
                cursor: googleLoading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px',
                transition: 'border-color 0.15s, background 0.15s',
                fontFamily: 'JetBrains Mono, monospace',
                letterSpacing: '0.02em',
              }}
              onMouseEnter={e => {
                if (!googleLoading) {
                  const b = e.currentTarget as HTMLButtonElement
                  b.style.borderColor = 'var(--text-muted)'
                  b.style.background = 'rgba(28,32,64,0.9)'
                }
              }}
              onMouseLeave={e => {
                const b = e.currentTarget as HTMLButtonElement
                b.style.borderColor = 'var(--border)'
                b.style.background = 'rgba(28,32,64,0.5)'
              }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
              {googleLoading ? 'redirecting...' : 'Continue with Google'}
            </button>

            {/* Divider */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '18px 0' }}>
              <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
              <span style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase' }}>or</span>
              <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
            </div>

            {/* Email / password form */}
            <form onSubmit={handleSubmit}>
              <div style={{ marginBottom: '10px' }}>
                <input
                  type="email"
                  placeholder="email address"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                  style={{
                    width: '100%',
                    padding: '11px 14px',
                    background: 'rgba(7,8,15,0.6)',
                    border: '1px solid var(--border)',
                    borderRadius: '6px',
                    color: 'var(--text)',
                    fontSize: '12px',
                    fontFamily: 'JetBrains Mono, monospace',
                    letterSpacing: '0.02em',
                    transition: 'border-color 0.15s',
                  }}
                />
              </div>
              <div style={{ marginBottom: '18px' }}>
                <input
                  type="password"
                  placeholder="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
                  style={{
                    width: '100%',
                    padding: '11px 14px',
                    background: 'rgba(7,8,15,0.6)',
                    border: '1px solid var(--border)',
                    borderRadius: '6px',
                    color: 'var(--text)',
                    fontSize: '12px',
                    fontFamily: 'JetBrains Mono, monospace',
                    letterSpacing: '0.02em',
                    transition: 'border-color 0.15s',
                  }}
                />
              </div>

              {error && (
                <div style={{
                  marginBottom: '14px',
                  padding: '10px 14px',
                  background: 'var(--red-dim)',
                  border: '1px solid rgba(242,85,85,0.3)',
                  borderRadius: '6px',
                  color: 'var(--red)',
                  fontSize: '11px',
                  lineHeight: 1.6,
                }}>
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '12px 14px',
                  background: loading
                    ? 'rgba(245,166,35,0.4)'
                    : 'linear-gradient(135deg, #F5A623 0%, #E0920E 100%)',
                  color: '#07080F',
                  fontSize: '11px',
                  fontWeight: 700,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  boxShadow: loading ? 'none' : '0 0 24px rgba(245,166,35,0.2)',
                  transition: 'box-shadow 0.15s, background 0.15s',
                  fontFamily: 'JetBrains Mono, monospace',
                }}
              >
                {loading
                  ? (mode === 'signin' ? 'signing in...' : 'creating account...')
                  : (mode === 'signin' ? 'sign in →' : 'create account →')}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  )
}
