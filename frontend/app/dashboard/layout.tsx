'use client'

export const dynamic = 'force-dynamic'

import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { useEffect, useState } from 'react'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [email, setEmail] = useState<string | null>(null)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      setEmail(data.user?.email ?? null)
    })
  }, [])

  async function handleSignOut() {
    await supabase.auth.signOut()
    router.push('/login')
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      {/* Ambient background glow */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '800px',
        height: '300px',
        background: 'radial-gradient(ellipse at 50% 0%, rgba(245,166,35,0.06) 0%, transparent 70%)',
        pointerEvents: 'none',
        zIndex: 0,
      }} />

      {/* Glassmorphic nav */}
      <nav style={{
        height: '52px',
        borderBottom: '1px solid rgba(28,32,64,0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 28px',
        position: 'sticky',
        top: 0,
        background: 'rgba(7,8,15,0.75)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        zIndex: 100,
      }}>
        {/* Logo with gradient */}
        <span className="font-display" style={{
          fontSize: '22px',
          fontStyle: 'italic',
          background: 'linear-gradient(135deg, #F5A623 0%, #FFD07A 60%, #F5A623 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          letterSpacing: '-0.01em',
        }}>
          underdogged
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          {email && (
            <span style={{
              color: 'var(--text-muted)',
              fontSize: '11px',
              letterSpacing: '0.02em',
            }}>
              {email}
            </span>
          )}
          <button
            onClick={handleSignOut}
            style={{
              background: 'rgba(28,32,64,0.5)',
              border: '1px solid var(--border)',
              color: 'var(--text-muted)',
              padding: '5px 14px',
              fontSize: '10px',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              cursor: 'pointer',
              borderRadius: '4px',
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => {
              const b = e.currentTarget
              b.style.color = 'var(--text)'
              b.style.borderColor = 'var(--text-muted)'
              b.style.background = 'rgba(28,32,64,0.9)'
            }}
            onMouseLeave={e => {
              const b = e.currentTarget
              b.style.color = 'var(--text-muted)'
              b.style.borderColor = 'var(--border)'
              b.style.background = 'rgba(28,32,64,0.5)'
            }}
          >
            sign out
          </button>
        </div>
      </nav>

      {/* Content */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '0 24px',
        position: 'relative',
        zIndex: 1,
      }}>
        {children}
      </div>
    </div>
  )
}
