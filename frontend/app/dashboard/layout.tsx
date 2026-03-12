'use client'

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
      {/* Nav */}
      <nav style={{
        height: '48px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        position: 'sticky',
        top: 0,
        background: 'var(--bg)',
        zIndex: 10,
      }}>
        <span className="font-display" style={{
          fontSize: '20px',
          fontStyle: 'italic',
          color: 'var(--text)',
        }}>
          underdogged
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          {email && (
            <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
              {email}
            </span>
          )}
          <button
            onClick={handleSignOut}
            style={{
              background: 'none',
              border: '1px solid var(--border)',
              color: 'var(--text-muted)',
              padding: '5px 12px',
              fontSize: '11px',
              cursor: 'pointer',
              letterSpacing: '0.05em',
              transition: 'color 0.15s, border-color 0.15s',
            }}
            onMouseEnter={e => {
              (e.target as HTMLButtonElement).style.color = 'var(--text)'
              ;(e.target as HTMLButtonElement).style.borderColor = 'var(--text-muted)'
            }}
            onMouseLeave={e => {
              (e.target as HTMLButtonElement).style.color = 'var(--text-muted)'
              ;(e.target as HTMLButtonElement).style.borderColor = 'var(--border)'
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
      }}>
        {children}
      </div>
    </div>
  )
}
