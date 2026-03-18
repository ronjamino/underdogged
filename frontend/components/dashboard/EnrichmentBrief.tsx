'use client'

import { useState, useEffect } from 'react'
import { fetchEnrichment } from '@/lib/api'
import type { EnrichmentItem, EnrichmentResponse } from '@/lib/api'

interface Props {
  section: 'predictions' | 'value-bets'
}

const VERDICT_STYLE = {
  BACK:    { color: 'var(--green)',  bg: 'var(--green-dim)',  border: 'rgba(16,217,122,0.2)',  icon: '↑', label: 'BACK'    },
  MONITOR: { color: 'var(--accent)', bg: 'var(--accent-dim)', border: 'rgba(245,166,35,0.2)',  icon: '◎', label: 'MONITOR' },
  SKIP:    { color: 'var(--red)',    bg: 'var(--red-dim)',    border: 'rgba(242,85,85,0.2)',   icon: '↓', label: 'SKIP'    },
}

function VerdictBadge({ verdict }: { verdict: 'BACK' | 'MONITOR' | 'SKIP' }) {
  const s = VERDICT_STYLE[verdict]
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      padding: '2px 8px',
      background: s.bg,
      border: `1px solid ${s.border}`,
      color: s.color,
      fontSize: '9px',
      letterSpacing: '0.1em',
      fontWeight: 700,
      borderRadius: '3px',
      textTransform: 'uppercase',
    }}>
      {s.icon} {s.label}
    </span>
  )
}

function PickCard({ item }: { item: EnrichmentItem }) {
  const s = VERDICT_STYLE[item.verdict]
  return (
    <div style={{
      flex: '1 1 220px',
      minWidth: '200px',
      maxWidth: '320px',
      padding: '14px 16px',
      background: 'rgba(12,14,26,0.6)',
      border: `1px solid ${s.border}`,
      borderTop: `2px solid ${s.color}`,
      borderRadius: '6px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
        <div>
          <div style={{ color: 'var(--text)', fontSize: '12px', fontWeight: 600, lineHeight: 1.3 }}>
            {item.home_team} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>vs</span> {item.away_team}
          </div>
          {item.market && (
            <div style={{ color: 'var(--text-muted)', fontSize: '10px', marginTop: '2px' }}>
              {item.market}
              {item.edge_pct != null && (
                <span style={{ color: 'var(--green)', marginLeft: '6px' }}>+{item.edge_pct}% edge</span>
              )}
            </div>
          )}
        </div>
        <VerdictBadge verdict={item.verdict} />
      </div>
      <p style={{
        color: 'var(--text-muted)',
        fontSize: '11px',
        lineHeight: 1.6,
        margin: 0,
      }}>
        {item.commentary}
      </p>
      {item.model_confidence != null && (
        <div style={{ marginTop: '8px', color: 'var(--text-muted)', fontSize: '10px' }}>
          Model confidence: <span style={{ color: 'var(--accent)' }}>{item.model_confidence}%</span>
        </div>
      )}
    </div>
  )
}

function Skeleton() {
  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '20px 24px',
      marginBottom: '24px',
      background: 'var(--bg-card)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div className="skeleton" style={{ height: '10px', width: '120px', borderRadius: '3px' }} />
        <div className="skeleton" style={{ height: '10px', width: '80px', borderRadius: '3px' }} />
      </div>
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        {[1, 2, 3].map(i => (
          <div key={i} style={{ flex: '1 1 220px', minWidth: '200px', padding: '14px 16px', border: '1px solid var(--border)', borderRadius: '6px' }}>
            <div className="skeleton" style={{ height: '12px', width: '80%', marginBottom: '8px', borderRadius: '3px' }} />
            <div className="skeleton" style={{ height: '10px', width: '60%', marginBottom: '12px', borderRadius: '3px' }} />
            <div className="skeleton" style={{ height: '10px', width: '95%', marginBottom: '6px', borderRadius: '3px' }} />
            <div className="skeleton" style={{ height: '10px', width: '70%', borderRadius: '3px' }} />
          </div>
        ))}
      </div>
    </div>
  )
}

export function EnrichmentBrief({ section }: Props) {
  const [data, setData] = useState<EnrichmentResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetchEnrichment(section)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [section])

  if (loading) return <Skeleton />
  if (!data || data.items.length === 0) return null

  const back    = data.items.filter(i => i.verdict === 'BACK').length
  const monitor = data.items.filter(i => i.verdict === 'MONITOR').length
  const skip    = data.items.filter(i => i.verdict === 'SKIP').length
  const topPicks = data.items.filter(i => i.verdict !== 'SKIP').slice(0, 3)

  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: '8px',
      marginBottom: '24px',
      background: 'var(--bg-card)',
      overflow: 'hidden',
    }}>
      {/* Header bar */}
      <div
        onClick={() => setCollapsed(c => !c)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 20px',
          borderBottom: collapsed ? 'none' : '1px solid var(--border)',
          cursor: 'pointer',
          background: 'rgba(28,32,64,0.2)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{
            fontSize: '9px',
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            fontWeight: 700,
            color: 'var(--accent)',
          }}>
            AI Brief
          </span>
          <div style={{ display: 'flex', gap: '8px' }}>
            {back > 0    && <span style={{ fontSize: '10px', color: 'var(--green)'  }}>↑ {back} back</span>}
            {monitor > 0 && <span style={{ fontSize: '10px', color: 'var(--accent)' }}>◎ {monitor} monitor</span>}
            {skip > 0    && <span style={{ fontSize: '10px', color: 'var(--red)'    }}>↓ {skip} skip</span>}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
            {data.run_date}
          </span>
          <span style={{
            color: 'var(--text-muted)',
            fontSize: '10px',
            transform: collapsed ? 'rotate(180deg)' : 'none',
            transition: 'transform 0.2s',
          }}>
            ▾
          </span>
        </div>
      </div>

      {/* Cards */}
      {!collapsed && (
        <div style={{ padding: '16px 20px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {topPicks.map((item, i) => (
            <PickCard key={i} item={item} />
          ))}
          {data.items.filter(i => i.verdict === 'SKIP').length > 0 && (
            <div style={{ width: '100%', paddingTop: '8px', borderTop: '1px solid var(--border)', marginTop: '4px' }}>
              <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px' }}>
                Skip
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {data.items.filter(i => i.verdict === 'SKIP').map((item, i) => (
                  <div key={i} style={{
                    padding: '6px 12px',
                    background: 'var(--red-dim)',
                    border: '1px solid rgba(242,85,85,0.15)',
                    borderRadius: '4px',
                    fontSize: '11px',
                  }}>
                    <span style={{ color: 'var(--text)' }}>{item.home_team} vs {item.away_team}</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: '8px', fontSize: '10px' }}>{item.commentary}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
