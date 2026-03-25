'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { LeagueSelector } from '@/components/dashboard/LeagueSelector'
import { PredictionsTable } from '@/components/dashboard/PredictionsTable'
import { ValueBetsTable } from '@/components/dashboard/ValueBetsTable'
import { PerformanceTab } from '@/components/dashboard/PerformanceTab'
import { fetchPredictions, fetchValueBets, fetchPerformance, fetchLiveRecord, fetchLastUpdated } from '@/lib/api'
import type { Prediction, PerformanceSummary, LiveRecord } from '@/lib/api'

const DEFAULT_LEAGUE = 'PL'
type Tab = 'predictions' | 'value' | 'performance'

const TABS: { id: Tab; label: string }[] = [
  { id: 'predictions', label: 'Predictions' },
  { id: 'value',       label: 'Value Bets'  },
  { id: 'performance', label: 'Performance' },
]


export default function DashboardPage() {
  const [tab, setTab] = useState<Tab>('predictions')
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeLeague, setActiveLeague] = useState(DEFAULT_LEAGUE)
  const [valueLeague, setValueLeague] = useState(DEFAULT_LEAGUE)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [allValueBets, setAllValueBets] = useState<Prediction[]>([])
  const [performance, setPerformance] = useState<PerformanceSummary | null>(null)
  const [liveRecord, setLiveRecord] = useState<LiveRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)

  // Fetch last updated timestamp once on mount
  useEffect(() => {
    fetchLastUpdated().then(ts => { if (ts) setLastUpdated(ts) })
  }, [])

  // Close hamburger menu on outside click
  useEffect(() => {
    if (!menuOpen) return
    function handler(e: MouseEvent) {
      if (!(e.target as Element).closest('#tab-menu')) setMenuOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [menuOpen])


  useEffect(() => {
    setLoading(true)
    setError('')
    if (tab === 'predictions') {
      fetchPredictions(activeLeague)
        .then(setPredictions)
        .catch(() => setError('Failed to load predictions. Check API connection.'))
        .finally(() => setLoading(false))
    } else if (tab === 'value') {
      fetchValueBets()
        .then(setAllValueBets)
        .catch(() => setError('Failed to load value bets. Check API connection.'))
        .finally(() => setLoading(false))
    } else {
      Promise.all([fetchPerformance(), fetchLiveRecord()])
        .then(([perf, live]) => { setPerformance(perf); setLiveRecord(live) })
        .catch(() => setError('Failed to load performance data.'))
        .finally(() => setLoading(false))
    }
  }, [tab, activeLeague])

  const valueBets = allValueBets.filter(p => p.league.toUpperCase() === valueLeague)

  const lastUpdatedStr = lastUpdated
    ? new Date(lastUpdated).toLocaleString('en-GB', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short',
      })
    : null

  return (
    <main style={{ padding: '20px 0' }}>
      {/* Last updated */}
      {lastUpdatedStr && (
        <div style={{ textAlign: 'right', marginBottom: '8px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
            Updated {lastUpdatedStr}
          </span>
        </div>
      )}

      {/* Hamburger menu */}
      <div id="tab-menu" style={{ position: 'relative', display: 'inline-block', marginBottom: '20px' }}>
        <button
          onClick={() => setMenuOpen(o => !o)}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '6px', padding: '7px 12px',
            color: 'var(--text)', fontSize: '11px',
            letterSpacing: '0.08em', textTransform: 'uppercase',
            cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
            transition: 'border-color 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--text-muted)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)' }}
        >
          <span style={{ fontSize: '14px', lineHeight: 1 }}>☰</span>
          {TABS.find(t => t.id === tab)?.label}
        </button>

        {menuOpen && (
          <div style={{
            position: 'absolute', top: 'calc(100% + 4px)', left: 0,
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '6px', overflow: 'hidden',
            boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
            zIndex: 50, minWidth: '160px',
          }}>
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => { setTab(id); setMenuOpen(false) }}
                style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  padding: '10px 14px', background: 'none', border: 'none',
                  borderLeft: tab === id ? '2px solid var(--accent)' : '2px solid transparent',
                  color: tab === id ? 'var(--accent)' : 'var(--text-muted)',
                  fontSize: '11px', letterSpacing: '0.08em', textTransform: 'uppercase',
                  cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
                  transition: 'color 0.1s, background 0.1s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'var(--bg-hover)'
                  if (tab !== id) e.currentTarget.style.color = 'var(--text)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'none'
                  if (tab !== id) e.currentTarget.style.color = 'var(--text-muted)'
                }}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>

      {tab === 'predictions' && (
        <>
          <LeagueSelector active={activeLeague} onChange={setActiveLeague} />
          <div style={{ marginTop: '24px' }}>
            <PredictionsTable
              predictions={predictions}
              loading={loading}
              error={error}
            />
          </div>
        </>
      )}

      {tab === 'value' && (
        <>
          <LeagueSelector active={valueLeague} onChange={setValueLeague} />
          <div style={{ marginTop: '24px' }}>
            <ValueBetsTable
              predictions={valueBets}
              loading={loading}
              error={error}
            />
          </div>
        </>
      )}

      {tab === 'performance' && (
        <PerformanceTab data={performance} live={liveRecord} loading={loading} error={error} />
      )}
    </main>
  )
}
