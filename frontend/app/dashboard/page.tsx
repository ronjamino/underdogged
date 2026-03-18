'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { LeagueSelector } from '@/components/dashboard/LeagueSelector'
import { PredictionsTable } from '@/components/dashboard/PredictionsTable'
import { ValueBetsTable } from '@/components/dashboard/ValueBetsTable'
import { PerformanceTab } from '@/components/dashboard/PerformanceTab'
import { fetchPredictions, fetchValueBets, fetchPerformance, fetchLiveRecord } from '@/lib/api'
import type { Prediction, PerformanceSummary, LiveRecord } from '@/lib/api'
import { EnrichmentBrief } from '@/components/dashboard/EnrichmentBrief'

const DEFAULT_LEAGUE = 'PL'
type Tab = 'predictions' | 'value' | 'performance'

const TABS: { id: Tab; label: string }[] = [
  { id: 'predictions', label: 'Predictions' },
  { id: 'value',       label: 'Value Bets'  },
  { id: 'performance', label: 'Performance' },
]

export default function DashboardPage() {
  const [tab, setTab] = useState<Tab>('predictions')
  const [activeLeague, setActiveLeague] = useState(DEFAULT_LEAGUE)
  const [valueLeague, setValueLeague] = useState(DEFAULT_LEAGUE)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [allValueBets, setAllValueBets] = useState<Prediction[]>([])
  const [performance, setPerformance] = useState<PerformanceSummary | null>(null)
  const [liveRecord, setLiveRecord] = useState<LiveRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

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

  return (
    <main style={{ padding: '28px 0' }}>
      {/* Top-level tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        marginBottom: '28px',
        gap: '4px',
      }}>
        {TABS.map(({ id, label }) => {
          const isActive = tab === id
          return (
            <button
              key={id}
              onClick={() => setTab(id)}
              style={{
                padding: '14px 22px',
                fontSize: '11px',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                fontWeight: 500,
                cursor: 'pointer',
                background: 'none',
                border: 'none',
                borderBottom: isActive
                  ? '2px solid var(--accent)'
                  : '2px solid transparent',
                color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                textShadow: isActive ? '0 0 20px var(--accent-glow)' : 'none',
                transition: 'color 0.15s, text-shadow 0.15s',
                fontFamily: 'JetBrains Mono, monospace',
                whiteSpace: 'nowrap',
                marginBottom: '-1px',
              }}
              onMouseEnter={e => {
                if (!isActive) e.currentTarget.style.color = 'var(--text)'
              }}
              onMouseLeave={e => {
                if (!isActive) e.currentTarget.style.color = 'var(--text-muted)'
              }}
            >
              {label}
            </button>
          )
        })}
      </div>

      {tab === 'predictions' && (
        <>
          <EnrichmentBrief section="predictions" />
          <LeagueSelector active={activeLeague} onChange={setActiveLeague} />
          <div style={{ marginTop: '24px' }}>
            <PredictionsTable predictions={predictions} loading={loading} error={error} />
          </div>
        </>
      )}

      {tab === 'value' && (
        <>
          <EnrichmentBrief section="value-bets" />
          <LeagueSelector active={valueLeague} onChange={setValueLeague} />
          <div style={{ marginTop: '24px' }}>
            <ValueBetsTable predictions={valueBets} loading={loading} error={error} />
          </div>
        </>
      )}

      {tab === 'performance' && (
        <PerformanceTab data={performance} live={liveRecord} loading={loading} error={error} />
      )}
    </main>
  )
}
