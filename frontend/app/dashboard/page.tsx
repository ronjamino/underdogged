'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { LeagueSelector } from '@/components/dashboard/LeagueSelector'
import { PredictionsTable } from '@/components/dashboard/PredictionsTable'
import { ValueBetsTable } from '@/components/dashboard/ValueBetsTable'
import { PerformanceTab } from '@/components/dashboard/PerformanceTab'
import { fetchPredictions, fetchValueBets, fetchPerformance, fetchLiveRecord } from '@/lib/api'
import type { Prediction, PerformanceSummary, LiveRecord } from '@/lib/api'

const DEFAULT_LEAGUE = 'PL'
type Tab = 'predictions' | 'value' | 'performance'

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

  const tabStyle = (t: Tab) => ({
    padding: '14px 20px',
    fontSize: '11px',
    letterSpacing: '0.1em',
    textTransform: 'uppercase' as const,
    fontWeight: 500,
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent',
    color: tab === t ? 'var(--accent)' : 'var(--text-muted)',
    transition: 'color 0.15s',
    fontFamily: 'JetBrains Mono, monospace',
    whiteSpace: 'nowrap' as const,
    marginBottom: '-1px',
  })

  return (
    <main style={{ padding: '24px 0' }}>
      {/* Top-level tab bar */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: '24px' }}>
        <button style={tabStyle('predictions')} onClick={() => setTab('predictions')}>
          Predictions
        </button>
        <button style={tabStyle('value')} onClick={() => setTab('value')}>
          Value Bets
        </button>
        <button style={tabStyle('performance')} onClick={() => setTab('performance')}>
          Performance
        </button>
      </div>

      {tab === 'predictions' && (
        <>
          <LeagueSelector active={activeLeague} onChange={setActiveLeague} />
          <div style={{ marginTop: '24px' }}>
            <PredictionsTable predictions={predictions} loading={loading} error={error} />
          </div>
        </>
      )}

      {tab === 'value' && (
        <>
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
