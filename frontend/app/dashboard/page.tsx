'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { LeagueSelector } from '@/components/dashboard/LeagueSelector'
import { PredictionsTable } from '@/components/dashboard/PredictionsTable'
import { ValueBetsTable } from '@/components/dashboard/ValueBetsTable'
import { fetchPredictions, fetchValueBets } from '@/lib/api'
import type { Prediction } from '@/lib/api'

const DEFAULT_LEAGUE = 'PL'
type Tab = 'predictions' | 'value'

export default function DashboardPage() {
  const [tab, setTab] = useState<Tab>('predictions')
  const [activeLeague, setActiveLeague] = useState(DEFAULT_LEAGUE)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [valueBets, setValueBets] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (tab === 'predictions') {
      setLoading(true)
      setError('')
      fetchPredictions(activeLeague)
        .then(setPredictions)
        .catch(() => setError('Failed to load predictions. Check API connection.'))
        .finally(() => setLoading(false))
    } else {
      setLoading(true)
      setError('')
      fetchValueBets()
        .then(setValueBets)
        .catch(() => setError('Failed to load value bets. Check API connection.'))
        .finally(() => setLoading(false))
    }
  }, [tab, activeLeague])

  const tabStyle = (t: Tab) => ({
    padding: '6px 16px',
    fontSize: '11px',
    letterSpacing: '0.08em',
    textTransform: 'uppercase' as const,
    fontWeight: 500,
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    borderBottom: tab === t ? '1px solid var(--accent)' : '1px solid transparent',
    color: tab === t ? 'var(--accent)' : 'var(--text-muted)',
    transition: 'color 0.15s',
  })

  return (
    <main style={{ padding: '24px 0' }}>
      {/* Tabs */}
      <div style={{ display: 'flex', gap: '4px', borderBottom: '1px solid var(--border)', marginBottom: '24px' }}>
        <button style={tabStyle('predictions')} onClick={() => setTab('predictions')}>
          Predictions
        </button>
        <button style={tabStyle('value')} onClick={() => setTab('value')}>
          Value Bets
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
        <ValueBetsTable predictions={valueBets} loading={loading} error={error} />
      )}
    </main>
  )
}
