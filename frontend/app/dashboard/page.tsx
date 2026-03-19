'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { LeagueSelector } from '@/components/dashboard/LeagueSelector'
import { PredictionsTable } from '@/components/dashboard/PredictionsTable'
import { ValueBetsTable } from '@/components/dashboard/ValueBetsTable'
import { PerformanceTab } from '@/components/dashboard/PerformanceTab'
import { fetchPredictions, fetchValueBets, fetchPerformance, fetchLiveRecord, fetchEnrichment, fetchLastUpdated } from '@/lib/api'
import type { Prediction, PerformanceSummary, LiveRecord, EnrichmentItem } from '@/lib/api'

const DEFAULT_LEAGUE = 'PL'
type Tab = 'predictions' | 'value' | 'performance'

const TABS: { id: Tab; label: string }[] = [
  { id: 'predictions', label: 'Predictions' },
  { id: 'value',       label: 'Value Bets'  },
  { id: 'performance', label: 'Performance' },
]

function enrichmentKey(homeTeam: string, awayTeam: string) {
  return `${homeTeam}|${awayTeam}`
}

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
  const [predEnrichment, setPredEnrichment] = useState<Map<string, EnrichmentItem>>(new Map())
  const [valueEnrichment, setValueEnrichment] = useState<Map<string, EnrichmentItem>>(new Map())
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)

  // Fetch last updated timestamp once on mount
  useEffect(() => {
    fetchLastUpdated().then(ts => { if (ts) setLastUpdated(ts) })
  }, [])

  // Fetch enrichment in background whenever tab switches to predictions or value
  useEffect(() => {
    if (tab === 'predictions') {
      fetchEnrichment('predictions').then(r => {
        const map = new Map(r.items.map(i => [enrichmentKey(i.home_team, i.away_team), i]))
        setPredEnrichment(map)
      }).catch(() => {})
    } else if (tab === 'value') {
      fetchEnrichment('value-bets').then(r => {
        const map = new Map(r.items.map(i => [enrichmentKey(i.home_team, i.away_team), i]))
        setValueEnrichment(map)
      }).catch(() => {})
    }
  }, [tab])

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

      {/* Top-level tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        marginBottom: '24px',
        gap: '4px',
        overflowX: 'auto',
        scrollbarWidth: 'none',
        msOverflowStyle: 'none',
      }}>
        {TABS.map(({ id, label }) => {
          const isActive = tab === id
          return (
            <button
              key={id}
              onClick={() => setTab(id)}
              style={{
                padding: '12px 16px',
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
          <LeagueSelector active={activeLeague} onChange={setActiveLeague} />
          <div style={{ marginTop: '24px' }}>
            <PredictionsTable
              predictions={predictions}
              loading={loading}
              error={error}
              enrichmentMap={predEnrichment}
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
              enrichmentMap={valueEnrichment}
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
