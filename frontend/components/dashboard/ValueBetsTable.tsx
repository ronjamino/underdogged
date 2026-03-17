'use client'

import type { Prediction } from '@/lib/api'

interface Props {
  predictions: Prediction[]
  loading: boolean
  error: string
}

const OUTCOME_LABEL: Record<string, string> = { H: 'Home', D: 'Draw', A: 'Away' }
const OUTCOME_COLOR: Record<string, string> = {
  H: 'var(--green)',
  D: 'var(--accent)',
  A: 'var(--red)',
}
const OUTCOME_BG: Record<string, string> = {
  H: 'var(--green-dim)',
  D: 'var(--accent-dim)',
  A: 'var(--red-dim)',
}

function oddsToProb(odds: number) {
  return 1 / odds
}

function edge(modelProb: number, odds: number) {
  const total = oddsToProb(odds)
  // bookmaker fair prob (single-market approximation)
  const bookProb = total
  return (modelProb - bookProb) * 100
}

function ValueBetRow({ p }: { p: Prediction }) {
  const vb = p.value_bet!
  const modelProb =
    vb === 'H' ? p.prob_home : vb === 'D' ? p.prob_draw : p.prob_away
  const odds =
    vb === 'H' ? p.odds_home : vb === 'D' ? p.odds_draw : p.odds_away
  const edgePct = odds ? edge(modelProb, odds) : null

  const date = new Date(p.match_date)
  const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })

  return (
    <tr
      style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.1s' }}
      onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)'}
      onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = 'transparent'}
    >
      {/* Date */}
      <td style={{ padding: '0 16px', color: 'var(--text-muted)', fontSize: '11px', width: '80px', whiteSpace: 'nowrap' }}>
        {dateStr}
      </td>

      {/* Match */}
      <td style={{ padding: '0 16px' }}>
        <span style={{ fontWeight: 600, color: 'var(--text)' }}>{p.home_team}</span>
        <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>vs</span>
        <span style={{ color: 'var(--text)' }}>{p.away_team}</span>
      </td>

      {/* Bet */}
      <td style={{ padding: '0 16px', width: '100px', whiteSpace: 'nowrap' }}>
        <span style={{
          display: 'inline-block',
          padding: '3px 8px',
          background: OUTCOME_BG[vb],
          color: OUTCOME_COLOR[vb],
          fontSize: '11px',
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
          fontWeight: 500,
        }}>
          {OUTCOME_LABEL[vb]}
        </span>
      </td>

      {/* Odds */}
      <td style={{ padding: '0 16px', width: '70px', color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
        {odds != null ? odds.toFixed(2) : '—'}
      </td>

      {/* Model prob */}
      <td style={{ padding: '0 16px', width: '90px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
        {(modelProb * 100).toFixed(1)}%
      </td>

      {/* Edge */}
      <td style={{ padding: '0 16px', width: '80px', fontVariantNumeric: 'tabular-nums',
        color: edgePct != null && edgePct > 0 ? 'var(--green)' : 'var(--text-muted)' }}>
        {edgePct != null ? `+${edgePct.toFixed(1)}%` : '—'}
      </td>
    </tr>
  )
}

function SkeletonRow() {
  return (
    <tr style={{ borderBottom: '1px solid var(--border)', height: '48px' }}>
      {[80, 300, 100, 70, 90, 80].map((w, i) => (
        <td key={i} style={{ padding: '0 16px', width: `${w}px` }}>
          <div className="skeleton" style={{ height: '12px', width: '70%' }} />
        </td>
      ))}
    </tr>
  )
}

export function ValueBetsTable({ predictions, loading, error }: Props) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Date', 'Match', 'Bet', 'Odds', 'Model', 'Edge'].map(col => (
              <th key={col} style={{
                padding: '10px 16px',
                textAlign: 'left',
                color: 'var(--text-muted)',
                fontSize: '10px',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                fontWeight: 500,
              }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody style={{ fontSize: '13px' }}>
          {loading && Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)}

          {!loading && error && (
            <tr>
              <td colSpan={6} style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--red)', fontSize: '12px' }}>
                {error}
              </td>
            </tr>
          )}

          {!loading && !error && predictions.length === 0 && (
            <tr>
              <td colSpan={6} style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
                No value bets found this gameweek.
              </td>
            </tr>
          )}

          {!loading && !error && predictions.map(p => (
            <ValueBetRow key={p.match_id} p={p} />
          ))}
        </tbody>
      </table>
    </div>
  )
}
