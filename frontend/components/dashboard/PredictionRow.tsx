'use client'

import { ConfidenceBar } from './ConfidenceBar'
import type { Prediction } from '@/lib/api'

interface Props {
  prediction: Prediction
}

const OUTCOME_STYLES: Record<string, { bg: string; color: string; label: (p: Prediction) => string }> = {
  H: { bg: 'var(--green-dim)', color: 'var(--green)',  label: p => p.home_team },
  D: { bg: 'var(--accent-dim)', color: 'var(--accent)', label: _ => 'Draw' },
  A: { bg: 'var(--red-dim)',   color: 'var(--red)',    label: p => p.away_team },
}

export function PredictionRow({ prediction: p }: Props) {
  const outcome = OUTCOME_STYLES[p.predicted_outcome] ?? OUTCOME_STYLES['D']
  const label = outcome.label(p)

  const date = new Date(p.match_date)
  const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  const timeStr = date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })

  return (
    <tr style={{
      borderBottom: '1px solid var(--border)',
      transition: 'background 0.1s',
    }}
      onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)'}
      onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = 'transparent'}
    >
      {/* Date */}
      <td style={{ padding: '0 16px', color: 'var(--text-muted)', fontSize: '11px', width: '80px', whiteSpace: 'nowrap' }}>
        {dateStr}
      </td>

      {/* Kickoff */}
      <td style={{ padding: '0 16px', color: 'var(--text-muted)', fontSize: '11px', width: '60px', whiteSpace: 'nowrap' }}>
        {timeStr}
      </td>

      {/* Match */}
      <td style={{ padding: '0 16px' }}>
        <span style={{ fontWeight: 600, color: 'var(--text)' }}>{p.home_team}</span>
        <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>vs</span>
        <span style={{ color: 'var(--text)' }}>{p.away_team}</span>
      </td>

      {/* Prediction badge */}
      <td style={{ padding: '0 16px', width: '140px', whiteSpace: 'nowrap' }}>
        <span style={{
          display: 'inline-block',
          padding: '3px 8px',
          background: outcome.bg,
          color: outcome.color,
          fontSize: '11px',
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
          fontWeight: 500,
        }}>
          {label}
        </span>
      </td>

      {/* Confidence */}
      <td style={{ padding: '0 16px', width: '130px' }}>
        <ConfidenceBar value={p.confidence} />
      </td>
    </tr>
  )
}
