'use client'

import { useState } from 'react'
import { ConfidenceBar } from './ConfidenceBar'
import { ExpandedDetail } from './ExpandedDetail'
import type { Prediction } from '@/lib/api'

interface Props {
  prediction: Prediction
}

const OUTCOME_STYLES: Record<string, { bg: string; border: string; color: string; label: (p: Prediction) => string }> = {
  H: { bg: 'var(--green-dim)', border: 'rgba(16,217,122,0.2)',  color: 'var(--green)',  label: p => p.home_team },
  D: { bg: 'var(--accent-dim)', border: 'rgba(245,166,35,0.2)', color: 'var(--accent)', label: _ => 'Draw' },
  A: { bg: 'var(--red-dim)',    border: 'rgba(242,85,85,0.2)',  color: 'var(--red)',    label: p => p.away_team },
}

const COL_SPAN = 5

export function PredictionRow({ prediction: p }: Props) {
  const [expanded, setExpanded] = useState(false)
  const outcome = OUTCOME_STYLES[p.predicted_outcome] ?? OUTCOME_STYLES['D']
  const label = outcome.label(p)

  const date = new Date(p.match_date)
  const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  const timeStr = date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })

  return (
    <>
      <tr
        onClick={() => setExpanded(e => !e)}
        style={{
          borderBottom: expanded ? 'none' : '1px solid var(--border)',
          transition: 'background 0.15s',
          cursor: 'pointer',
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
            padding: '3px 10px',
            background: outcome.bg,
            border: `1px solid ${outcome.border}`,
            color: outcome.color,
            fontSize: '10px',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            fontWeight: 600,
            borderRadius: '4px',
          }}>
            {label}
          </span>
        </td>

        {/* Confidence + expand chevron */}
        <td style={{ padding: '0 16px', width: '130px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ flex: 1 }}>
              <ConfidenceBar value={p.confidence} />
            </div>
            <span style={{
              color: 'var(--text-muted)',
              fontSize: '10px',
              transform: expanded ? 'rotate(180deg)' : 'none',
              transition: 'transform 0.2s',
              flexShrink: 0,
            }}>
              ▾
            </span>
          </div>
        </td>
      </tr>

      {expanded && <ExpandedDetail p={p} colSpan={COL_SPAN} />}
    </>
  )
}
