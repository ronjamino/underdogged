'use client'

import { useState } from 'react'
import { ExpandedDetail } from './ExpandedDetail'
import { useIsMobile } from '@/lib/hooks/useIsMobile'
import type { Prediction, EnrichmentItem } from '@/lib/api'

interface Props {
  predictions: Prediction[]
  loading: boolean
  error: string
  enrichmentMap: Map<string, EnrichmentItem>
}

const OUTCOME_LABEL: Record<string, string> = { H: 'Home', D: 'Draw', A: 'Away' }
const OUTCOME_COLOR: Record<string, string> = { H: 'var(--green)', D: 'var(--accent)', A: 'var(--red)' }
const OUTCOME_BG:    Record<string, string> = { H: 'var(--green-dim)', D: 'var(--accent-dim)', A: 'var(--red-dim)' }
const OUTCOME_BORDER: Record<string, string> = {
  H: 'rgba(16,217,122,0.2)', D: 'rgba(245,166,35,0.2)', A: 'rgba(242,85,85,0.2)',
}
const VERDICT_COLOR: Record<string, string> = {
  BACK: 'var(--green)', MONITOR: 'var(--accent)', SKIP: 'var(--red)',
}

const COL_SPAN = 6

function edge(modelProb: number, odds: number) {
  return (modelProb - 1 / odds) * 100
}

function ValueBetRow({ p, enrichment }: { p: Prediction; enrichment?: EnrichmentItem }) {
  const [expanded, setExpanded] = useState(false)
  const vb = p.value_bet!
  const modelProb = vb === 'H' ? p.prob_home : vb === 'D' ? p.prob_draw : p.prob_away
  const odds = vb === 'H' ? p.odds_home : vb === 'D' ? p.odds_draw : p.odds_away
  const edgePct = odds ? edge(modelProb, odds) : null

  const date = new Date(p.match_date)
  const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })

  return (
    <>
      <tr
        onClick={() => setExpanded(e => !e)}
        style={{ borderBottom: expanded ? 'none' : '1px solid var(--border)', transition: 'background 0.15s', cursor: 'pointer' }}
        onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)'}
        onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = 'transparent'}
      >
        <td style={{ padding: '0 16px', color: 'var(--text-muted)', fontSize: '11px', width: '80px', whiteSpace: 'nowrap' }}>
          {dateStr}
        </td>
        <td style={{ padding: '0 16px' }}>
          <span style={{ fontWeight: 600, color: 'var(--text)' }}>{p.home_team}</span>
          <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>vs</span>
          <span style={{ color: 'var(--text)' }}>{p.away_team}</span>
        </td>
        <td style={{ padding: '0 16px', width: '100px', whiteSpace: 'nowrap' }}>
          <span style={{
            display: 'inline-block', padding: '3px 10px',
            background: OUTCOME_BG[vb], color: OUTCOME_COLOR[vb],
            border: `1px solid ${OUTCOME_BORDER[vb]}`,
            fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600,
            borderRadius: '4px',
          }}>
            {OUTCOME_LABEL[vb]}
          </span>
        </td>
        <td style={{ padding: '0 16px', width: '70px', color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
          {odds != null ? odds.toFixed(2) : '—'}
        </td>
        <td style={{ padding: '0 16px', width: '90px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
          {(modelProb * 100).toFixed(1)}%
        </td>
        <td style={{ padding: '0 16px', width: '110px', fontVariantNumeric: 'tabular-nums' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: edgePct != null && edgePct > 0 ? 'var(--green)' : 'var(--text-muted)' }}>
              {edgePct != null ? `+${edgePct.toFixed(1)}%` : '—'}
            </span>
            <span
              title={enrichment ? `${enrichment.verdict}: ${enrichment.commentary}` : undefined}
              style={{
                width: '15px',
                flexShrink: 0,
                fontSize: '13px',
                color: enrichment ? (VERDICT_COLOR[enrichment.verdict] ?? 'var(--text-muted)') : 'transparent',
                filter: enrichment ? `drop-shadow(0 0 4px ${VERDICT_COLOR[enrichment.verdict] ?? 'transparent'})` : 'none',
              }}
            >
              {enrichment ? '💡' : ''}
            </span>
            <span style={{
              color: 'var(--text-muted)', fontSize: '10px',
              transform: expanded ? 'rotate(180deg)' : 'none',
              transition: 'transform 0.2s',
              flexShrink: 0,
            }}>▾</span>
          </div>
        </td>
      </tr>
      {expanded && <ExpandedDetail p={p} colSpan={COL_SPAN} enrichment={enrichment} />}
    </>
  )
}

function MobileValueBetCard({ p, enrichment }: { p: Prediction; enrichment?: EnrichmentItem }) {
  const [expanded, setExpanded] = useState(false)
  const vb = p.value_bet!
  const modelProb = vb === 'H' ? p.prob_home : vb === 'D' ? p.prob_draw : p.prob_away
  const odds = vb === 'H' ? p.odds_home : vb === 'D' ? p.odds_draw : p.odds_away
  const edgePct = odds ? edge(modelProb, odds) : null

  const date = new Date(p.match_date)
  const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })

  return (
    <div
      onClick={() => setExpanded(e => !e)}
      style={{
        borderBottom: '1px solid var(--border)',
        padding: '14px 16px',
        cursor: 'pointer',
        transition: 'background 0.15s',
      }}
      onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-hover)'}
      onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background = 'transparent'}
    >
      {/* Date + chevron */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{dateStr}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {enrichment && (
            <span style={{
              fontSize: '12px',
              color: VERDICT_COLOR[enrichment.verdict] ?? 'var(--text-muted)',
              filter: `drop-shadow(0 0 4px ${VERDICT_COLOR[enrichment.verdict] ?? 'transparent'})`,
            }}>💡</span>
          )}
          <span style={{
            color: 'var(--text-muted)', fontSize: '10px',
            transform: expanded ? 'rotate(180deg)' : 'none',
            transition: 'transform 0.2s',
          }}>▾</span>
        </div>
      </div>

      {/* Match name */}
      <div style={{ marginBottom: '10px' }}>
        <span style={{ fontWeight: 600, color: 'var(--text)' }}>{p.home_team}</span>
        <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>vs</span>
        <span style={{ color: 'var(--text)' }}>{p.away_team}</span>
      </div>

      {/* Outcome badge + odds + edge inline */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <span style={{
          display: 'inline-block', padding: '3px 10px',
          background: OUTCOME_BG[vb], color: OUTCOME_COLOR[vb],
          border: `1px solid ${OUTCOME_BORDER[vb]}`,
          fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase',
          fontWeight: 600, borderRadius: '4px', flexShrink: 0,
        }}>
          {OUTCOME_LABEL[vb]}
        </span>
        <span style={{ color: 'var(--text)', fontSize: '12px', fontVariantNumeric: 'tabular-nums' }}>
          {odds != null ? odds.toFixed(2) : '—'}
        </span>
        <span style={{ color: edgePct != null && edgePct > 0 ? 'var(--green)' : 'var(--text-muted)', fontSize: '12px', fontVariantNumeric: 'tabular-nums' }}>
          {edgePct != null ? `+${edgePct.toFixed(1)}%` : '—'}
        </span>
      </div>

      {expanded && <ExpandedDetail p={p} colSpan={1} enrichment={enrichment} variant="card" />}
    </div>
  )
}

function SkeletonRow() {
  return (
    <tr style={{ borderBottom: '1px solid var(--border)', height: '52px' }}>
      {[80, 300, 100, 70, 90, 80].map((w, i) => (
        <td key={i} style={{ padding: '0 16px', width: `${w}px` }}>
          <div className="skeleton" style={{ height: '10px', width: '70%', borderRadius: '3px' }} />
        </td>
      ))}
    </tr>
  )
}

function MobileSkeletonCard() {
  return (
    <div style={{ borderBottom: '1px solid var(--border)', padding: '14px 16px' }}>
      <div className="skeleton" style={{ height: '10px', width: '25%', marginBottom: '10px', borderRadius: '3px' }} />
      <div className="skeleton" style={{ height: '12px', width: '65%', marginBottom: '10px', borderRadius: '3px' }} />
      <div className="skeleton" style={{ height: '24px', width: '55%', borderRadius: '3px' }} />
    </div>
  )
}

export function ValueBetsTable({ predictions, loading, error, enrichmentMap }: Props) {
  const isMobile = useIsMobile()

  const enrichmentFor = (p: Prediction) =>
    enrichmentMap.get(`${p.home_team}|${p.away_team}`)

  if (isMobile) {
    return (
      <div style={{
        border: '1px solid var(--border)',
        borderRadius: '8px',
        background: 'var(--bg-card)',
        overflow: 'hidden',
      }}>
        {loading && Array.from({ length: 5 }).map((_, i) => <MobileSkeletonCard key={i} />)}

        {!loading && error && (
          <div style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--red)', fontSize: '12px' }}>{error}</div>
        )}

        {!loading && !error && predictions.length === 0 && (
          <div style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
            No value bets found this gameweek.
          </div>
        )}

        {!loading && !error && predictions.map(p => (
          <MobileValueBetCard
            key={p.match_id}
            p={p}
            enrichment={enrichmentFor(p)}
          />
        ))}
      </div>
    )
  }

  return (
    <div style={{
      overflowX: 'auto',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      background: 'var(--bg-card)',
    }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Date', 'Match', 'Bet', 'Odds', 'Model', 'Edge'].map(col => (
              <th key={col} style={{
                padding: '12px 16px', textAlign: 'left',
                color: 'var(--text-muted)', fontSize: '10px',
                letterSpacing: '0.12em', textTransform: 'uppercase', fontWeight: 500,
                background: 'rgba(28,32,64,0.3)',
              }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody style={{ fontSize: '13px' }}>
          {loading && Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)}

          {!loading && error && (
            <tr><td colSpan={COL_SPAN} style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--red)', fontSize: '12px' }}>{error}</td></tr>
          )}

          {!loading && !error && predictions.length === 0 && (
            <tr><td colSpan={COL_SPAN} style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>No value bets found this gameweek.</td></tr>
          )}

          {!loading && !error && predictions.map(p => (
            <ValueBetRow
              key={p.match_id}
              p={p}
              enrichment={enrichmentFor(p)}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}
