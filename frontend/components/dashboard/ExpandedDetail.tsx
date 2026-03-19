'use client'

import type { Prediction, EnrichmentItem } from '@/lib/api'

interface Props {
  p: Prediction
  colSpan: number
  enrichment?: EnrichmentItem
  variant?: 'table' | 'card'
}

type Result = 'W' | 'D' | 'L'

/** Parse "W,D,L,W,W" → ['W','D','L','W','W']. Returns null if no real data. */
function parseForm(str: string | null | undefined): Result[] | null {
  if (!str || str.trim() === '') return null
  const parts = str.split(',').map(s => s.trim()).filter(Boolean)
  if (parts.length === 0) return null
  return parts.filter((r): r is Result => r === 'W' || r === 'D' || r === 'L')
}

const CIRCLE: Record<Result, { bg: string; border: string; color: string }> = {
  W: { bg: 'var(--green-dim)',  border: 'rgba(16,217,122,0.3)',  color: 'var(--green)'  },
  D: { bg: 'var(--accent-dim)', border: 'rgba(245,166,35,0.3)',  color: 'var(--accent)' },
  L: { bg: 'var(--red-dim)',    border: 'rgba(242,85,85,0.3)',   color: 'var(--red)'    },
}

function FormCircle({ result }: { result: Result }) {
  const { bg, border, color } = CIRCLE[result]
  return (
    <div style={{
      width: '26px',
      height: '26px',
      borderRadius: '50%',
      background: bg,
      border: `1px solid ${border}`,
      color,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '10px',
      fontWeight: 700,
      flexShrink: 0,
    }}>
      {result}
    </div>
  )
}

function FormRow({ label, formStr, winRate, momentum }: {
  label: string
  formStr: string | null | undefined
  winRate: number
  momentum: number
}) {
  const circles = parseForm(formStr)

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px' }}>
        {label}
      </div>
      {circles ? (
        <div style={{ display: 'flex', gap: '6px', marginBottom: '6px' }}>
          {circles.map((r, i) => <FormCircle key={i} result={r} />)}
        </div>
      ) : (
        <div style={{ color: 'var(--text-muted)', fontSize: '11px', fontStyle: 'italic', marginBottom: '6px' }}>
          No sequence data
        </div>
      )}
      <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
        {Math.round(winRate * 100)}% win rate · {Math.round(momentum * 100)}% momentum
      </div>
    </div>
  )
}

function StatLine({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
      <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{label}</span>
      <span style={{ color: 'var(--text)', fontSize: '11px', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ flex: 1, minWidth: '180px' }}>
      <div style={{
        color: 'var(--text-muted)',
        fontSize: '10px',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        marginBottom: '16px',
        paddingBottom: '8px',
        borderBottom: '1px solid var(--border)',
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

const VERDICT_STYLE: Record<string, { color: string; bg: string; border: string; label: string }> = {
  BACK:    { color: 'var(--green)',  bg: 'var(--green-dim)',  border: 'rgba(16,217,122,0.2)',  label: '↑ BACK'    },
  MONITOR: { color: 'var(--accent)', bg: 'var(--accent-dim)', border: 'rgba(245,166,35,0.2)',  label: '◎ MONITOR' },
  SKIP:    { color: 'var(--red)',    bg: 'var(--red-dim)',    border: 'rgba(242,85,85,0.2)',   label: '↓ SKIP'    },
}

export function ExpandedDetail({ p, colSpan, enrichment, variant = 'table' }: Props) {
  const hasForm  = p.home_form_winrate != null && p.away_form_winrate != null
  const hasH2H   = p.h2h_home_winrate != null
  const hasGoals = p.home_avg_goals_scored != null

  const inner = (
    <div style={{ padding: '16px 20px', display: 'flex', gap: '24px', flexWrap: 'wrap' }}>

          {/* Form */}
          {hasForm && (
            <Section title="Form (last 5 at venue)">
              <FormRow
                label={p.home_team}
                formStr={p.home_form}
                winRate={p.home_form_winrate!}
                momentum={p.home_momentum ?? 0.5}
              />
              <FormRow
                label={p.away_team}
                formStr={p.away_form}
                winRate={p.away_form_winrate!}
                momentum={p.away_momentum ?? 0.5}
              />
            </Section>
          )}

          {/* H2H */}
          {hasH2H && (
            <Section title="Head to Head">
              <StatLine label="Home win rate"  value={`${Math.round(p.h2h_home_winrate! * 100)}%`} />
              <StatLine label="Draw rate"      value={`${Math.round(p.h2h_draw_rate! * 100)}%`} />
              <StatLine label="Away win rate"  value={`${Math.round((1 - p.h2h_home_winrate! - p.h2h_draw_rate!) * 100)}%`} />
              <StatLine label="Avg goals/game" value={p.h2h_total_goals!.toFixed(1)} />
              {(() => {
                const circles = parseForm(p.h2h_form)
                return circles ? (
                  <div style={{ marginTop: '12px', display: 'flex', gap: '6px' }}>
                    {circles.map((r, i) => <FormCircle key={i} result={r} />)}
                  </div>
                ) : null
              })()}
            </Section>
          )}

          {/* AI Brief */}
          {enrichment && (() => {
            const s = VERDICT_STYLE[enrichment.verdict] ?? VERDICT_STYLE['MONITOR']
            return (
              <Section title="AI Brief">
                <div style={{ marginBottom: '10px' }}>
                  <span style={{
                    display: 'inline-block',
                    padding: '3px 10px',
                    background: s.bg,
                    border: `1px solid ${s.border}`,
                    color: s.color,
                    fontSize: '10px',
                    letterSpacing: '0.1em',
                    fontWeight: 700,
                    borderRadius: '3px',
                    textTransform: 'uppercase',
                  }}>
                    💡 {s.label}
                  </span>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '11px', lineHeight: 1.7, margin: 0 }}>
                  {enrichment.commentary}
                </p>
              </Section>
            )
          })()}

          {/* Goals */}
          {hasGoals && (
            <Section title="Goals">
              <div style={{ marginBottom: '12px' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px' }}>
                  {p.home_team}
                </div>
                <StatLine label="Scored/game"   value={p.home_avg_goals_scored!.toFixed(2)} />
                <StatLine label="Conceded/game" value={p.home_avg_goals_conceded!.toFixed(2)} />
              </div>
              <div style={{ marginBottom: '12px' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px' }}>
                  {p.away_team}
                </div>
                <StatLine label="Scored/game"   value={p.away_avg_goals_scored!.toFixed(2)} />
                <StatLine label="Conceded/game" value={p.away_avg_goals_conceded!.toFixed(2)} />
              </div>
              {p.expected_total_goals != null && (
                <div style={{ paddingTop: '8px', borderTop: '1px solid var(--border)' }}>
                  <StatLine label="Expected total" value={p.expected_total_goals.toFixed(1)} />
                </div>
              )}
            </Section>
          )}

    </div>
  )

  if (variant === 'card') {
    return (
      <div style={{ background: 'var(--bg-card)', borderTop: '1px solid var(--border)', margin: '12px -16px -14px' }}>
        {inner}
      </div>
    )
  }

  return (
    <tr>
      <td colSpan={colSpan} style={{ padding: 0, background: 'var(--bg-card)', borderBottom: '1px solid var(--border)' }}>
        {inner}
      </td>
    </tr>
  )
}
