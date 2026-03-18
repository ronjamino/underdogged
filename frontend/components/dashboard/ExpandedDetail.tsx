'use client'

import type { Prediction } from '@/lib/api'

interface Props {
  p: Prediction
  colSpan: number
}

// Generate 5 approximate W/D/L results from win rate and draw rate
function formCircles(winRate: number, drawRate: number): ('W' | 'D' | 'L')[] {
  const wins  = Math.round(winRate * 5)
  const draws = Math.min(Math.round(drawRate * 5), 5 - wins)
  const losses = Math.max(5 - wins - draws, 0)
  const results: ('W' | 'D' | 'L')[] = [
    ...Array(wins).fill('W'),
    ...Array(draws).fill('D'),
    ...Array(losses).fill('L'),
  ]
  // Sort so most recent (right) tends to be newest — wins first for positive form
  return results
}

const CIRCLE_COLOR: Record<'W' | 'D' | 'L', { bg: string; color: string }> = {
  W: { bg: 'var(--green-dim)',   color: 'var(--green)'  },
  D: { bg: 'var(--accent-dim)',  color: 'var(--accent)' },
  L: { bg: 'var(--red-dim)',     color: 'var(--red)'    },
}

function FormCircle({ result }: { result: 'W' | 'D' | 'L' }) {
  const { bg, color } = CIRCLE_COLOR[result]
  return (
    <div style={{
      width: '26px',
      height: '26px',
      borderRadius: '50%',
      background: bg,
      border: `1px solid ${color}`,
      color,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '10px',
      fontWeight: 600,
      letterSpacing: 0,
      flexShrink: 0,
    }}>
      {result}
    </div>
  )
}

function FormRow({ label, winRate, drawRate, momentum }: {
  label: string
  winRate: number
  drawRate: number
  momentum: number
}) {
  const circles = formCircles(winRate, drawRate)
  const momentumPct = Math.round(momentum * 100)
  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px' }}>
        {label}
      </div>
      <div style={{ display: 'flex', gap: '6px', marginBottom: '6px' }}>
        {circles.map((r, i) => <FormCircle key={i} result={r} />)}
      </div>
      <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
        {Math.round(winRate * 100)}% win rate · {momentumPct}% momentum
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

export function ExpandedDetail({ p, colSpan }: Props) {
  const hasForm = p.home_form_winrate != null && p.away_form_winrate != null
  const hasH2H  = p.h2h_home_winrate != null
  const hasGoals = p.home_avg_goals_scored != null

  return (
    <tr>
      <td colSpan={colSpan} style={{ padding: 0, background: 'var(--bg-card)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ padding: '20px 24px', display: 'flex', gap: '32px', flexWrap: 'wrap' }}>

          {/* Form */}
          {hasForm && (
            <Section title="Form (last 5)">
              <FormRow
                label={p.home_team}
                winRate={p.home_form_winrate!}
                drawRate={p.home_venue_draw_rate ?? 0.25}
                momentum={p.home_momentum ?? 0.5}
              />
              <FormRow
                label={p.away_team}
                winRate={p.away_form_winrate!}
                drawRate={p.away_venue_draw_rate ?? 0.25}
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
              <div style={{ marginTop: '12px', display: 'flex', gap: '6px' }}>
                {((): ('W' | 'D' | 'L')[] => {
                  const w = Math.round(p.h2h_home_winrate! * 5)
                  const d = Math.min(Math.round(p.h2h_draw_rate! * 5), 5 - w)
                  const l = Math.max(5 - w - d, 0)
                  return [...Array(w).fill('W'), ...Array(d).fill('D'), ...Array(l).fill('L')]
                })().map((r, i) => <FormCircle key={i} result={r} />)}
              </div>
            </Section>
          )}

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
      </td>
    </tr>
  )
}
