'use client'

import type { PerformanceSummary, LiveRecord } from '@/lib/api'

interface Props {
  data: PerformanceSummary | null
  live: LiveRecord | null
  loading: boolean
  error: string
}

const OUTCOME_LABEL: Record<string, string> = { H: 'Home Win', D: 'Draw', A: 'Away Win' }
const OUTCOME_COLOR: Record<string, string> = { H: 'var(--green)', D: 'var(--accent)', A: 'var(--red)' }
const OUTCOME_DIM: Record<string, string>   = { H: 'var(--green-dim)', D: 'var(--accent-dim)', A: 'var(--red-dim)' }

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{ border: '1px solid var(--border)', padding: '20px 24px', flex: 1, minWidth: '140px' }}>
      <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '8px' }}>
        {label}
      </div>
      <div style={{ color: 'var(--accent)', fontSize: '24px', fontWeight: 600, lineHeight: 1 }}>
        {value}
      </div>
      {sub && <div style={{ color: 'var(--text-muted)', fontSize: '10px', marginTop: '6px' }}>{sub}</div>}
    </div>
  )
}

function SkeletonCard() {
  return (
    <div style={{ border: '1px solid var(--border)', padding: '20px 24px', flex: 1, minWidth: '140px' }}>
      <div className="skeleton" style={{ height: '10px', width: '60%', marginBottom: '12px' }} />
      <div className="skeleton" style={{ height: '24px', width: '40%' }} />
    </div>
  )
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '16px' }}>
      {children}
    </div>
  )
}

function ConfusionMatrix({ confusion }: { confusion: Record<string, Record<string, number>> }) {
  const labels: ('H' | 'D' | 'A')[] = ['H', 'D', 'A']
  const maxVal = Math.max(1, ...labels.flatMap(p => labels.map(a => confusion[p]?.[a] ?? 0)))

  return (
    <div>
      {/* Column headers (Actual) */}
      <div style={{ display: 'grid', gridTemplateColumns: '80px repeat(3, 1fr)', gap: '4px', marginBottom: '4px' }}>
        <div />
        {labels.map(a => (
          <div key={a} style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', paddingBottom: '4px' }}>
            {OUTCOME_LABEL[a]}
          </div>
        ))}
      </div>

      {/* Rows (Predicted) */}
      {labels.map(pred => (
        <div key={pred} style={{ display: 'grid', gridTemplateColumns: '80px repeat(3, 1fr)', gap: '4px', marginBottom: '4px' }}>
          {/* Row label */}
          <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', justifyContent: 'flex-end', paddingRight: '12px' }}>
            {OUTCOME_LABEL[pred]}
          </div>
          {labels.map(actual => {
            const count = confusion[pred]?.[actual] ?? 0
            const isDiag = pred === actual
            const intensity = count / maxVal
            const bg = isDiag
              ? `rgba(76, 175, 110, ${0.08 + intensity * 0.35})`   // green for correct
              : count > 0 ? `rgba(224, 82, 82, ${0.05 + intensity * 0.2})` : 'transparent'
            const color = isDiag ? 'var(--green)' : count > 0 ? 'var(--red)' : 'var(--border)'
            const pct = (() => {
              const rowTotal = labels.reduce((s, a) => s + (confusion[pred]?.[a] ?? 0), 0)
              return rowTotal > 0 ? Math.round(count / rowTotal * 100) : 0
            })()

            return (
              <div key={actual} style={{
                background: bg,
                border: `1px solid ${isDiag ? 'rgba(76,175,110,0.3)' : count > 0 ? 'rgba(224,82,82,0.2)' : 'var(--border)'}`,
                padding: '14px 8px',
                textAlign: 'center',
                position: 'relative',
              }}>
                <div style={{ color: isDiag ? 'var(--green)' : count > 0 ? 'var(--text)' : 'var(--text-muted)', fontSize: '20px', fontWeight: 600, lineHeight: 1 }}>
                  {count}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '10px', marginTop: '4px' }}>
                  {pct > 0 ? `${pct}%` : '—'}
                </div>
              </div>
            )
          })}
        </div>
      ))}

      {/* Axis labels */}
      <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr', marginTop: '8px' }}>
        <div />
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          ← Actual Result
        </div>
      </div>
      <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', writingMode: 'vertical-rl', transform: 'rotate(180deg)', position: 'absolute', marginTop: '-120px' }}>
      </div>
    </div>
  )
}

function AccuracyBars({ confusion }: { confusion: Record<string, Record<string, number>> }) {
  const labels: ('H' | 'D' | 'A')[] = ['H', 'D', 'A']

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {labels.map(pred => {
        const total  = labels.reduce((s, a) => s + (confusion[pred]?.[a] ?? 0), 0)
        const correct = confusion[pred]?.[pred] ?? 0
        const pct = total > 0 ? correct / total : 0

        return (
          <div key={pred}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ color: 'var(--text)', fontSize: '11px' }}>{OUTCOME_LABEL[pred]}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px', fontVariantNumeric: 'tabular-nums' }}>
                {correct}/{total} · {total > 0 ? `${Math.round(pct * 100)}%` : '—'}
              </span>
            </div>
            <div style={{ height: '6px', background: 'var(--border)', position: 'relative' }}>
              <div style={{
                position: 'absolute', left: 0, top: 0, height: '100%',
                width: `${pct * 100}%`,
                background: OUTCOME_COLOR[pred],
                transition: 'width 0.6s ease',
              }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function PerformanceTab({ data, live, loading, error }: Props) {
  if (error) {
    return <div style={{ padding: '48px 0', textAlign: 'center', color: 'var(--red)', fontSize: '12px' }}>{error}</div>
  }

  const hasLive = live && live.total_predicted > 0

  return (
    <div>
      {/* Backtest headline stats */}
      <SectionLabel>Backtest Summary</SectionLabel>
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '32px' }}>
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : data ? (
          <>
            <StatCard label="Model Accuracy"  value={`${(data.avg_accuracy * 100).toFixed(1)}%`}  sub="avg across all test windows" />
            <StatCard label="Hit Rate"         value={`${(data.avg_hit_rate * 100).toFixed(1)}%`}  sub="confident predictions only (≥55%)" />
            <StatCard label="Simulated ROI"    value={`${data.overall_roi_pct > 0 ? '+' : ''}${data.overall_roi_pct.toFixed(1)}%`} sub="flat £1 stake, raw bookmaker odds" />
            <StatCard label="Bets Tested"      value={data.total_bets.toLocaleString()}            sub={`across ${data.total_matches_tested.toLocaleString()} matches`} />
          </>
        ) : null}
      </div>

      {/* Live results section */}
      <SectionLabel>Live Prediction Record</SectionLabel>
      {!hasLive ? (
        <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '32px', padding: '24px', border: '1px solid var(--border)', textAlign: 'center' }}>
          No resolved predictions yet — results will appear here after matches are played.
        </div>
      ) : (
        <>
          {/* Summary cards */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '28px' }}>
            <StatCard label="Live Accuracy" value={live.accuracy != null ? `${(live.accuracy * 100).toFixed(1)}%` : '—'} sub={`${live.correct} correct of ${live.total_predicted} resolved`} />
            <StatCard label="Correct"   value={String(live.correct)}   sub="predictions matched" />
            <StatCard label="Incorrect" value={String(live.incorrect)} sub="predictions missed" />
            <StatCard label="Pending"   value={String(live.pending)}   sub="matches not yet played" />
          </div>

          {/* Confusion matrix + accuracy bars side by side */}
          <div style={{ display: 'flex', gap: '40px', flexWrap: 'wrap', marginBottom: '12px' }}>
            <div style={{ flex: '1 1 360px' }}>
              <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '12px' }}>
                Predicted vs Actual
              </div>
              <ConfusionMatrix confusion={live.confusion} />
            </div>
            <div style={{ flex: '1 1 220px' }}>
              <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '12px' }}>
                Accuracy by Outcome
              </div>
              <AccuracyBars confusion={live.confusion} />
              <div style={{ marginTop: '20px', color: 'var(--text-muted)', fontSize: '10px', lineHeight: 1.7 }}>
                Diagonal cells (green) are correct predictions.
                Off-diagonal (red) shows where the model was wrong and what actually happened.
              </div>
            </div>
          </div>
        </>
      )}

      {/* Per-window backtest table */}
      <SectionLabel>Walk-Forward Backtest Windows</SectionLabel>
      <div style={{ overflowX: 'auto', marginBottom: '20px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Period', 'Matches', 'Accuracy', 'Draw Recall', 'Bets', 'Hit Rate', 'ROI'].map(col => (
                <th key={col} style={{ padding: '10px 16px', textAlign: 'left', color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: 500 }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody style={{ fontSize: '13px' }}>
            {loading && Array.from({ length: 5 }).map((_, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border)', height: '48px' }}>
                {[120, 80, 80, 90, 60, 80, 70].map((w, j) => (
                  <td key={j} style={{ padding: '0 16px', width: `${w}px` }}>
                    <div className="skeleton" style={{ height: '12px', width: '70%' }} />
                  </td>
                ))}
              </tr>
            ))}
            {!loading && data?.windows.map(w => {
              const roiColor = w.roi_pct == null ? 'var(--text-muted)' : w.roi_pct > 0 ? 'var(--green)' : 'var(--red)'
              return (
                <tr key={w.window} style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.1s' }}
                  onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)'}
                  onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = 'transparent'}>
                  <td style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '11px', whiteSpace: 'nowrap' }}>
                    {w.start_date.slice(0, 7)} → {w.end_date.slice(0, 7)}
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>{w.test_size.toLocaleString()}</td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>{(w.accuracy * 100).toFixed(1)}%</td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>{(w.draw_recall * 100).toFixed(1)}%</td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>{w.n_confident_bets}</td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>{w.hit_rate != null ? `${(w.hit_rate * 100).toFixed(1)}%` : '—'}</td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums', color: roiColor }}>
                    {w.roi_pct != null ? `${w.roi_pct > 0 ? '+' : ''}${w.roi_pct.toFixed(1)}%` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div style={{ color: 'var(--text-muted)', fontSize: '11px', lineHeight: 1.8 }}>
        Walk-forward backtest: each window trains the full ensemble on all preceding data,
        then predicts on unseen matches. Hit rate and ROI apply only to predictions at ≥55% confidence
        using raw B365 bookmaker odds.
      </div>
    </div>
  )
}
