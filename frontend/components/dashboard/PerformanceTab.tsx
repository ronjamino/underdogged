'use client'

import type { PerformanceSummary } from '@/lib/api'

interface Props {
  data: PerformanceSummary | null
  loading: boolean
  error: string
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{
      border: '1px solid var(--border)',
      padding: '20px 24px',
      flex: 1,
      minWidth: '140px',
    }}>
      <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '8px' }}>
        {label}
      </div>
      <div style={{ color: 'var(--accent)', fontSize: '24px', fontWeight: 600, lineHeight: 1 }}>
        {value}
      </div>
      {sub && (
        <div style={{ color: 'var(--text-muted)', fontSize: '10px', marginTop: '6px' }}>
          {sub}
        </div>
      )}
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

export function PerformanceTab({ data, loading, error }: Props) {
  if (error) {
    return (
      <div style={{ padding: '48px 0', textAlign: 'center', color: 'var(--red)', fontSize: '12px' }}>
        {error}
      </div>
    )
  }

  return (
    <div>
      {/* Headline stats */}
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '32px' }}>
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : data ? (
          <>
            <StatCard
              label="Model Accuracy"
              value={`${(data.avg_accuracy * 100).toFixed(1)}%`}
              sub="avg across all test windows"
            />
            <StatCard
              label="Hit Rate"
              value={`${(data.avg_hit_rate * 100).toFixed(1)}%`}
              sub="confident predictions only (≥55%)"
            />
            <StatCard
              label="Simulated ROI"
              value={`${data.overall_roi_pct > 0 ? '+' : ''}${data.overall_roi_pct.toFixed(1)}%`}
              sub="flat £1 stake, raw bookmaker odds"
            />
            <StatCard
              label="Bets Tested"
              value={data.total_bets.toLocaleString()}
              sub={`across ${data.total_matches_tested.toLocaleString()} matches`}
            />
          </>
        ) : null}
      </div>

      {/* Per-window table */}
      <div style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
        Walk-Forward Backtest Windows
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Period', 'Matches', 'Accuracy', 'Draw Recall', 'Bets', 'Hit Rate', 'ROI'].map(col => (
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
              const roiColor = w.roi_pct == null ? 'var(--text-muted)'
                : w.roi_pct > 0 ? 'var(--green)' : 'var(--red)'
              return (
                <tr
                  key={w.window}
                  style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.1s' }}
                  onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)'}
                  onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = 'transparent'}
                >
                  <td style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '11px', whiteSpace: 'nowrap' }}>
                    {w.start_date.slice(0, 7)} → {w.end_date.slice(0, 7)}
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>
                    {w.test_size.toLocaleString()}
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>
                    {(w.accuracy * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>
                    {(w.draw_recall * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>
                    {w.n_confident_bets}
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums' }}>
                    {w.hit_rate != null ? `${(w.hit_rate * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ padding: '12px 16px', fontVariantNumeric: 'tabular-nums', color: roiColor }}>
                    {w.roi_pct != null ? `${w.roi_pct > 0 ? '+' : ''}${w.roi_pct.toFixed(1)}%` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: '20px', color: 'var(--text-muted)', fontSize: '11px', lineHeight: 1.8 }}>
        Walk-forward backtest: each window trains the full ensemble on all preceding data,
        then predicts on unseen matches. Hit rate and ROI apply only to predictions at ≥55% confidence
        using raw B365 bookmaker odds.
      </div>
    </div>
  )
}
