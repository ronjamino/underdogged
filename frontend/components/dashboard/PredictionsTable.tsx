'use client'

import { PredictionRow } from './PredictionRow'
import type { Prediction, EnrichmentItem } from '@/lib/api'

interface Props {
  predictions: Prediction[]
  loading: boolean
  error: string
  enrichmentMap: Map<string, EnrichmentItem>
}

function SkeletonRow() {
  return (
    <tr style={{ borderBottom: '1px solid var(--border)', height: '52px' }}>
      {[80, 60, 300, 140, 130].map((w, i) => (
        <td key={i} style={{ padding: '0 16px', width: `${w}px` }}>
          <div className="skeleton" style={{ height: '10px', width: '70%', borderRadius: '3px' }} />
        </td>
      ))}
    </tr>
  )
}

export function PredictionsTable({ predictions, loading, error, enrichmentMap }: Props) {
  return (
    <div style={{
      overflowX: 'auto',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      background: 'var(--bg-card)',
    }}>
      <table style={{
        width: '100%',
        borderCollapse: 'collapse',
        tableLayout: 'fixed',
      }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Date', 'Kickoff', 'Match', 'Prediction', 'Confidence'].map(col => (
              <th key={col} style={{
                padding: '12px 16px',
                textAlign: 'left',
                color: 'var(--text-muted)',
                fontSize: '10px',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                fontWeight: 500,
                background: 'rgba(28,32,64,0.3)',
              }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody style={{ fontSize: '13px' }}>
          {loading && Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)}

          {!loading && error && (
            <tr>
              <td colSpan={5} style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--red)', fontSize: '12px' }}>
                {error}
              </td>
            </tr>
          )}

          {!loading && !error && predictions.length === 0 && (
            <tr>
              <td colSpan={5} style={{ padding: '48px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
                No predictions available for this gameweek.
              </td>
            </tr>
          )}

          {!loading && !error && predictions.map(p => (
            <PredictionRow
              key={p.match_id}
              prediction={p}
              enrichment={enrichmentMap.get(`${p.home_team}|${p.away_team}`)}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}
