'use client'

import { PredictionRow } from './PredictionRow'
import type { Prediction } from '@/lib/api'

interface Props {
  predictions: Prediction[]
  loading: boolean
  error: string
}

function SkeletonRow() {
  return (
    <tr style={{ borderBottom: '1px solid var(--border)', height: '48px' }}>
      {[80, 60, 300, 140, 130].map((w, i) => (
        <td key={i} style={{ padding: '0 16px', width: `${w}px` }}>
          <div className="skeleton" style={{ height: '12px', width: '70%' }} />
        </td>
      ))}
    </tr>
  )
}

export function PredictionsTable({ predictions, loading, error }: Props) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{
        width: '100%',
        borderCollapse: 'collapse',
        tableLayout: 'fixed',
      }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Date', 'Kickoff', 'Match', 'Prediction', 'Confidence'].map(col => (
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
          {loading && Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)}

          {!loading && error && (
            <tr>
              <td colSpan={5} style={{
                padding: '48px 16px',
                textAlign: 'center',
                color: 'var(--red)',
                fontSize: '12px',
              }}>
                {error}
              </td>
            </tr>
          )}

          {!loading && !error && predictions.length === 0 && (
            <tr>
              <td colSpan={5} style={{
                padding: '48px 16px',
                textAlign: 'center',
                color: 'var(--text-muted)',
                fontSize: '12px',
              }}>
                No predictions available for this gameweek.
              </td>
            </tr>
          )}

          {!loading && !error && predictions.map(p => (
            <PredictionRow key={p.match_id} prediction={p} />
          ))}
        </tbody>
      </table>
    </div>
  )
}
