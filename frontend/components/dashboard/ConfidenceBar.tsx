'use client'

interface Props {
  value: number  // 0–1
}

export function ConfidenceBar({ value }: Props) {
  const pct = Math.round(value * 100)

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      {/* Track */}
      <div style={{
        flex: 1,
        height: '4px',
        background: 'var(--border)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Fill */}
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          height: '100%',
          width: `${pct}%`,
          background: 'var(--accent)',
          animation: 'bar-grow 0.6s ease forwards',
        }} />
      </div>
      {/* Label */}
      <span style={{
        color: 'var(--text-muted)',
        fontSize: '11px',
        minWidth: '34px',
        textAlign: 'right',
      }}>
        {pct}%
      </span>
    </div>
  )
}
