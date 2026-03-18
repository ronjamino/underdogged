'use client'

interface Props {
  value: number  // 0–1
}

export function ConfidenceBar({ value }: Props) {
  const pct = Math.round(value * 100)
  const isHigh = pct >= 65
  const isMid  = pct >= 55

  const fillColor = isHigh
    ? 'linear-gradient(90deg, #10D97A, #4DFFA3)'
    : isMid
    ? 'linear-gradient(90deg, #F5A623, #FFD07A)'
    : 'linear-gradient(90deg, #4E567A, #6B7599)'

  const glowColor = isHigh
    ? 'rgba(16,217,122,0.35)'
    : isMid
    ? 'rgba(245,166,35,0.35)'
    : 'transparent'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      {/* Track */}
      <div style={{
        flex: 1,
        height: '3px',
        background: 'var(--border)',
        position: 'relative',
        overflow: 'visible',
        borderRadius: '2px',
      }}>
        {/* Fill */}
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          height: '100%',
          width: `${pct}%`,
          background: fillColor,
          borderRadius: '2px',
          boxShadow: `0 0 8px ${glowColor}`,
          animation: 'bar-grow 0.6s ease forwards',
        }} />
      </div>
      {/* Label */}
      <span style={{
        color: isHigh ? 'var(--green)' : isMid ? 'var(--accent)' : 'var(--text-muted)',
        fontSize: '11px',
        minWidth: '34px',
        textAlign: 'right',
        fontVariantNumeric: 'tabular-nums',
      }}>
        {pct}%
      </span>
    </div>
  )
}
