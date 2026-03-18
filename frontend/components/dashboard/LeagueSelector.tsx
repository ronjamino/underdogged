'use client'

const LEAGUES = [
  { id: 'PL',  name: 'Premier League', abbrev: 'EPL'  },
  { id: 'ELC', name: 'Championship',   abbrev: 'CHMP' },
  { id: 'BL1', name: 'Bundesliga',     abbrev: 'BUN'  },
  { id: 'SA',  name: 'Serie A',        abbrev: 'SA'   },
  { id: 'PD',  name: 'La Liga',        abbrev: 'LL'   },
]

interface Props {
  active: string
  onChange: (id: string) => void
}

export function LeagueSelector({ active, onChange }: Props) {
  return (
    <div style={{
      display: 'flex',
      borderBottom: '1px solid var(--border)',
      gap: '0',
    }}>
      {LEAGUES.map(league => {
        const isActive = active === league.id
        return (
          <button
            key={league.id}
            onClick={() => onChange(league.id)}
            style={{
              padding: '10px 18px',
              background: isActive ? 'rgba(245,166,35,0.06)' : 'none',
              border: 'none',
              borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
              color: isActive ? 'var(--accent)' : 'var(--text-muted)',
              fontSize: '10px',
              fontFamily: 'JetBrains Mono, monospace',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              cursor: 'pointer',
              transition: 'color 0.15s, background 0.15s',
              whiteSpace: 'nowrap',
              marginBottom: '-1px',
              textShadow: isActive ? '0 0 16px var(--accent-glow)' : 'none',
            }}
            onMouseEnter={e => {
              if (!isActive) {
                e.currentTarget.style.color = 'var(--text)'
                e.currentTarget.style.background = 'rgba(28,32,64,0.4)'
              }
            }}
            onMouseLeave={e => {
              if (!isActive) {
                e.currentTarget.style.color = 'var(--text-muted)'
                e.currentTarget.style.background = 'none'
              }
            }}
          >
            <span className="hidden sm:inline">{league.name}</span>
            <span className="sm:hidden">{league.abbrev}</span>
          </button>
        )
      })}
    </div>
  )
}
