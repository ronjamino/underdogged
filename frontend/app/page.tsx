import Link from 'next/link'
import { createClient } from '@/lib/supabase/server'

export default async function LandingPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  const isLoggedIn = !!user
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)', fontFamily: 'JetBrains Mono, monospace' }}>

      {/* Nav */}
      <nav className="landing-nav" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 32px', height: '56px',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(7,8,15,0.8)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <span style={{
          fontFamily: 'DM Serif Display, serif', fontStyle: 'italic', fontSize: '22px',
          background: 'linear-gradient(135deg, #F5A623 0%, #FFD07A 60%, #F5A623 100%)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
        }}>
          underdogged
        </span>
        <Link href={isLoggedIn ? '/dashboard' : '/login'} style={{
          padding: '7px 18px', fontSize: '11px', letterSpacing: '0.08em',
          textTransform: 'uppercase', textDecoration: 'none',
          border: '1px solid var(--accent)', color: 'var(--accent)',
          borderRadius: '4px', background: 'rgba(245,166,35,0.06)',
        }}>
          {isLoggedIn ? 'Dashboard →' : 'Sign in →'}
        </Link>
      </nav>

      {/* Hero */}
      <section className="landing-hero" style={{
        maxWidth: '1100px', margin: '0 auto', padding: '80px 32px 60px',
        display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center',
      }}>
        {/* Eyebrow badge */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '8px',
          padding: '5px 14px', marginBottom: '32px',
          border: '1px solid rgba(245,166,35,0.25)',
          background: 'rgba(245,166,35,0.06)',
          borderRadius: '20px', fontSize: '10px', letterSpacing: '0.12em',
          textTransform: 'uppercase', color: 'var(--accent)',
        }}>
          <span style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: 'var(--green)', display: 'inline-block',
            boxShadow: '0 0 6px var(--green)',
          }} />
          Live predictions · 5 leagues · Daily updates
        </div>

        <h1 style={{
          fontFamily: 'DM Serif Display, serif', fontStyle: 'italic',
          fontSize: 'clamp(40px, 7vw, 76px)', fontWeight: 400, lineHeight: 1.05,
          margin: '0 0 24px',
          background: 'linear-gradient(160deg, #E8ECF4 0%, #8A93B8 100%)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
        }}>
          Football predictions<br />built on data, not gut.
        </h1>

        <p style={{
          fontSize: '13px', lineHeight: 1.9, color: 'var(--text-muted)',
          maxWidth: '520px', margin: '0 0 56px',
        }}>
          An ensemble ML model analyses every fixture across Europe&apos;s top five leagues —
          surfacing high-confidence picks, value bets, and AI-researched match briefs before kickoff.
        </p>

        {/* ── Pitch graphic ── */}
        <PitchGraphic />

        <Link href={isLoggedIn ? '/dashboard' : '/login'} style={{
          marginTop: '52px',
          display: 'inline-block', padding: '14px 40px',
          background: 'linear-gradient(135deg, #F5A623 0%, #E0920E 100%)',
          color: '#07080F', fontSize: '12px', letterSpacing: '0.1em',
          textTransform: 'uppercase', fontWeight: 700, textDecoration: 'none',
          borderRadius: '5px', boxShadow: '0 0 36px rgba(245,166,35,0.28)',
        }}>
          {isLoggedIn ? 'Go to dashboard →' : 'View predictions →'}
        </Link>
      </section>

      {/* Divider */}
      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 32px' }}>
        <div style={{ borderTop: '1px solid var(--border)' }} />
      </div>

      {/* Feature cards */}
      <section className="landing-section" style={{ maxWidth: '1100px', margin: '0 auto', padding: '72px 32px' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
          gap: '20px',
        }}>
          {FEATURES.map(f => (
            <div key={f.title} style={{
              padding: '28px', border: '1px solid var(--border)',
              borderRadius: '8px', background: 'var(--bg-card)',
            }}>
              <div style={{ fontSize: '22px', marginBottom: '14px' }}>{f.icon}</div>
              <div style={{
                fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase',
                color: 'var(--accent)', fontWeight: 600, marginBottom: '10px',
              }}>
                {f.title}
              </div>
              <p style={{ fontSize: '12px', lineHeight: 1.8, color: 'var(--text-muted)', margin: 0 }}>
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Leagues strip */}
      <div style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}>
        <div style={{
          maxWidth: '1100px', margin: '0 auto', padding: '20px 32px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: '10px', flexWrap: 'wrap',
        }}>
          <span style={{ color: 'var(--text-muted)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', marginRight: '4px' }}>
            Covering
          </span>
          {LEAGUES.map(l => (
            <span key={l} style={{
              padding: '5px 14px', border: '1px solid var(--border)',
              borderRadius: '3px', fontSize: '10px', letterSpacing: '0.06em',
              textTransform: 'uppercase', color: 'var(--text-muted)',
            }}>
              {l}
            </span>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="landing-footer" style={{
        maxWidth: '1100px', margin: '0 auto', padding: '28px 32px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px',
      }}>
        <span style={{ fontFamily: 'DM Serif Display, serif', fontStyle: 'italic', fontSize: '16px', color: 'var(--text-muted)' }}>
          underdogged
        </span>
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
          For informational purposes only. Please bet responsibly.
        </span>
      </footer>

    </div>
  )
}

// ─── Pitch graphic ─────────────────────────────────────────────────────────────

function PitchGraphic() {
  const dim   = 'rgba(70,82,140,0.45)'
  const mid   = 'rgba(100,115,175,0.65)'
  const bright = 'rgba(130,148,210,0.85)'

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: '700px' }}>
      {/* Background glow */}
      <div style={{
        position: 'absolute', inset: '-60px',
        background: 'radial-gradient(ellipse 70% 55% at 50% 50%, rgba(245,166,35,0.07) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      <svg viewBox="0 0 700 440" style={{ width: '100%', display: 'block' }}>
        {/* Pitch surface */}
        <rect x="20" y="20" width="660" height="400" rx="2"
          fill="rgba(9,11,22,0.65)" stroke={bright} strokeWidth="1.5" />

        {/* Alternating stripes */}
        {[0,1,2,3,4,5,6].map(i => (
          <rect key={i} x="20" y={20 + i * 57} width="660" height="57"
            fill={i % 2 === 0 ? 'rgba(14,17,36,0.5)' : 'transparent'} />
        ))}

        {/* Halfway line */}
        <line x1="350" y1="20" x2="350" y2="420" stroke={dim} strokeWidth="1" />

        {/* Centre circle */}
        <circle cx="350" cy="220" r="68" fill="none" stroke={dim} strokeWidth="1" />
        <circle cx="350" cy="220" r="3.5" fill={mid} />

        {/* ── Left end ── */}
        {/* Penalty area */}
        <rect x="20" y="110" width="114" height="220" fill="none" stroke={dim} strokeWidth="1" />
        {/* 6-yard box */}
        <rect x="20" y="160" width="38" height="120" fill="none" stroke={dim} strokeWidth="1" />
        {/* Goal */}
        <rect x="6" y="192" width="14" height="56" fill="none" stroke={mid} strokeWidth="1.5" />
        {/* Penalty spot */}
        <circle cx="88" cy="220" r="2.5" fill={dim} />
        {/* Penalty arc */}
        <path d="M 134 172 A 68 68 0 0 1 134 268" fill="none" stroke={dim} strokeWidth="1" />

        {/* ── Right end ── */}
        <rect x="566" y="110" width="114" height="220" fill="none" stroke={dim} strokeWidth="1" />
        <rect x="642" y="160" width="38" height="120" fill="none" stroke={dim} strokeWidth="1" />
        <rect x="680" y="192" width="14" height="56" fill="none" stroke={mid} strokeWidth="1.5" />
        <circle cx="612" cy="220" r="2.5" fill={dim} />
        <path d="M 566 172 A 68 68 0 0 0 566 268" fill="none" stroke={dim} strokeWidth="1" />

        {/* Corner arcs */}
        <path d="M 20 38 A 18 18 0 0 1 38 20" fill="none" stroke={dim} strokeWidth="1" />
        <path d="M 662 20 A 18 18 0 0 1 680 38" fill="none" stroke={dim} strokeWidth="1" />
        <path d="M 20 402 A 18 18 0 0 0 38 420" fill="none" stroke={dim} strokeWidth="1" />
        <path d="M 680 402 A 18 18 0 0 1 662 420" fill="none" stroke={dim} strokeWidth="1" />

        {/* ──────────────────────────────────────────────
            Central prediction card
        ────────────────────────────────────────────── */}
        <g>
          {/* Card */}
          <rect x="198" y="152" width="304" height="136" rx="7"
            fill="rgba(6,7,15,0.94)" stroke="rgba(245,166,35,0.4)" strokeWidth="1" />
          {/* Gold top accent */}
          <rect x="198" y="152" width="304" height="3" rx="2" fill="#F5A623" />

          {/* Match */}
          <text x="350" y="180" textAnchor="middle" fill="#E8ECF4"
            fontSize="13" fontFamily="JetBrains Mono" fontWeight="600" letterSpacing="0.5">
            Arsenal  vs  Man City
          </text>
          <text x="350" y="196" textAnchor="middle" fill="rgba(130,145,195,0.6)"
            fontSize="8.5" fontFamily="JetBrains Mono" letterSpacing="1.5">
            PREMIER LEAGUE · SAT 22 MAR
          </text>

          {/* Divider */}
          <line x1="214" y1="206" x2="486" y2="206" stroke="rgba(28,32,64,1)" strokeWidth="1" />

          {/* HOME */}
          <text x="214" y="222" fill="rgba(140,155,200,0.65)" fontSize="8" fontFamily="JetBrains Mono" letterSpacing="1">HOME WIN</text>
          <rect x="214" y="226" width="178" height="5" rx="2.5" fill="rgba(28,32,64,0.9)" />
          <rect x="214" y="226" width="126" height="5" rx="2.5" fill="#10D97A" />
          <text x="398" y="232" fill="#10D97A" fontSize="9" fontFamily="JetBrains Mono">71%</text>

          {/* DRAW */}
          <text x="214" y="248" fill="rgba(140,155,200,0.65)" fontSize="8" fontFamily="JetBrains Mono" letterSpacing="1">DRAW</text>
          <rect x="214" y="252" width="178" height="5" rx="2.5" fill="rgba(28,32,64,0.9)" />
          <rect x="214" y="252" width="36" height="5" rx="2.5" fill="#F5A623" />
          <text x="398" y="258" fill="rgba(140,155,200,0.65)" fontSize="9" fontFamily="JetBrains Mono">20%</text>

          {/* AWAY */}
          <text x="214" y="274" fill="rgba(140,155,200,0.65)" fontSize="8" fontFamily="JetBrains Mono" letterSpacing="1">AWAY WIN</text>
          <rect x="214" y="278" width="178" height="5" rx="2.5" fill="rgba(28,32,64,0.9)" />
          <rect x="214" y="278" width="16" height="5" rx="2.5" fill="#F25555" />
          <text x="398" y="284" fill="rgba(140,155,200,0.65)" fontSize="9" fontFamily="JetBrains Mono">9%</text>

          {/* Verdict badge */}
          <rect x="424" y="222" width="68" height="66" rx="5"
            fill="rgba(16,217,122,0.08)" stroke="rgba(16,217,122,0.28)" strokeWidth="1" />
          <text x="458" y="248" textAnchor="middle" fill="#10D97A"
            fontSize="17" fontFamily="JetBrains Mono" fontWeight="700">↑</text>
          <text x="458" y="263" textAnchor="middle" fill="#10D97A"
            fontSize="8.5" fontFamily="JetBrains Mono" fontWeight="700" letterSpacing="1">BACK</text>
          <text x="458" y="279" textAnchor="middle" fill="rgba(16,217,122,0.5)"
            fontSize="8" fontFamily="JetBrains Mono">71% conf.</text>
        </g>

        {/* ── Floating stat chips ── */}

        {/* Top-left: Form */}
        <g>
          <rect x="38" y="38" width="120" height="44" rx="5"
            fill="rgba(6,7,15,0.88)" stroke="rgba(40,50,100,0.9)" strokeWidth="1" />
          <text x="48" y="54" fill="rgba(140,155,200,0.6)" fontSize="7.5" fontFamily="JetBrains Mono" letterSpacing="1">FORM (HOME)</text>
          {/* W/D/L circles */}
          {['W','W','D','W','W'].map((r, i) => (
            <g key={i}>
              <circle cx={48 + i * 20} cy="70" r="8"
                fill={r === 'W' ? 'rgba(16,217,122,0.15)' : r === 'D' ? 'rgba(245,166,35,0.15)' : 'rgba(242,85,85,0.15)'}
                stroke={r === 'W' ? 'rgba(16,217,122,0.5)' : r === 'D' ? 'rgba(245,166,35,0.5)' : 'rgba(242,85,85,0.5)'}
                strokeWidth="1"
              />
              <text x={48 + i * 20} y="74" textAnchor="middle"
                fill={r === 'W' ? '#10D97A' : r === 'D' ? '#F5A623' : '#F25555'}
                fontSize="7.5" fontFamily="JetBrains Mono" fontWeight="700">{r}</text>
            </g>
          ))}
        </g>

        {/* Top-right: Form */}
        <g>
          <rect x="542" y="38" width="120" height="44" rx="5"
            fill="rgba(6,7,15,0.88)" stroke="rgba(40,50,100,0.9)" strokeWidth="1" />
          <text x="552" y="54" fill="rgba(140,155,200,0.6)" fontSize="7.5" fontFamily="JetBrains Mono" letterSpacing="1">FORM (AWAY)</text>
          {['W','D','L','W','W'].map((r, i) => (
            <g key={i}>
              <circle cx={552 + i * 20} cy="70" r="8"
                fill={r === 'W' ? 'rgba(16,217,122,0.15)' : r === 'D' ? 'rgba(245,166,35,0.15)' : 'rgba(242,85,85,0.15)'}
                stroke={r === 'W' ? 'rgba(16,217,122,0.5)' : r === 'D' ? 'rgba(245,166,35,0.5)' : 'rgba(242,85,85,0.5)'}
                strokeWidth="1"
              />
              <text x={552 + i * 20} y="74" textAnchor="middle"
                fill={r === 'W' ? '#10D97A' : r === 'D' ? '#F5A623' : '#F25555'}
                fontSize="7.5" fontFamily="JetBrains Mono" fontWeight="700">{r}</text>
            </g>
          ))}
        </g>

        {/* Bottom-left: Goals */}
        <g>
          <rect x="38" y="358" width="120" height="44" rx="5"
            fill="rgba(6,7,15,0.88)" stroke="rgba(40,50,100,0.9)" strokeWidth="1" />
          <text x="48" y="374" fill="rgba(140,155,200,0.6)" fontSize="7.5" fontFamily="JetBrains Mono" letterSpacing="1">xG / GAME</text>
          <text x="48" y="393" fill="#F5A623" fontSize="15" fontFamily="JetBrains Mono" fontWeight="600">2.8</text>
          <text x="82" y="393" fill="rgba(140,155,200,0.5)" fontSize="9" fontFamily="JetBrains Mono"> goals avg</text>
        </g>

        {/* Bottom-right: H2H */}
        <g>
          <rect x="542" y="358" width="120" height="44" rx="5"
            fill="rgba(6,7,15,0.88)" stroke="rgba(40,50,100,0.9)" strokeWidth="1" />
          <text x="552" y="374" fill="rgba(140,155,200,0.6)" fontSize="7.5" fontFamily="JetBrains Mono" letterSpacing="1">H2H WIN %</text>
          <text x="552" y="393" fill="#E8ECF4" fontSize="15" fontFamily="JetBrains Mono" fontWeight="600">58%</text>
          <text x="584" y="393" fill="rgba(140,155,200,0.5)" fontSize="9" fontFamily="JetBrains Mono"> home</text>
        </g>

        {/* Bottom-centre: AI brief chip */}
        <g>
          <rect x="270" y="306" width="160" height="26" rx="4"
            fill="rgba(16,217,122,0.06)" stroke="rgba(16,217,122,0.22)" strokeWidth="1" />
          <text x="350" y="323" textAnchor="middle"
            fill="rgba(16,217,122,0.75)" fontSize="9" fontFamily="JetBrains Mono" letterSpacing="0.5">
            💡 AI: Clean bill of health
          </text>
        </g>

      </svg>
    </div>
  )
}

// ─── Static data ──────────────────────────────────────────────────────────────

const FEATURES = [
  {
    icon: '⚡',
    title: 'Ensemble ML Model',
    body: 'Random Forest, XGBoost, and MLP combined in a stacking ensemble — walk-forward backtested across thousands of historical fixtures.',
  },
  {
    icon: '🎯',
    title: 'Value Bet Detection',
    body: 'Identifies positive expected-value opportunities where our model probability exceeds the bookmaker\'s implied probability by more than 5%.',
  },
  {
    icon: '💡',
    title: 'AI Match Briefs',
    body: 'Before each matchday, Claude web-searches injury news, lineup concerns, and motivation factors — then issues a BACK, MONITOR, or SKIP verdict.',
  },
  {
    icon: '📊',
    title: 'Full Transparency',
    body: 'Every prediction shows confidence, real form circles, H2H record, and expected goals. You see exactly what the model sees.',
  },
]

const LEAGUES = ['Premier League', 'Bundesliga', 'Serie A', 'La Liga', 'Championship']
