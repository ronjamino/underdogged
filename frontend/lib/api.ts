const API_BASE = process.env.NEXT_PUBLIC_API_URL

export interface League {
  id: string
  name: string
  country: string
  match_count: number
  last_updated: string
}

export interface Prediction {
  match_id: string
  home_team: string
  away_team: string
  league: string
  match_date: string
  prob_home: number
  prob_draw: number
  prob_away: number
  predicted_outcome: 'H' | 'D' | 'A'
  confidence: number
  odds_home: number | null
  odds_draw: number | null
  odds_away: number | null
  value_bet: string | null
}

export async function fetchLeagues(): Promise<League[]> {
  const res = await fetch(`${API_BASE}/leagues`, { next: { revalidate: 3600 } })
  if (!res.ok) throw new Error('Failed to fetch leagues')
  return res.json()
}

// League IDs from the API are codes: PL, ELC, BL1, SA, PD
export async function fetchPredictions(leagueId: string): Promise<Prediction[]> {
  const res = await fetch(`${API_BASE}/predictions?league=${leagueId}`, {
    next: { revalidate: 1800 },
  })
  if (!res.ok) throw new Error('Failed to fetch predictions')
  return res.json()
}

export async function fetchTopPicks(): Promise<Prediction[]> {
  const res = await fetch(`${API_BASE}/predictions/top`, {
    next: { revalidate: 1800 },
  })
  if (!res.ok) throw new Error('Failed to fetch top picks')
  return res.json()
}
