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
  // Actual result
  actual_result: string | null   // 'H', 'D', 'A' or null
  home_score: number | null
  away_score: number | null
  // Form
  home_form_winrate: number | null
  away_form_winrate: number | null
  home_momentum: number | null
  away_momentum: number | null
  home_venue_draw_rate: number | null
  away_venue_draw_rate: number | null
  // H2H
  h2h_home_winrate: number | null
  h2h_draw_rate: number | null
  h2h_total_goals: number | null
  // Goals
  home_avg_goals_scored: number | null
  home_avg_goals_conceded: number | null
  away_avg_goals_scored: number | null
  away_avg_goals_conceded: number | null
  expected_total_goals: number | null
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

export interface WindowResult {
  window: number
  start_date: string
  end_date: string
  test_size: number
  accuracy: number
  draw_recall: number
  draw_precision: number
  n_confident_bets: number
  hit_rate: number | null
  roi_pct: number | null
}

export interface PerformanceSummary {
  avg_accuracy: number
  avg_hit_rate: number
  overall_roi_pct: number
  total_bets: number
  total_matches_tested: number
  windows: WindowResult[]
}

export interface LiveRecord {
  total_predicted: number
  correct: number
  incorrect: number
  pending: number
  accuracy: number | null
  // confusion[predicted][actual] = count
  confusion: Record<string, Record<string, number>>
}

export async function fetchLiveRecord(): Promise<LiveRecord> {
  const res = await fetch(`${API_BASE}/performance/live`, { next: { revalidate: 3600 } })
  if (!res.ok) throw new Error('Failed to fetch live record')
  return res.json()
}

export async function fetchPerformance(): Promise<PerformanceSummary> {
  const res = await fetch(`${API_BASE}/performance`, { next: { revalidate: 86400 } })
  if (!res.ok) throw new Error('Failed to fetch performance data')
  return res.json()
}

export async function fetchValueBets(): Promise<Prediction[]> {
  const res = await fetch(`${API_BASE}/predictions/value`, {
    next: { revalidate: 1800 },
  })
  if (!res.ok) throw new Error('Failed to fetch value bets')
  return res.json()
}
