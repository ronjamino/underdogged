'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { LeagueSelector } from '@/components/dashboard/LeagueSelector'
import { PredictionsTable } from '@/components/dashboard/PredictionsTable'
import { fetchPredictions } from '@/lib/api'
import type { Prediction } from '@/lib/api'

const DEFAULT_LEAGUE = 'PL'

export default function DashboardPage() {
  const [activeLeague, setActiveLeague] = useState(DEFAULT_LEAGUE)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    setLoading(true)
    setError('')
    fetchPredictions(activeLeague)
      .then(setPredictions)
      .catch(() => setError('Failed to load predictions. Check API connection.'))
      .finally(() => setLoading(false))
  }, [activeLeague])

  return (
    <main style={{ padding: '24px 0' }}>
      <LeagueSelector active={activeLeague} onChange={setActiveLeague} />
      <div style={{ marginTop: '24px' }}>
        <PredictionsTable predictions={predictions} loading={loading} error={error} />
      </div>
    </main>
  )
}
