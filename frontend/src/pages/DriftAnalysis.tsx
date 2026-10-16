import { useState, useEffect } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { RefreshCw, Upload } from 'lucide-react'
import type { DriftScore, Model } from '@/types'

export default function DriftAnalysis({ onOpenRegister }: { onOpenRegister?: () => void }) {
  const [models, setModels] = useState<Model[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string>('')
  const [driftScores, setDriftScores] = useState<DriftScore[]>([])
  const [timeWindow, setTimeWindow] = useState<string>('30d')
  const [isLoading, setIsLoading] = useState(false)
  const [isModelsLoading, setIsModelsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getModels().then(res => {
      setModels(res)
      if (res.length > 0) setSelectedModelId(res[0].id)
    }).catch(fetchError => {
      setError(fetchError instanceof Error ? fetchError.message : 'Unable to load models.')
    }).finally(() => setIsModelsLoading(false))
  }, [])

  useEffect(() => {
    if (selectedModelId) {
      setIsLoading(true)
      setError(null)
      api.getModelDrift(selectedModelId, timeWindow)
        .then(setDriftScores)
        .catch(fetchError => setError(fetchError instanceof Error ? fetchError.message : 'Unable to load drift scores.'))
        .finally(() => setIsLoading(false))
    }
  }, [selectedModelId, timeWindow])

  const selectedModel = models.find(m => m.id === selectedModelId)

  const handleRecheck = async () => {
    if (!selectedModelId) return
    setIsLoading(true)
    setError(null)
    try {
      await api.reevaluateModel(selectedModelId)
      setDriftScores(await api.getModelDrift(selectedModelId, timeWindow))
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Unable to recheck drift scores.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-6 gap-4">
        <div>
          <h1 className="text-[32px] font-semibold leading-[40px] tracking-[-0.02em] text-on-surface mb-2">
            Drift Analysis
          </h1>
          <p className="text-[16px] leading-[24px] text-on-surface-variant">
            Population Stability Index (PSI) and Jensen-Shannon distance metrics across model features.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={timeWindow}
            onChange={(e) => setTimeWindow(e.target.value)}
            className="h-10 px-3 rounded-lg border border-outline-variant bg-surface text-on-surface text-sm focus:outline-none focus:border-primary"
          >
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
          {onOpenRegister && (
            <Button onClick={onOpenRegister}>
              <Upload className="w-4 h-4 mr-2" /> Live Ingest Dataset
            </Button>
          )}
        </div>
      </div>

      {/* Model Selection */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 border-b border-outline-variant min-w-0">
        {models.map(m => (
          <button
            key={m.id}
            onClick={() => setSelectedModelId(m.id)}
            className={`px-4 py-2.5 rounded-lg text-sm font-semibold whitespace-nowrap transition-all ${
              selectedModelId === m.id
                ? 'bg-primary text-on-primary shadow-md'
                : 'bg-surface text-on-surface-variant hover:text-on-surface border border-outline-variant'
            }`}
          >
            {m.name} ({m.version})
          </button>
        ))}
      </div>

      {/* Overview stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-1">Target Model</div>
            <div className="text-title-lg font-bold text-on-surface">{selectedModel?.name || 'Loading...'}</div>
            <div className="text-xs text-on-surface-variant mt-1">{selectedModel?.algorithm} • {selectedModel?.environment}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-1">Feature Count</div>
            <div className="text-headline-lg font-bold text-on-surface">{driftScores.length} Features</div>
            <div className="text-xs text-on-surface-variant mt-1">Evaluated over {timeWindow}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-1">Highest Drift Feature</div>
            <div className="text-title-lg font-bold text-error">
              {driftScores[0]
                ? `${driftScores[0].featureName} (${driftScores[0].driftScore ?? 'n/a'})`
                : 'None'}
            </div>
            <div className="text-xs text-on-surface-variant mt-1">Status: {driftScores[0]?.status?.toUpperCase() || 'N/A'}</div>
          </CardContent>
        </Card>
      </div>

      {/* Feature Drift Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Feature Drift & Population Stability</CardTitle>
            <CardDescription>Baseline distribution comparison against production inferencing</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleRecheck} disabled={isLoading || !selectedModelId}>
            <RefreshCw className="h-4 w-4 mr-2" /> Recheck Drift
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Feature Name</TableHead>
                <TableHead>Baseline Mean</TableHead>
                <TableHead>Current Mean</TableHead>
                <TableHead>PSI Drift Score</TableHead>
                <TableHead>JS Distance</TableHead>
                <TableHead>Live Samples</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isModelsLoading ? (
                <TableRow><TableCell colSpan={7} className="text-center py-10 text-on-surface-variant animate-pulse">Loading models...</TableCell></TableRow>
              ) : error ? (
                <TableRow><TableCell colSpan={7} className="text-center py-10 text-error">Unable to load drift data: {error}</TableCell></TableRow>
              ) : isLoading ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-10 text-on-surface-variant animate-pulse">
                    Loading PSI and JS distance scores...
                  </TableCell>
                </TableRow>
              ) : driftScores.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-10 text-on-surface-variant">
                    Not enough data yet: upload a baseline and send live predictions before drift can be calculated.
                  </TableCell>
                </TableRow>
              ) : (
                driftScores.map((row) => (
                  <TableRow key={row.featureName}>
                    <TableCell className="font-semibold text-on-surface">{row.featureName}</TableCell>
                    <TableCell>{row.baselineMean != null ? Number(row.baselineMean).toLocaleString() : '—'}</TableCell>
                    <TableCell>{row.currentMean != null ? Number(row.currentMean).toLocaleString() : '—'}</TableCell>
                    <TableCell className="font-mono font-semibold">{row.driftScore != null ? Number(row.driftScore).toFixed(4) : '—'}</TableCell>
                    <TableCell className="font-mono text-on-surface-variant">{row.jsDistance != null ? Number(row.jsDistance).toFixed(4) : 'N/A'}</TableCell>
                    <TableCell>{row.nLive != null ? row.nLive.toLocaleString() : '—'}</TableCell>
                    <TableCell>
                      <Badge variant={row.status as any} className="uppercase leading-none py-1 text-[10px]">
                        {row.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
