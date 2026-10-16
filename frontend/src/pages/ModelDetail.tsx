import { useState, useEffect } from "react"
import { useParams } from "react-router-dom"
import { api } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { RefreshCw, ArrowDown, Minus, Loader2 } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { Model, DriftScore } from "@/types"

type DriftRow = Omit<DriftScore, 'status'> & {
  status: DriftScore['status']
  confidence?: string | null
}

export default function ModelDetail() {
  const { id } = useParams()
  const [model, setModel] = useState<Model | null>(null)
  const [drift, setDrift] = useState<DriftRow[]>([])
  const [performance, setPerformance] = useState<{
    metrics: Array<{ label: string; value: number | null; baseline: number | null; delta: number | null; reason?: string }>
    timeSeries: Array<{ time: string; current: number | null; base: number | null }>
    hasLabels: boolean
    labelCoverage: number | null
    taskType: string | null
  } | null>(null)
  const [isReevaluating, setIsReevaluating] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    if (!id) return
    setIsLoading(true)
    setError(null)
    try {
      const [modelResult, driftResult, performanceResult] = await Promise.all([
        api.getModel(id),
        api.getModelDrift(id),
        api.getModelPerformance(id),
      ])
      setModel(modelResult)
      setDrift(driftResult)
      setPerformance(performanceResult)
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Unable to load model metrics.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [id])

  const handleReevaluate = async () => {
    if (!id) return
    setIsReevaluating(true)
    try {
      await api.reevaluateModel(id)
      await fetchData()
    } catch (reevaluateError) {
      setError(reevaluateError instanceof Error ? reevaluateError.message : 'Unable to reevaluate this model.')
    } finally {
      setIsReevaluating(false)
    }
  }

  if (isLoading && !model) return <div className="p-10 animate-pulse text-on-surface-variant text-center">Loading model metrics...</div>
  if (error && !model) return <div className="p-10 text-error text-center">Unable to load model metrics: {error}</div>
  if (!model) return <div className="p-10 text-on-surface-variant text-center">Model data is unavailable.</div>

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-4 pb-4 border-b border-outline-variant">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-headline-lg font-bold text-on-surface">{model.name}</h1>
            <Badge variant="stable" className="uppercase tracking-wider">{model.status}</Badge>
          </div>
          <p className="text-body-sm text-on-surface-variant font-medium">
            Last evaluated: {model.lastUpdated}
          </p>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="relative h-14 w-14 flex items-center justify-center rounded-full border-[4px] border-stable-bg">
              <div className="absolute inset-0 rounded-full border-[4px] border-stable-text/80 transform -rotate-45" />
              <span className="font-bold text-lg">{model.healthScore}</span>
            </div>
          </div>
          <Button onClick={handleReevaluate} disabled={isReevaluating}>
            {isReevaluating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Evaluating...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                Re-evaluate
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Grid of Key Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Algorithm</div>
            <div className="text-title-lg font-semibold">{model.algorithm}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Environment</div>
            <div className="text-title-lg font-semibold capitalize">{model.environment}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Version</div>
            <div className="text-title-lg font-semibold">{model.version}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Predictions (7d)</div>
            <div className="text-title-lg font-semibold">{model.predictionsCount?.toLocaleString()}</div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="drift">
        <TabsList>
          <TabsTrigger value="drift">Feature Drift (PSI)</TabsTrigger>
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
        </TabsList>

        {/* Feature Drift Tab */}
        <TabsContent value="drift" className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Feature Drift Summary</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature Name</TableHead>
                    <TableHead>Baseline Mean</TableHead>
                    <TableHead>Current Mean</TableHead>
                    <TableHead>PSI Score</TableHead>
                    <TableHead>JS Distance</TableHead>
                    <TableHead>Live Samples</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {error ? (
                    <TableRow><TableCell colSpan={7} className="text-center py-8 text-error">Unable to load drift data: {error}</TableCell></TableRow>
                  ) : isLoading ? (
                    <TableRow><TableCell colSpan={7} className="text-center py-8 text-on-surface-variant animate-pulse">Loading drift scores...</TableCell></TableRow>
                  ) : drift.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center py-8 text-on-surface-variant">
                        Not enough data yet: upload a baseline and send live predictions before drift can be calculated.
                      </TableCell>
                    </TableRow>
                  ) : (
                    drift.map((row) => (
                      <TableRow key={row.featureName}>
                        <TableCell className="font-semibold text-on-surface">{row.featureName}</TableCell>
                        <TableCell>{row.baselineMean != null ? row.baselineMean.toLocaleString() : '—'}</TableCell>
                        <TableCell>{row.currentMean != null ? row.currentMean.toLocaleString() : '—'}</TableCell>
                        <TableCell className="font-semibold">{row.driftScore != null ? row.driftScore.toFixed(4) : '—'}</TableCell>
                        <TableCell className="text-on-surface-variant">
                          {row.jsDistance != null ? row.jsDistance.toFixed(4) : '—'}
                        </TableCell>
                        <TableCell>{row.nLive != null ? row.nLive.toLocaleString() : '—'}</TableCell>
                        <TableCell>
                          {row.status === 'stable' && <Badge variant="stable">Stable</Badge>}
                          {row.status === 'warning' && <Badge variant="warning">Warning</Badge>}
                          {row.status === 'critical' && <Badge variant="critical">Critical</Badge>}
                          {row.status === 'insufficient_data' && (
                            <Badge variant="outline" className="text-on-surface-variant border-outline-variant">Insufficient Data</Badge>
                          )}
                          {row.reason && row.status === 'insufficient_data' && (
                            <div className="text-[10px] text-on-surface-variant mt-0.5 max-w-[200px] truncate" title={row.reason}>{row.reason}</div>
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          {!performance?.hasLabels && (
            <div className="p-4 rounded-lg border border-outline-variant bg-surface text-sm text-on-surface-variant">
              ⏳ Awaiting ground-truth labels — send labels via <code className="text-xs bg-surface-low px-1 rounded">POST /api/models/{'{id}'}/ground-truth</code> to see performance metrics.
              {performance?.labelCoverage != null && (
                <span className="ml-2 font-semibold">Label coverage: {(performance.labelCoverage * 100).toFixed(1)}%</span>
              )}
            </div>
          )}
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {performance?.metrics.map((m) => (
              <Card key={m.label}>
                <CardContent className="p-6">
                  <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">{m.label}</div>
                  {m.value == null ? (
                    <div className="text-sm text-on-surface-variant italic">{m.reason ?? 'awaiting_labels'}</div>
                  ) : (
                    <>
                      <div className="flex items-end justify-between">
                        <div className="text-headline-lg font-semibold">{m.value}%</div>
                        <div className="text-xs text-on-surface-variant mb-1">Base: {m.baseline ?? '—'}%</div>
                      </div>
                      {m.delta != null && (
                        <div className={`text-xs font-semibold flex items-center mt-3 w-fit px-2 py-0.5 rounded-full ${
                          m.delta >= 0 ? 'bg-stable-bg/50 text-stable-text' : 'bg-critical-bg/50 text-critical-text'
                        }`}>
                          {m.delta < 0 ? <ArrowDown className="h-3 w-3 mr-1" /> : <Minus className="h-3 w-3 mr-1" />}
                          {m.delta}%
                        </div>
                      )}
                    </>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Performance Chart */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Accuracy Degradation Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-72 min-h-72 w-full min-w-0">
                {performance && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performance.timeSeries}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E0E3E5" />
                      <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: '#40484C', fontSize: 12 }} />
                      <YAxis domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{ fill: '#40484C', fontSize: 12 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#191C1D', color: '#EFF1F2', borderRadius: '8px' }} />
                      <Line type="monotone" dataKey="current" name="Current Accuracy" stroke="#B3261E" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="base" name="Baseline Accuracy" stroke="#70787D" strokeDasharray="5 5" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
