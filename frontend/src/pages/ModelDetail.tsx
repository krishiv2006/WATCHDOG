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

export default function ModelDetail() {
  const { id } = useParams()
  const [model, setModel] = useState<Model | null>(null)
  const [drift, setDrift] = useState<DriftScore[]>([])
  const [performance, setPerformance] = useState<{
    metrics: Array<{ label: string; value: number; baseline: number; delta: number }>
    timeSeries: Array<{ time: string; current: number; base: number }>
  } | null>(null)
  const [isReevaluating, setIsReevaluating] = useState(false)

  const fetchData = () => {
    if (id) {
      api.getModel(id).then(setModel)
      api.getModelDrift(id).then(setDrift)
      api.getModelPerformance(id).then(setPerformance)
    }
  }

  useEffect(() => {
    fetchData()
  }, [id])

  const handleReevaluate = async () => {
    if (!id) return
    setIsReevaluating(true)
    await api.reevaluateModel(id)
    setTimeout(() => {
      fetchData()
      setIsReevaluating(false)
    }, 1500)
  }

  if (!model) return <div className="p-10 animate-pulse text-on-surface-variant text-center">Loading model metrics...</div>

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
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {drift.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center py-8 text-on-surface-variant">
                        No feature drift calculated yet.
                      </TableCell>
                    </TableRow>
                  ) : (
                    drift.map((row) => (
                      <TableRow key={row.featureName}>
                        <TableCell className="font-semibold text-on-surface">{row.featureName}</TableCell>
                        <TableCell>{row.baselineMean.toLocaleString()}</TableCell>
                        <TableCell>{row.currentMean.toLocaleString()}</TableCell>
                        <TableCell className="font-semibold">{row.driftScore.toFixed(4)}</TableCell>
                        <TableCell className="text-on-surface-variant">
                          {row.jsDistance !== undefined ? row.jsDistance.toFixed(4) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {row.status === 'stable' && <Badge variant="stable">Stable</Badge>}
                          {row.status === 'warning' && <Badge variant="warning">Warning</Badge>}
                          {row.status === 'critical' && <Badge variant="critical">Critical</Badge>}
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
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {performance?.metrics.map((m) => (
              <Card key={m.label}>
                <CardContent className="p-6">
                  <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">{m.label}</div>
                  <div className="flex items-end justify-between">
                    <div className="text-headline-lg font-semibold">{m.value}%</div>
                    <div className="text-xs text-on-surface-variant mb-1">Base: {m.baseline}%</div>
                  </div>
                  <div className={`text-xs font-semibold flex items-center mt-3 w-fit px-2 py-0.5 rounded-full ${
                    m.delta >= 0
                      ? 'bg-stable-bg/50 text-stable-text'
                      : 'bg-critical-bg/50 text-critical-text'
                  }`}>
                    {m.delta < 0 ? <ArrowDown className="h-3 w-3 mr-1" /> : <Minus className="h-3 w-3 mr-1" />}
                    {m.delta}%
                  </div>
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
              <div className="h-72 w-full">
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
