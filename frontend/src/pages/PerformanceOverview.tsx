import { useState, useEffect } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

export default function PerformanceOverview() {
  const [models, setModels] = useState<any[]>([])
  const [perfMap, setPerfMap] = useState<Record<string, any>>({})

  useEffect(() => {
    api.getModels().then(async (modelList) => {
      setModels(modelList)
      const results: Record<string, any> = {}
      for (const m of modelList) {
        try {
          const perf = await api.getModelPerformance(m.id, '30d')
          results[m.id] = perf
        } catch {
          // ignore error
        }
      }
      setPerfMap(results)
    })
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-6 gap-4">
        <div>
          <h1 className="text-[32px] font-semibold leading-[40px] tracking-[-0.02em] text-on-surface mb-2">
            Model Performance Matrix
          </h1>
          <p className="text-[16px] leading-[24px] text-on-surface-variant">
            Cross-deployment evaluation of accuracy, precision, recall, and F1 degradation against baselines.
          </p>
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((m) => {
          const perf = perfMap[m.id]

          return (
            <Card key={m.id} className="flex flex-col">
              <CardHeader className="pb-3 border-b border-outline-variant">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-title-lg">{m.name}</CardTitle>
                    <CardDescription>{m.algorithm} • {m.version}</CardDescription>
                  </div>
                  <Badge variant={m.status === 'production' ? 'stable' : 'warning'} className="uppercase">
                    {m.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="p-6 space-y-4 flex-1">
                {perf ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    {perf.metrics.map((metric: any) => (
                      <div key={metric.label} className="p-3 bg-surface rounded-lg border border-outline-variant">
                        <div className="text-[11px] font-semibold uppercase text-on-surface-variant">{metric.label}</div>
                        <div className="text-title-lg font-bold text-on-surface mt-1">{metric.value}%</div>
                        <div className={`text-[10px] font-semibold flex items-center justify-center mt-1 ${
                          metric.delta >= 0 ? 'text-stable-text' : 'text-error'
                        }`}>
                          {metric.delta >= 0 ? '+' : ''}{metric.delta}%
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-6 text-center text-xs text-on-surface-variant animate-pulse">Loading model metrics...</div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Matrix Table */}
      <Card>
        <CardHeader>
          <CardTitle>Cross-Model Evaluation Matrix</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model Name</TableHead>
                <TableHead>Algorithm</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>Precision</TableHead>
                <TableHead>Recall</TableHead>
                <TableHead>F1 Score</TableHead>
                <TableHead>Health Score</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map((m) => {
                const perf = perfMap[m.id]
                const acc = perf?.metrics.find((x: any) => x.label === 'Accuracy')?.value
                const prec = perf?.metrics.find((x: any) => x.label === 'Precision')?.value
                const rec = perf?.metrics.find((x: any) => x.label === 'Recall')?.value
                const f1 = perf?.metrics.find((x: any) => x.label === 'F1 Score')?.value

                return (
                  <TableRow key={m.id}>
                    <TableCell className="font-bold text-on-surface">{m.name} <span className="text-xs font-normal text-on-surface-variant">({m.version})</span></TableCell>
                    <TableCell>{m.algorithm}</TableCell>
                    <TableCell className="font-semibold">{acc !== undefined ? `${acc}%` : 'N/A'}</TableCell>
                    <TableCell>{prec !== undefined ? `${prec}%` : 'N/A'}</TableCell>
                    <TableCell>{rec !== undefined ? `${rec}%` : 'N/A'}</TableCell>
                    <TableCell className="font-semibold">{f1 !== undefined ? `${f1}%` : 'N/A'}</TableCell>
                    <TableCell>
                      <span className={`font-bold ${m.healthScore > 80 ? 'text-stable-text' : m.healthScore > 50 ? 'text-warning-text' : 'text-error'}`}>
                        {m.healthScore}/100
                      </span>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
