import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Plus, Search, RefreshCw } from 'lucide-react'
import type { Model } from '@/types'

export default function ModelsList() {
  const navigate = useNavigate()
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const fetchModels = () => {
    setLoading(true)
    api.getModels()
      .then(setModels)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchModels() }, [])

  const filtered = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase()) ||
    m.algorithm.toLowerCase().includes(search.toLowerCase()) ||
    m.version.toLowerCase().includes(search.toLowerCase())
  )

  const openRegisterModal = () => {
    window.dispatchEvent(new CustomEvent('open-modal', { detail: 'Register Model' }))
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 gap-4">
        <div>
          <h1 className="text-headline-lg font-semibold tracking-tight text-on-surface mb-2">Registered Models</h1>
          <p className="text-body-lg text-on-surface-variant">All models being monitored by Watchdog.</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={fetchModels}>
            <RefreshCw className="w-4 h-4 mr-2" /> Refresh
          </Button>
          <Button onClick={openRegisterModal}>
            <Plus className="w-4 h-4 mr-2" /> Register Model
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4">
          <CardTitle>All Models ({models.length})</CardTitle>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-on-surface-variant" />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="h-9 rounded-md border border-outline-variant bg-surface-lowest pl-8 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 w-[220px]"
            />
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="py-16 text-center text-on-surface-variant animate-pulse">Loading models...</div>
          ) : error ? (
            <div className="py-16 text-center text-error text-sm">{error}</div>
          ) : filtered.length === 0 ? (
            <div className="py-16 text-center space-y-3">
              <p className="text-on-surface-variant text-sm">
                {models.length === 0 ? 'No models registered yet.' : 'No models match your search.'}
              </p>
              {models.length === 0 && (
                <Button onClick={openRegisterModal}>
                  <Plus className="w-4 h-4 mr-2" /> Register your first model
                </Button>
              )}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Algorithm</TableHead>
                  <TableHead>Environment</TableHead>
                  <TableHead>Predictions</TableHead>
                  <TableHead className="min-w-[160px]">Health</TableHead>
                  <TableHead>Drift</TableHead>
                  <TableHead>Last Evaluated</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map(model => (
                  <TableRow
                    key={model.id}
                    onClick={() => navigate(`/models/${model.id}`)}
                    className="cursor-pointer group hover:bg-surface-low/60 transition-colors"
                  >
                    <TableCell>
                      <div className="font-bold group-hover:text-primary transition-colors">{model.name}</div>
                      <div className="text-xs text-on-surface-variant mt-0.5">{model.version}</div>
                    </TableCell>
                    <TableCell className="text-sm">{model.algorithm}</TableCell>
                    <TableCell className="text-sm capitalize">{model.environment}</TableCell>
                    <TableCell className="text-sm">
                      {model.predictionsCount ? model.predictionsCount.toLocaleString() : '0'}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress
                          value={model.healthScore}
                          indicatorColor={
                            model.healthScore > 80 ? 'bg-stable-text' :
                            model.healthScore > 50 ? 'bg-warning-text' : 'bg-error'
                          }
                          className="w-full bg-surface-low"
                        />
                        <span className="text-xs font-semibold w-8 shrink-0">{model.healthScore}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className={`text-sm font-semibold ${
                        model.driftStatus === 'High' ? 'text-error' :
                        model.driftStatus === 'Moderate' ? 'text-warning-text' : 'text-stable-text'
                      }`}>
                        {model.driftStatus}
                      </span>
                    </TableCell>
                    <TableCell className="text-sm text-on-surface-variant">{model.lastUpdated}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className="gap-1.5 border-outline-variant"
                      >
                        <span className={`h-1.5 w-1.5 rounded-full ${
                          model.status === 'production' ? 'bg-stable-text' : 'bg-primary'
                        }`} />
                        {model.status === 'production' ? 'Active' : model.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
