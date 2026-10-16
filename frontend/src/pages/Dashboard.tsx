import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Progress } from "@/components/ui/progress"
import { Search, MoreHorizontal, Plus } from "lucide-react"

export default function Dashboard() {
  const navigate = useNavigate()
  const [data, setData] = useState<any>(null)
  const [models, setModels] = useState<any[]>([])
  const [tableSearch, setTableSearch] = useState("")
  const [dateFilter, setDateFilter] = useState("30d")
  const [statusFilter, setStatusFilter] = useState("all")
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setIsLoading(true)
    setError(null)
    Promise.all([api.getDashboardSummary(dateFilter), api.getModels()])
      .then(([summary, modelList]) => {
        setData(summary)
        setModels(modelList)
      })
      .catch(fetchError => setError(fetchError instanceof Error ? fetchError.message : 'Unable to load dashboard metrics.'))
      .finally(() => setIsLoading(false))
  }, [dateFilter])

  if (isLoading && !data) return <div className="flex items-center justify-center h-full text-on-surface-variant animate-pulse p-10">Fetching metrics...</div>
  if (error && !data) return <div className="flex items-center justify-center h-full text-error p-10">Unable to load dashboard metrics: {error}</div>
  if (!data) return <div className="flex items-center justify-center h-full text-on-surface-variant p-10">Dashboard metrics are unavailable.</div>

  const filteredModels = models.filter(m => {
    const matchesSearch = m.name.toLowerCase().includes(tableSearch.toLowerCase()) || m.algorithm.toLowerCase().includes(tableSearch.toLowerCase())
    const matchesStatus = statusFilter === "all" || m.status === statusFilter
    return matchesSearch && matchesStatus
  })

  const openRegisterModal = () => {
    window.dispatchEvent(new CustomEvent('open-modal', { detail: 'Register Model' }))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 gap-4">
        <div>
          <h1 className="text-headline-lg font-semibold tracking-tight text-on-surface mb-2">Model Monitoring Overview</h1>
          <p className="text-body-lg text-on-surface-variant">Real-time visibility into production model health, drift and performance.</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="h-10 px-3 rounded-lg border border-outline-variant bg-surface text-on-surface text-sm font-medium focus:outline-none focus:border-primary"
          >
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
          <Button onClick={openRegisterModal}>
            <Plus className="w-4 h-4 mr-2" />
            Register Model
          </Button>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Model Health</div>
            <div className="flex items-end justify-between">
              <div className="text-headline-lg font-semibold">{data.healthAvg != null ? `${data.healthAvg}/100` : '—'}</div>
              <Badge
                variant={data.healthBadge === 'Healthy' ? 'stable' : data.healthBadge === 'Critical' ? 'critical' : 'warning'}
                className="mb-1"
              >
                {data.healthBadge || 'Unknown'}
              </Badge>
            </div>
            <div className="text-sm font-medium text-on-surface-variant mt-3">
              Weighted PSI, performance, and open critical alerts
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Active Models</div>
            <div className="text-headline-lg font-semibold">{data.activeModels.total}</div>
            <div className="text-sm font-medium text-on-surface-variant mt-3">
              {data.activeModels.prod} Prod, {data.activeModels.staging} Staging
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Predictions Monitored</div>
            <div className="text-headline-lg font-semibold">
              {data.predictionsTotal >= 1_000_000
                ? `${(data.predictionsTotal / 1_000_000).toFixed(1)}M`
                : data.predictionsTotal >= 1000
                  ? `${(data.predictionsTotal / 1000).toFixed(1)}K`
                  : data.predictionsTotal}
            </div>
            <div className="text-sm font-medium text-on-surface-variant mt-3">
              {data.predictionsTrend == null
                ? `Logged in ${dateFilter}`
                : `${data.predictionsTrend >= 0 ? '+' : ''}${data.predictionsTrend}% vs prior window`}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-label-md font-semibold text-on-surface-variant uppercase mb-2">Active Alerts</div>
            <div className="text-headline-lg font-semibold">{data.activeAlerts.total}</div>
            <div className="flex items-center gap-2 mt-3">
              <Badge variant="critical">{data.activeAlerts.critical} Critical</Badge>
              <Badge variant="warning">{data.activeAlerts.warning} Warning</Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Analytics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 flex flex-col">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle>Model Health Over Time</CardTitle>
            <Button variant="ghost" size="icon" className="h-8 w-8 text-on-surface-variant -m-2">
              <MoreHorizontal className="h-5 w-5" />
            </Button>
          </CardHeader>
          <CardContent className="flex-1 p-6">
            <div className="h-[250px] min-h-[250px] w-full min-w-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.healthOverTime} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorHealth" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0047c1" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#0047c1" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#c3c6d8" />
                  <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#434655', fontSize: 12}} />
                  <YAxis domain={['auto', 100]} axisLine={false} tickLine={false} tick={{fill: '#434655', fontSize: 12}} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="value" stroke="#0047c1" strokeWidth={3} fillOpacity={1} fill="url(#colorHealth)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Feature Drift Alerts</CardTitle>
            <CardDescription>Top divergent features across models</CardDescription>
          </CardHeader>
          <CardContent className="flex-1 p-0">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-[120px]">FEATURE</TableHead>
                  <TableHead className="text-right">SCORE</TableHead>
                  <TableHead className="text-right">STATUS</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(data.featureDriftAlerts || []).length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={3} className="text-center py-8 text-sm text-on-surface-variant">
                      No drift scores in this window. Upload a baseline and send live predictions first.
                    </TableCell>
                  </TableRow>
                ) : (data.featureDriftAlerts || []).map((item: any, i: number) => (
                  <TableRow key={i}>
                    <TableCell className="font-medium text-sm truncate max-w-[120px]" title={item.feature}>{item.feature}</TableCell>
                    <TableCell className="text-right text-sm">{item.score}</TableCell>
                    <TableCell className="text-right">
                      <Badge variant={item.status as any} className="text-[10px] uppercase leading-none py-1">{item.status}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* Models Table */}
      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-4">
          <CardTitle>Production Models</CardTitle>
          <div className="flex items-center gap-3">
             <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-on-surface-variant" />
                <input 
                  type="text" 
                  placeholder="Find model..." 
                  value={tableSearch}
                  onChange={(e) => setTableSearch(e.target.value)}
                  className="h-9 rounded-md border border-outline-variant bg-surface-lowest pl-8 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 w-[200px]"
                />
              </div>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="h-9 rounded-md border border-outline-variant bg-surface-lowest px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20"
              >
                <option value="all">All Statuses</option>
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="retraining">Retraining</option>
              </select>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>MODEL VERSION</TableHead>
                <TableHead>ALGORITHM</TableHead>
                <TableHead>PREDICTIONS</TableHead>
                <TableHead className="min-w-[150px]">HEALTH</TableHead>
                <TableHead>DRIFT</TableHead>
                <TableHead>UPDATED</TableHead>
                <TableHead>STATUS</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredModels.map((model) => (
                <TableRow key={model.id} onClick={() => navigate(`/models/${model.id}`)} className="cursor-pointer group hover:bg-surface-low/60 transition-colors">
                  <TableCell>
                    <div className="font-bold group-hover:text-primary transition-colors">{model.name}</div>
                    <div className="text-xs text-on-surface-variant mt-0.5">{model.version}</div>
                  </TableCell>
                  <TableCell className="text-sm">{model.algorithm}</TableCell>
                  <TableCell className="text-sm">{model.predictionsCount ? model.predictionsCount.toLocaleString() : '0'}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                       <Progress 
                        value={model.healthScore} 
                        indicatorColor={model.healthScore > 80 ? 'bg-stable-text' : model.healthScore > 50 ? 'bg-warning-text' : 'bg-error'}
                        className="w-full bg-surface-low" 
                       />
                       <span className="text-xs font-semibold w-6">{model.healthScore}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className={`text-sm font-semibold ${
                      model.driftStatus === 'High' ? 'text-error' : model.driftStatus === 'Moderate' ? 'text-warning-text' : 'text-stable-text'
                    }`}>{model.driftStatus}</span>
                  </TableCell>
                  <TableCell className="text-sm text-on-surface-variant">{model.lastUpdated}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className={`gap-1.5 border-outline-variant ${model.status === 'production' ? 'bg-surface-lowest text-on-surface' : ''}`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${model.status === 'production' ? 'bg-stable-text' : 'bg-primary'}`}></span>
                      {model.status === 'production' ? 'Active' : model.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <div className="h-4"></div>
    </div>
  )
}
