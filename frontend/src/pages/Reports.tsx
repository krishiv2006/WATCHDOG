import { useState, useEffect } from "react"
import { api } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Download, FileText, Play, AlertCircle, Loader2 } from "lucide-react"

interface ReportItem {
  id: string
  name: string
  modelId: string
  modelName: string
  dateRange: string
  status: 'ready' | 'generating' | 'failed'
  generatedAt: string
}

export default function Reports() {
  const [reports, setReports] = useState<ReportItem[]>([])
  const [models, setModels] = useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [reportType, setReportType] = useState<string>('drift_analysis')
  const [isGenerating, setIsGenerating] = useState(false)
  const [dateStart, setDateStart] = useState<string>(() => {
    const d = new Date()
    d.setDate(d.getDate() - 30)
    return d.toISOString().slice(0, 10)
  })
  const [dateEnd, setDateEnd] = useState<string>(() => new Date().toISOString().slice(0, 10))

  const fetchReports = () => {
    api.getReports().then(res => setReports(res as ReportItem[]))
  }

  useEffect(() => {
    fetchReports()
    api.getModels().then(res => {
      setModels(res)
      if (res.length > 0) setSelectedModel(res[0].id)
    })
  }, [])

  const handleGenerate = async () => {
    if (!selectedModel) return
    setIsGenerating(true)
    try {
      await api.generateReport({
        modelId: selectedModel,
        dateRangeStart: new Date(dateStart).toISOString(),
        dateRangeEnd: new Date(dateEnd + 'T23:59:59').toISOString(),
        reportType: reportType,
      })
      fetchReports()
      // Poll until ready
      const interval = setInterval(() => {
        api.getReports().then(updated => {
          setReports(updated as ReportItem[])
          if (!updated.some(r => r.status === 'generating')) {
            clearInterval(interval)
            setIsGenerating(false)
          }
        })
      }, 2000)
    } catch {
      setIsGenerating(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 gap-4">
        <div>
          <h1 className="text-[32px] font-semibold leading-[40px] tracking-[-0.02em] text-on-surface mb-2">Monitoring Reports</h1>
          <p className="text-[16px] leading-[24px] text-on-surface-variant">Configure and generate detailed performance analysis for your deployed models.</p>
        </div>
      </div>

      {/* Report Configuration */}
      <Card>
        <CardHeader className="flex flex-row items-center gap-2">
          <FileText className="h-5 w-5 text-primary" />
          <CardTitle>Report Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="text-label-md font-semibold text-on-surface-variant block mb-2">Target Model</label>
              <select
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-on-surface text-sm focus:outline-none focus:border-primary"
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name} ({m.version})</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-label-md font-semibold text-on-surface-variant block mb-2">Report Type</label>
              <select
                value={reportType}
                onChange={e => setReportType(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-on-surface text-sm focus:outline-none focus:border-primary"
              >
                <option value="drift_analysis">Feature & Data Drift Analysis</option>
                <option value="performance">Model Performance Metrics</option>
                <option value="comprehensive">Comprehensive Audit Report</option>
              </select>
            </div>
            <div>
              <label className="text-label-md font-semibold text-on-surface-variant block mb-2">Date From</label>
              <input
                type="date"
                value={dateStart}
                onChange={e => setDateStart(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-on-surface text-sm focus:outline-none focus:border-primary"
              />
            </div>
            <div>
              <label className="text-label-md font-semibold text-on-surface-variant block mb-2">Date To</label>
              <input
                type="date"
                value={dateEnd}
                onChange={e => setDateEnd(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-on-surface text-sm focus:outline-none focus:border-primary"
              />
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={handleGenerate} disabled={isGenerating}>
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Generating Report...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Generate Report
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* History */}
      <Card>
        <CardHeader>
          <CardTitle>Report History</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Report Name</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Date Range</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Generated</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {reports.map((report) => (
                <TableRow key={report.id}>
                  <TableCell className="font-semibold text-on-surface text-sm">{report.name}</TableCell>
                  <TableCell className="text-sm text-on-surface-variant">{report.modelName}</TableCell>
                  <TableCell className="text-xs text-on-surface-variant">{report.dateRange}</TableCell>
                  <TableCell>
                    {report.status === 'ready' && <Badge variant="stable">Ready</Badge>}
                    {report.status === 'generating' && (
                      <Badge variant="warning" className="gap-1 animate-pulse">
                        <Loader2 className="w-3 h-3 animate-spin" /> Generating
                      </Badge>
                    )}
                    {report.status === 'failed' && (
                      <Badge variant="critical" className="gap-1">
                        <AlertCircle className="w-3 h-3" /> Failed
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-on-surface-variant">{report.generatedAt}</TableCell>
                  <TableCell className="text-right">
                    {report.status === 'ready' ? (
                      <Button size="sm" variant="outline" onClick={() => api.downloadReport(report.id)}>
                        <Download className="w-3 h-3 mr-1" /> Download CSV
                      </Button>
                    ) : (
                      <Button size="sm" variant="ghost" disabled>
                        Unavailable
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
