import { useState, useEffect } from "react"
import { api, connectAlertsWebSocket } from "@/lib/api"
import type { Alert } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { AlertCircle, AlertTriangle, Settings, ChevronLeft, ChevronRight, CheckCircle2 } from "lucide-react"

export default function AlertCenter() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [activeTab, setActiveTab] = useState<'all' | 'critical' | 'warning' | 'resolved'>('all')
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalCount, setTotalCount] = useState(0)

  const fetchAlerts = () => {
    const statusFilter = activeTab === 'all' ? 'active' : activeTab === 'resolved' ? 'resolved' : undefined
    const severityFilter = (activeTab === 'critical' || activeTab === 'warning') ? activeTab : undefined

    api.getAlerts({
      status: statusFilter,
      severity: severityFilter,
      page: currentPage,
      perPage: 6,
    }).then(res => {
      setAlerts(res.items)
      setTotalPages(res.totalPages)
      setTotalCount(res.total)
    })
  }

  useEffect(() => {
    fetchAlerts()
  }, [activeTab, currentPage])

  // Real-time WebSocket connection for live alert updates
  useEffect(() => {
    const disconnect = connectAlertsWebSocket((event) => {
      if (event.event === 'alert_updated' || event.event === 'new_alert') {
        fetchAlerts()
      }
    })
    return disconnect
  }, [activeTab, currentPage])

  const handleStatusChange = async (alertId: string, newStatus: 'unacknowledged' | 'investigating' | 'resolved') => {
    await api.updateAlertStatus(alertId, newStatus)
    fetchAlerts()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 gap-4">
        <div>
          <h1 className="text-[32px] font-semibold leading-[40px] tracking-[-0.02em] text-on-surface mb-2">Alert Center</h1>
          <p className="text-[16px] leading-[24px] text-on-surface-variant">Manage and resolve active monitoring alerts across your ML deployments.</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline">
            <Settings className="w-4 h-4 mr-2" />
            Configure Rules
          </Button>
        </div>
      </div>

      {/* Tab Row */}
      <div className="flex items-center gap-1 border-b border-outline-variant">
        {[
          { key: 'all' as const, label: 'All Active' },
          { key: 'critical' as const, label: 'Critical' },
          { key: 'warning' as const, label: 'Warning' },
          { key: 'resolved' as const, label: 'Resolved' },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => { setActiveTab(tab.key); setCurrentPage(1); }}
            className={`relative px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'text-primary after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-primary'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[100px]">Severity</TableHead>
                <TableHead>Title & Description</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Detected</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {alerts.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-12 text-on-surface-variant">
                    No alerts found for this filter.
                  </TableCell>
                </TableRow>
              ) : (
                alerts.map((alert) => (
                  <TableRow key={alert.id}>
                    <TableCell>
                      {alert.severity === 'critical' ? (
                        <Badge variant="critical" className="gap-1">
                          <AlertCircle className="w-3.5 h-3.5" />
                          Critical
                        </Badge>
                      ) : (
                        <Badge variant="warning" className="gap-1">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          Warning
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="font-semibold text-on-surface text-sm">{alert.title}</div>
                      <div className="text-xs text-on-surface-variant mt-0.5">{alert.description}</div>
                    </TableCell>
                    <TableCell className="text-sm text-on-surface font-medium">
                      {alert.modelName}
                    </TableCell>
                    <TableCell className="capitalize text-xs text-on-surface-variant">
                      {alert.category.replace('_', ' ')}
                    </TableCell>
                    <TableCell className="text-xs text-on-surface-variant">
                      {alert.detectedAt}
                    </TableCell>
                    <TableCell>
                      <span className={`inline-flex items-center text-xs font-semibold px-2 py-0.5 rounded-full capitalize ${
                        alert.status === 'unacknowledged'
                          ? 'bg-critical-bg/50 text-critical-text'
                          : alert.status === 'investigating'
                          ? 'bg-warning-bg/50 text-warning-text'
                          : 'bg-stable-bg/50 text-stable-text'
                      }`}>
                        {alert.status}
                      </span>
                    </TableCell>
                    <TableCell className="text-right space-x-2">
                      {alert.status === 'unacknowledged' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleStatusChange(alert.id, 'investigating')}
                        >
                          Investigate
                        </Button>
                      )}
                      {alert.status !== 'resolved' && (
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => handleStatusChange(alert.id, 'resolved')}
                        >
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Resolve
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          {/* Pagination */}
          <div className="flex items-center justify-between px-6 py-4 border-t border-outline-variant">
            <span className="text-xs text-on-surface-variant">
              Showing page {currentPage} of {totalPages} ({totalCount} total alerts)
            </span>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage <= 1}
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage >= totalPages}
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
