import type { Model, Alert, DriftScore } from '@/types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
const WS_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000/ws/alerts'
const INGEST_API_KEY = import.meta.env.VITE_INGEST_API_KEY as string | undefined

function ingestHeaders(extra?: HeadersInit): HeadersInit {
  const headers = new Headers(extra)
  if (INGEST_API_KEY) headers.set('X-API-Key', INGEST_API_KEY)
  return headers
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  getDashboardSummary: async (window: string = '30d') => {
    return fetchJson<{
      healthAvg: number | null
      healthBadge: string
      healthTrend: number | null
      activeModels: { total: number; prod: number; staging: number }
      predictionsTotal: number
      predictionsTrend: number | null
      activeAlerts: { total: number; critical: number; warning: number }
      healthOverTime: Array<{ time: string; value: number | null }>
      featureDriftAlerts: Array<{ feature: string; score: string; status: 'stable' | 'warning' | 'critical' }>
    }>(`${BASE_URL}/dashboard/summary?window=${window}`)
  },

  getPublicSettings: async () => {
    return fetchJson<{
      ingestAuthRequired: boolean
      ingestHeader: string
      evalWindowHours: number
      driftCheckIntervalMinutes: number
      minSampleSize: number
      psiWarning: number
      psiCritical: number
      schedulerEnabled: boolean
    }>(`${BASE_URL}/settings`)
  },

  getModels: async () => {
    return fetchJson<Model[]>(`${BASE_URL}/models`)
  },

  getModel: async (id: string) => {
    return fetchJson<Model>(`${BASE_URL}/models/${id}`)
  },

  getModelDrift: async (id: string, window: string = '30d') => {
    return fetchJson<DriftScore[]>(`${BASE_URL}/models/${id}/drift?window=${window}`)
  },

  getModelPerformance: async (id: string, window: string = '30d') => {
    return fetchJson<{
      metrics: Array<{ label: string; value: number | null; baseline: number | null; delta: number | null; reason?: string }>
      timeSeries: Array<{ time: string; current: number | null; base: number | null }>
      hasLabels: boolean
      sampleSize: number
      labelCoverage: number | null
      taskType: string | null
    }>(`${BASE_URL}/models/${id}/performance?window=${window}`)
  },

  reevaluateModel: async (id: string) => {
    return fetchJson<{ status: string; model_id: string }>(`${BASE_URL}/models/${id}/reevaluate`, {
      method: 'POST',
    })
  },

  analyzeCsv: async (id: string, file: File, columns: {
    predictionColumn?: string
    actualColumn?: string
    timestampColumn?: string
  }) => {
    const formData = new FormData()
    formData.append('file', file)
    if (columns.predictionColumn) formData.append('prediction_column', columns.predictionColumn)
    if (columns.actualColumn) formData.append('actual_column', columns.actualColumn)
    if (columns.timestampColumn) formData.append('timestamp_column', columns.timestampColumn)
    return fetchJson<{
      status: string
      rows_ingested: number
      warnings: string[]
      drifts: DriftScore[]
      healthScore: number | null
      perfStatus: string
      performance: {
        status: string
        nLabeled: number
        labelCoverage: number | null
        accuracy: number | null
        precision: number | null
        recall: number | null
        f1: number | null
        mae: number | null
        rmse: number | null
        r2: number | null
      }
      n: number
      minimum_required: number | null
      duplicate: boolean
    }>(`${BASE_URL}/models/${id}/analyze-csv`, {
      method: 'POST',
      body: formData,
      headers: ingestHeaders(),
    })
  },

  getAlerts: async (params?: { status?: string; severity?: string; modelId?: string; page?: number; perPage?: number }) => {
    const query = new URLSearchParams()
    if (params?.status) query.set('status', params.status)
    if (params?.severity) query.set('severity', params.severity)
    if (params?.modelId) query.set('model_id', params.modelId)
    if (params?.page) query.set('page', String(params.page))
    if (params?.perPage) query.set('per_page', String(params.perPage))

    const url = `${BASE_URL}/alerts${query.toString() ? `?${query.toString()}` : ''}`
    return fetchJson<{
      items: Alert[]
      total: number
      page: number
      perPage: number
      totalPages: number
    }>(url)
  },

  updateAlertStatus: async (id: string, status: 'unacknowledged' | 'investigating' | 'resolved') => {
    return fetchJson<Alert>(`${BASE_URL}/alerts/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    })
  },

  getReports: async () => {
    return fetchJson<Array<{
      id: string
      name: string
      modelId: string
      modelName: string
      dateRange: string
      status: 'ready' | 'generating' | 'failed'
      generatedAt: string
    }>>(`${BASE_URL}/reports`)
  },

  generateReport: async (payload: { modelId: string; dateRangeStart: string; dateRangeEnd: string; reportType: string }) => {
    return fetchJson<{ id: string; status: string }>(`${BASE_URL}/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: payload.modelId,
        date_range_start: payload.dateRangeStart,
        date_range_end: payload.dateRangeEnd,
        report_type: payload.reportType,
      }),
    })
  },

  downloadReport: (id: string) => {
    window.open(`${BASE_URL}/reports/${id}/download`, '_blank')
  },

  createModel: async (payload: { name: string; version: string; algorithm: string; environment: string }) => {
    return fetchJson<{ id: string; name: string; status: string }>(`${BASE_URL}/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },

  uploadBaseline: async (id: string, file: File, targetColumn?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (targetColumn) formData.append('target_column', targetColumn)
    return fetchJson<{
      status: string
      baselineVersionId: string
      versionNumber: number
      featureCount: number
      sampleSize: number
      targetColumn: string | null
      predictionDistribution: Record<string, unknown> | null
      warnings: string[]
    }>(`${BASE_URL}/models/${id}/baseline`, {
      method: 'POST',
      body: formData,
      headers: ingestHeaders(),
    })
  },

  getAlertRules: async () => {
    return fetchJson<Array<{
      id: string
      modelId?: string
      metric: string
      thresholdWarning: number
      thresholdCritical: number
      enabled: boolean
    }>>(`${BASE_URL}/alert-rules`)
  },

  updateAlertRule: async (id: string, data: { thresholdWarning?: number; thresholdCritical?: number; enabled?: boolean }) => {
    return fetchJson(`${BASE_URL}/alert-rules/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        threshold_warning: data.thresholdWarning,
        threshold_critical: data.thresholdCritical,
        enabled: data.enabled,
      }),
    })
  },
}

/**
 * Real-time WebSocket hook/client helper for alerts.
 */
export function connectAlertsWebSocket(onMessage: (data: any) => void): () => void {
  const socket = new WebSocket(WS_URL)

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch {
      // ignore non-json
    }
  }

  // Ping interval to keep connection alive
  const interval = setInterval(() => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send('ping')
    }
  }, 30000)

  return () => {
    clearInterval(interval)
    socket.close()
  }
}
