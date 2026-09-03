export interface Model {
  id: string
  name: string
  version: string
  algorithm: string
  environment: string
  status: 'production' | 'staging' | 'retired' | 'retraining'
  predictionsCount: number
  healthScore: number
  driftStatus: 'Low' | 'Moderate' | 'High'
  lastUpdated: string
}

export interface Alert {
  id: string
  modelId: string
  modelName: string
  severity: 'critical' | 'warning'
  category: string
  title: string
  description: string
  detectedAt: string
  status: 'unacknowledged' | 'investigating' | 'resolved'
}

export interface DriftScore {
  featureName: string
  baselineMean: number
  currentMean: number
  driftScore: number
  jsDistance?: number
  status: 'critical' | 'warning' | 'stable'
}
