import { useState } from 'react'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { X, Upload, CheckCircle2, AlertTriangle, Loader2, Database } from 'lucide-react'

interface Props {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

export function RegisterModelModal({ isOpen, onClose, onSuccess }: Props) {
  const [activeTab, setActiveTab] = useState<'upload' | 'form'>('upload')
  const [name, setName] = useState('')
  const [version, setVersion] = useState('v1.0')
  const [algorithm, setAlgorithm] = useState('XGBoost')
  const [environment, setEnvironment] = useState('production')
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await api.createModel({
        name: name || (file ? file.name.replace('.csv', '') : 'New ML Model'),
        version,
        algorithm,
        environment,
      })
      const baseline = file ? await api.uploadBaseline(res.id, file) : null
      setResult({ ...res, baseline })
      setTimeout(() => {
        onSuccess()
      }, 1500)
    } catch (err: any) {
      setError(err.message || 'Error processing dataset')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in">
      <div className="w-full max-w-xl rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-primary/10 text-primary rounded-lg">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-title-lg font-bold text-on-surface">Register & Monitor Model</h2>
              <p className="text-body-sm text-on-surface-variant">Register a model and optionally store a training baseline CSV</p>
            </div>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface p-1 rounded-md hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Tab Selection */}
        <div className="flex rounded-lg bg-surface p-1 border border-outline-variant">
          <button
            type="button"
            onClick={() => setActiveTab('upload')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              activeTab === 'upload' ? 'bg-surface-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            Baseline CSV upload
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('form')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              activeTab === 'form' ? 'bg-surface-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            📝 Manual Registration
          </button>
        </div>

        <form onSubmit={handleIngest} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-label-md font-semibold text-on-surface-variant mb-1">Model Name</label>
              <input
                type="text"
                placeholder="e.g. Fraud Classifier"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-sm focus:outline-none focus:border-primary"
              />
            </div>
            <div>
              <label className="block text-label-md font-semibold text-on-surface-variant mb-1">Version</label>
              <input
                type="text"
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-sm focus:outline-none focus:border-primary"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-label-md font-semibold text-on-surface-variant mb-1">Algorithm</label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-sm focus:outline-none focus:border-primary"
              >
                <option value="XGBoost">XGBoost</option>
                <option value="Random Forest">Random Forest</option>
                <option value="LightGBM">LightGBM</option>
                <option value="Neural Network">Neural Network</option>
                <option value="Logistic Regression">Logistic Regression</option>
              </select>
            </div>
            <div>
              <label className="block text-label-md font-semibold text-on-surface-variant mb-1">Environment</label>
              <select
                value={environment}
                onChange={(e) => setEnvironment(e.target.value)}
                className="w-full h-10 px-3 rounded-lg border border-outline-variant bg-surface text-sm focus:outline-none focus:border-primary"
              >
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="retraining">Retraining</option>
              </select>
            </div>
          </div>

          {activeTab === 'upload' && (
            <div className="space-y-2">
              <label className="block text-label-md font-semibold text-on-surface-variant">Upload Live Dataset (CSV)</label>
              <div className="border-2 border-dashed border-outline-variant hover:border-primary rounded-xl p-6 text-center cursor-pointer bg-surface/50 hover:bg-surface transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  id="csv-file-input"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
                <label htmlFor="csv-file-input" className="cursor-pointer flex flex-col items-center gap-2">
                  <Upload className="h-8 w-8 text-primary" />
                  <span className="text-sm font-semibold text-on-surface">
                    {file ? file.name : 'Click to select CSV dataset'}
                  </span>
                  <span className="text-xs text-on-surface-variant">
                    {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Contains feature values or prediction logs to calculate real-time drift & status'}
                  </span>
                </label>
              </div>
            </div>
          )}

          {error && (
            <div className="p-3 bg-error/10 border border-error/20 text-error rounded-lg text-xs flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}

          {result && (
            <div className="p-4 bg-stable-bg/40 border border-stable-text/30 text-stable-text rounded-lg text-xs space-y-1 animate-in fade-in">
              <div className="flex items-center gap-2 font-bold text-sm">
                <CheckCircle2 className="h-4 w-4" /> Live Status Generated!
              </div>
              <p>Model Health Score: <span className="font-bold">{result.model.healthScore}/100</span> ({result.model.driftStatus} Drift)</p>
              <p className="text-[11px] opacity-80">{result.drifts?.length || 0} numerical features calculated live.</p>
            </div>
          )}

          <div className="flex justify-end gap-3 border-t border-outline-variant pt-4">
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Calculating Live Status...
                </>
              ) : (
                'Register & Ingest Live'
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
