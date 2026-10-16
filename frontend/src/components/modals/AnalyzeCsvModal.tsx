import { useState } from 'react'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { AlertTriangle, CheckCircle2, Loader2, Upload, X } from 'lucide-react'

interface Props {
  modelId: string
  isOpen: boolean
  onClose: () => void
  onComplete: () => Promise<void>
}

export function AnalyzeCsvModal({ modelId, isOpen, onClose, onComplete }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [columns, setColumns] = useState<string[]>([])
  const [predictionColumn, setPredictionColumn] = useState('')
  const [actualColumn, setActualColumn] = useState('')
  const [timestampColumn, setTimestampColumn] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleFile = async (selected: File | null) => {
    setFile(selected)
    setColumns([])
    setPredictionColumn('')
    setActualColumn('')
    setTimestampColumn('')
    setResult(null)
    setError(null)
    if (!selected) return
    const header = (await selected.slice(0, 1024 * 1024).text()).split(/\r?\n/, 1)[0]
    const parsed = header.split(',').map((column) => column.trim().replace(/^"|"$/g, '')).filter(Boolean)
    setColumns(parsed)
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!file) {
      setError('Choose a CSV file before submitting.')
      return
    }
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await api.analyzeCsv(modelId, file, {
        predictionColumn: predictionColumn || undefined,
        actualColumn: actualColumn || undefined,
        timestampColumn: timestampColumn || undefined,
      })
      setResult(response)
      await onComplete()
      window.dispatchEvent(new CustomEvent('model-data-invalidated', { detail: { modelId } }))
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Unable to analyze this CSV.')
    } finally {
      setIsLoading(false)
    }
  }

  const selectColumn = (label: string, value: string, onChange: (value: string) => void, optional = true) => (
    <label className="block text-sm font-semibold text-on-surface-variant">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 h-10 w-full rounded-lg border border-outline-variant bg-surface px-3 text-sm text-on-surface"
        required={!optional}
      >
        <option value="">{optional ? 'Not provided' : 'Select a column'}</option>
        {columns.map((column) => <option key={column} value={column}>{column}</option>)}
      </select>
    </label>
  )

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg space-y-5 rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <h2 className="text-title-lg font-bold text-on-surface">Analyze new data</h2>
          <button onClick={onClose} className="rounded-md p-1 text-on-surface-variant hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>
        <form onSubmit={submit} className="space-y-4">
          <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-outline-variant p-6 text-center hover:border-primary">
            <Upload className="h-8 w-8 text-primary" />
            <span className="text-sm font-semibold text-on-surface">{file?.name ?? 'Choose a CSV file'}</span>
            <input type="file" accept=".csv" className="hidden" onChange={(event) => void handleFile(event.target.files?.[0] ?? null)} />
          </label>
          {file && (
            <div className="space-y-3">
              {selectColumn('Prediction column', predictionColumn, setPredictionColumn)}
              {selectColumn('Actual/label column', actualColumn, setActualColumn)}
              {selectColumn('Timestamp column', timestampColumn, setTimestampColumn)}
              <p className="text-xs text-on-surface-variant">Provide both prediction and actual columns to compute labelled performance.</p>
            </div>
          )}
          {error && <div className="flex gap-2 rounded-lg border border-error/20 bg-error/10 p-3 text-xs text-error"><AlertTriangle className="h-4 w-4 shrink-0" />{error}</div>}
          {result && (
            <div className={`space-y-1 rounded-lg border p-3 text-xs ${result.status === 'completed' ? 'border-stable-text/30 bg-stable-bg/40 text-stable-text' : 'border-warning/30 bg-warning/10 text-warning-text'}`}>
              <div className="flex items-center gap-2 font-bold">
                {result.status === 'completed' ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
                {result.status === 'completed' ? 'Analysis complete' : 'Analysis needs more data'}
              </div>
              <div>{result.rows_ingested} rows ingested. Status: {result.status}</div>
              {result.status === 'completed' && <div>{result.drifts?.filter((drift: any) => ['warning', 'critical'].includes(drift.status)).length ?? 0} drifting features. Health: {result.healthScore ?? 'No data yet'}</div>}
              {result.status !== 'completed' && <div>The model was not evaluated because the uploaded rows did not produce an evaluated window.</div>}
              {result.warnings?.map((warning: string) => <div key={warning}>Warning: {warning}</div>)}
            </div>
          )}
          <div className="flex justify-end gap-3 border-t border-outline-variant pt-4">
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
            {result ? (
              <Button type="button" onClick={onClose}>View results</Button>
            ) : (
              <Button type="submit" disabled={isLoading || !file}>
                {isLoading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Analyzing...</> : 'Analyze CSV'}
              </Button>
            )}
          </div>
        </form>
      </div>
    </div>
  )
}
