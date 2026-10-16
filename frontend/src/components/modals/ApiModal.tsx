import { X, Code2, ExternalLink, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useState } from 'react'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export function ApiModal({ isOpen, onClose }: Props) {
  const [copied, setCopied] = useState(false)
  if (!isOpen) return null

  const endpoints = [
    { method: 'GET', path: '/api/dashboard/summary', desc: 'Fetch overall dashboard health & metrics' },
    { method: 'GET', path: '/api/models', desc: 'List all registered ML models & statuses' },
    { method: 'POST', path: '/api/models/{id}/baseline', desc: 'Upload a training CSV and create an immutable baseline version' },
    { method: 'POST', path: '/api/models/{id}/predictions', desc: 'Ingest single or batch production predictions idempotently' },
    { method: 'POST', path: '/api/models/{id}/ground-truth', desc: 'Attach delayed labels to previously ingested predictions' },
    { method: 'GET', path: '/api/models/{id}/drift', desc: 'Get feature-level PSI and JS distance scores' },
    { method: 'GET', path: '/api/models/{id}/performance', desc: 'Get model accuracy degradation over time' },
    { method: 'GET', path: '/api/alerts', desc: 'Fetch paginated alerts with status and severity filters' },
    { method: 'GET', path: '/api/reports/{id}/download', desc: 'Download generated monitoring report CSV' },
  ]

  const handleCopyBase = () => {
    navigator.clipboard.writeText('http://localhost:8000/api')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in">
      <div className="w-full max-w-2xl rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl space-y-6 max-h-[85vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-primary/10 text-primary rounded-lg">
              <Code2 className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-title-lg font-bold text-on-surface">REST API Reference</h2>
              <p className="text-body-sm text-on-surface-variant">FastAPI Swagger Specification & Integration Endpoints</p>
            </div>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface p-1 rounded-md hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 bg-surface rounded-lg border border-outline-variant text-xs">
          <span className="font-mono text-on-surface">Base URL: <strong className="text-primary">http://localhost:8000/api</strong></span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={handleCopyBase} className="h-7 text-xs">
              {copied ? <Check className="h-3 w-3 mr-1 text-stable-text" /> : null}
              {copied ? 'Copied' : 'Copy'}
            </Button>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 bg-primary text-on-primary px-3 py-1 rounded text-xs font-medium hover:bg-primary/90 transition-colors"
            >
              Open Swagger UI <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-on-surface-variant">Available Endpoints</h3>
          <div className="space-y-2">
            {endpoints.map((ep) => (
              <div key={ep.path} className="p-3 bg-surface rounded-lg border border-outline-variant flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-mono">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    ep.method === 'GET' ? 'bg-primary/10 text-primary' : 'bg-stable-bg text-stable-text'
                  }`}>
                    {ep.method}
                  </span>
                  <span className="font-semibold text-on-surface">{ep.path}</span>
                </div>
                <span className="text-on-surface-variant text-[11px]">{ep.desc}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <Button onClick={onClose}>Done</Button>
        </div>
      </div>
    </div>
  )
}
