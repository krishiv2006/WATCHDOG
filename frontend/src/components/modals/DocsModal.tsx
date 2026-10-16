import { X, BookOpen, Code, Cpu, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export function DocsModal({ isOpen, onClose }: Props) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in">
      <div className="w-full max-w-2xl rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl space-y-6 max-h-[85vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-primary/10 text-primary rounded-lg">
              <BookOpen className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-title-lg font-bold text-on-surface">Watchdog AI Documentation</h2>
              <p className="text-body-sm text-on-surface-variant">Enterprise Machine Learning Monitoring Architecture & Algorithms</p>
            </div>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface p-1 rounded-md hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-6 text-sm text-on-surface">
          <div className="p-4 bg-surface rounded-lg border border-outline-variant space-y-2">
            <h3 className="font-semibold text-primary flex items-center gap-2">
              <Activity className="h-4 w-4" /> Population Stability Index (PSI)
            </h3>
            <p className="text-on-surface-variant text-xs leading-relaxed">
              PSI measures how much a variable has shifted in distribution over time between a baseline dataset and live production predictions.
            </p>
            <div className="grid grid-cols-3 gap-2 text-xs font-mono pt-2">
              <div className="bg-stable-bg/40 text-stable-text p-2 rounded text-center">PSI &lt; 0.1: Stable</div>
              <div className="bg-warning-bg/40 text-warning-text p-2 rounded text-center">0.1 - 0.25: Warning</div>
              <div className="bg-critical-bg/40 text-critical-text p-2 rounded text-center">PSI &gt; 0.25: Critical</div>
            </div>
          </div>

          <div className="p-4 bg-surface rounded-lg border border-outline-variant space-y-2">
            <h3 className="font-semibold text-primary flex items-center gap-2">
              <Cpu className="h-4 w-4" /> Jensen-Shannon (JS) Distance
            </h3>
            <p className="text-on-surface-variant text-xs leading-relaxed">
              Calculates a symmetric probability distribution distance between 0 and 1. Useful for detection of multimodal feature drift.
            </p>
          </div>

          <div className="p-4 bg-surface rounded-lg border border-outline-variant space-y-2">
            <h3 className="font-semibold text-primary flex items-center gap-2">
              <Code className="h-4 w-4" />               Training Baseline and Live Prediction APIs
            </h3>
            <pre className="bg-background text-on-surface p-3 rounded text-xs overflow-x-auto border border-outline-variant font-mono">
{`curl -X POST "http://localhost:8000/api/models/{id}/baseline" \\
  -F "file=@training_data.csv"

curl -X POST "http://localhost:8000/api/models/{id}/predictions" \\
  -H "Content-Type: application/json" \\
  -d '{"predictionId":"pred-1","features":{"age":35},"prediction":1}'`}
            </pre>
          </div>
        </div>

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <Button onClick={onClose}>Close Documentation</Button>
        </div>
      </div>
    </div>
  )
}
