import { X, Grid, Database, FileText, Bell, RefreshCw, Upload, Shield, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  isOpen: boolean
  onClose: () => void
  onOpenRegister: () => void
  onNavigate: (path: string) => void
}

export function QuickToolsModal({ isOpen, onClose, onOpenRegister, onNavigate }: Props) {
  if (!isOpen) return null

  const tools = [
    {
      name: 'Upload Live Dataset',
      desc: 'Ingest CSV dataset for real-time drift calculation',
      icon: Upload,
      action: () => { onClose(); onOpenRegister(); }
    },
    {
      name: 'Register Model',
      desc: 'Add new machine learning deployment to monitor',
      icon: Database,
      action: () => { onClose(); onOpenRegister(); }
    },
    {
      name: 'Alert Center',
      desc: 'Review critical and warning monitoring alerts',
      icon: Bell,
      action: () => { onClose(); onNavigate('/alerts'); }
    },
    {
      name: 'Generate Reports',
      desc: 'Audit drift analysis & download CSV reports',
      icon: FileText,
      action: () => { onClose(); onNavigate('/reports'); }
    },
    {
      name: 'Drift Analysis',
      desc: 'Cross-model feature PSI breakdown',
      icon: RefreshCw,
      action: () => { onClose(); onNavigate('/drift'); }
    },
    {
      name: 'System Settings',
      desc: 'Configure alert thresholds & API rules',
      icon: Shield,
      action: () => { onClose(); onNavigate('/settings'); }
    },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in">
      <div className="w-full max-w-xl rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl space-y-6">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-primary/10 text-primary rounded-lg">
              <Grid className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-title-lg font-bold text-on-surface">Quick Action Launcher</h2>
              <p className="text-body-sm text-on-surface-variant">Shortcut suite for enterprise ML workflows</p>
            </div>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface p-1 rounded-md hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {tools.map((tool) => (
            <button
              key={tool.name}
              onClick={tool.action}
              className="p-4 rounded-xl border border-outline-variant bg-surface hover:bg-surface-low hover:border-primary/40 text-left transition-all group flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-on-primary transition-colors">
                  <tool.icon className="h-4 w-4" />
                </div>
                <CheckCircle2 className="h-4 w-4 text-on-surface-variant/30 group-hover:text-primary transition-colors" />
              </div>
              <div>
                <h4 className="font-bold text-sm text-on-surface group-hover:text-primary transition-colors">{tool.name}</h4>
                <p className="text-xs text-on-surface-variant leading-normal mt-0.5">{tool.desc}</p>
              </div>
            </button>
          ))}
        </div>

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <Button variant="outline" onClick={onClose}>Close Launcher</Button>
        </div>
      </div>
    </div>
  )
}
