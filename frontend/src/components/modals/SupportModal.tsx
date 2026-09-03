import { X, HelpCircle, LifeBuoy, MessageSquare, Mail, FileQuestion } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export function SupportModal({ isOpen, onClose }: Props) {
  if (!isOpen) return null

  const faqs = [
    {
      q: 'How does Watchdog AI calculate Model Health?',
      a: 'Health score (0-100) is a weighted composite of average PSI feature drift (40%), model accuracy degradation (40%), and open unresolved critical alerts (20%).'
    },
    {
      q: 'How can I connect a live production stream?',
      a: 'Send prediction payload events or CSV files to /api/models/ingest-dataset. The system processes distributions and updates metrics in real time.'
    },
    {
      q: 'What triggers automated alert notifications?',
      a: 'Alerts are raised automatically whenever Population Stability Index (PSI) exceeds warning (0.1) or critical (0.25) thresholds.'
    }
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in">
      <div className="w-full max-w-2xl rounded-xl border border-outline-variant bg-surface-lowest p-6 shadow-2xl space-y-6 max-h-[85vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-outline-variant pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-primary/10 text-primary rounded-lg">
              <LifeBuoy className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-title-lg font-bold text-on-surface">Help & Support Center</h2>
              <p className="text-body-sm text-on-surface-variant">24/7 Enterprise ML Monitoring Assistance & FAQs</p>
            </div>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface p-1 rounded-md hover:bg-surface">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-surface rounded-xl border border-outline-variant flex items-center gap-3">
            <div className="p-2.5 bg-primary/10 text-primary rounded-lg">
              <MessageSquare className="h-5 w-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-on-surface">Live Chat Support</h4>
              <p className="text-xs text-on-surface-variant">Available for Enterprise users</p>
            </div>
          </div>

          <div className="p-4 bg-surface rounded-xl border border-outline-variant flex items-center gap-3">
            <div className="p-2.5 bg-primary/10 text-primary rounded-lg">
              <Mail className="h-5 w-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-on-surface">Email Support</h4>
              <p className="text-xs text-on-surface-variant">support@watchdog-ai.io</p>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-on-surface-variant flex items-center gap-1.5">
            <FileQuestion className="h-4 w-4" /> Frequently Asked Questions
          </h3>
          <div className="space-y-3">
            {faqs.map((faq, i) => (
              <div key={i} className="p-3.5 bg-surface rounded-lg border border-outline-variant space-y-1">
                <h4 className="font-semibold text-sm text-on-surface">{faq.q}</h4>
                <p className="text-xs text-on-surface-variant leading-relaxed">{faq.a}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <Button onClick={onClose}>Close Help Center</Button>
        </div>
      </div>
    </div>
  )
}
