import { useState, useEffect } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Settings, Shield, Bell, Check, Save, Key, Sliders } from 'lucide-react'

export default function SettingsPage() {
  const [rules, setRules] = useState<any[]>([])
  const [saved, setSaved] = useState(false)
  const [apiKey, setApiKey] = useState('wd_live_9482049182390148')

  useEffect(() => {
    api.getAlertRules().then(setRules)
  }, [])

  const handleUpdateRule = async (id: string, warning: number, critical: number, enabled: boolean) => {
    await api.updateAlertRule(id, { thresholdWarning: warning, thresholdCritical: critical, enabled })
    setRules(prev => prev.map(r => r.id === id ? { ...r, thresholdWarning: warning, thresholdCritical: critical, enabled } : r))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-6 gap-4">
        <div>
          <h1 className="text-[32px] font-semibold leading-[40px] tracking-[-0.02em] text-on-surface mb-2">
            Monitoring & Alert Settings
          </h1>
          <p className="text-[16px] leading-[24px] text-on-surface-variant">
            Configure Population Stability Index thresholds, evaluation schedules, and API access keys.
          </p>
        </div>
        {saved && (
          <div className="bg-stable-bg/50 text-stable-text px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 animate-in fade-in">
            <Check className="h-4 w-4" /> Settings Saved!
          </div>
        )}
      </div>

      {/* Rules Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sliders className="h-5 w-5 text-primary" />
            <div>
              <CardTitle>Threshold Rules Configuration</CardTitle>
              <CardDescription>System-wide Population Stability Index (PSI) & Latency threshold rules</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead>Warning Threshold</TableHead>
                <TableHead>Critical Threshold</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rules.map((rule) => (
                <TableRow key={rule.id}>
                  <TableCell className="font-semibold text-on-surface uppercase text-xs">{rule.metric}</TableCell>
                  <TableCell>
                    <input
                      type="number"
                      step="0.01"
                      defaultValue={rule.thresholdWarning}
                      className="w-24 h-8 px-2 rounded border border-outline-variant bg-surface text-xs focus:outline-none focus:border-primary"
                      id={`warn-${rule.id}`}
                    />
                  </TableCell>
                  <TableCell>
                    <input
                      type="number"
                      step="0.01"
                      defaultValue={rule.thresholdCritical}
                      className="w-24 h-8 px-2 rounded border border-outline-variant bg-surface text-xs focus:outline-none focus:border-primary"
                      id={`crit-${rule.id}`}
                    />
                  </TableCell>
                  <TableCell>
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                      rule.enabled ? 'bg-stable-bg/50 text-stable-text' : 'bg-surface text-on-surface-variant'
                    }`}>
                      {rule.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const warnVal = parseFloat((document.getElementById(`warn-${rule.id}`) as HTMLInputElement).value)
                        const critVal = parseFloat((document.getElementById(`crit-${rule.id}`) as HTMLInputElement).value)
                        handleUpdateRule(rule.id, warnVal, critVal, rule.enabled)
                      }}
                    >
                      <Save className="h-3 w-3 mr-1" /> Save
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* API Key Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Key className="h-5 w-5 text-primary" />
            <div>
              <CardTitle>API Ingestion Security</CardTitle>
              <CardDescription>Authentication key for submitting live data to the monitoring pipeline</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-on-surface-variant mb-1">Production API Key</label>
            <div className="flex gap-2">
              <input
                type="text"
                readOnly
                value={apiKey}
                className="flex-1 h-10 px-3 rounded-lg border border-outline-variant bg-surface text-xs font-mono text-on-surface"
              />
              <Button variant="outline" onClick={() => {
                setApiKey(`wd_live_${Math.random().toString(36).substring(2, 18)}`)
                setSaved(true)
                setTimeout(() => setSaved(false), 2000)
              }}>
                Rotate Key
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
