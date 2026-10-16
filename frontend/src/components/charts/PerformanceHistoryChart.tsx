import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export interface PerformanceHistoryPoint {
  time: string
  accuracy: number | null | undefined
  base: number | null | undefined
}

interface Props {
  history: PerformanceHistoryPoint[]
  modelId: string
}

export function PerformanceHistoryChart({ history, modelId }: Props) {
  const hasData = history.some((point) => point.accuracy !== null && point.accuracy !== undefined)
  const chartData = history.filter((point) => point.accuracy !== null && point.accuracy !== undefined)

  return (
    <div className="h-72 min-h-72 w-full min-w-0">
      {hasData ? (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E0E3E5" />
            <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: '#40484C', fontSize: 12 }} />
            <YAxis domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{ fill: '#40484C', fontSize: 12 }} />
            <Tooltip contentStyle={{ backgroundColor: '#191C1D', color: '#EFF1F2', borderRadius: '8px' }} />
            <Line type="monotone" dataKey="accuracy" name="Current Accuracy" stroke="#B3261E" strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="base" name="Baseline Accuracy" stroke="#70787D" strokeDasharray="5 5" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex h-full items-center justify-center px-6 text-center text-sm text-on-surface-variant">
          No labelled data yet. Send labels via POST /api/models/{modelId}/ground-truth to see accuracy over time.
        </div>
      )}
    </div>
  )
}
