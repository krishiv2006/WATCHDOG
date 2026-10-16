import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { PerformanceHistoryChart } from './PerformanceHistoryChart'

describe('PerformanceHistoryChart', () => {
  it('shows the labelled-data empty state without rendering a chart for all-null accuracy', () => {
    const markup = renderToStaticMarkup(
      <PerformanceHistoryChart
        modelId="model-1"
        history={[
          { time: 'Oct 16', accuracy: null, base: null },
          { time: 'Oct 17', accuracy: undefined, base: null },
        ]}
      />,
    )

    expect(markup).toContain('No labelled data yet. Send labels via POST /api/models/model-1/ground-truth to see accuracy over time.')
    expect(markup).not.toContain('<svg')
  })
})
