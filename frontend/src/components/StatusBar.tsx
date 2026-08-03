import { useEffect, useState } from 'react'
import { listDatasets, trainingHistory } from '../api'

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

export default function StatusBar() {
  const [healthy, setHealthy] = useState<boolean | null>(null)
  const [datasetCount, setDatasetCount] = useState<number | null>(null)
  const [trainedCount, setTrainedCount] = useState<number | null>(null)

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => setHealthy(r.ok))
      .catch(() => setHealthy(false))
    listDatasets()
      .then((d) => setDatasetCount(d.filter((x) => x.on_disk).length))
      .catch(() => setDatasetCount(null))
    trainingHistory()
      .then((h) => setTrainedCount([h.ner, h.leaf_classifier].filter(Boolean).length))
      .catch(() => setTrainedCount(null))
  }, [])

  return (
    <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center', fontSize: '0.8rem', color: 'var(--muted)' }}>
      <span>
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: healthy === null ? '#c9cdc4' : healthy ? 'var(--ok)' : 'var(--danger)',
            marginRight: 6,
          }}
        />
        API {healthy === null ? '...' : healthy ? 'доступен' : 'недоступен'}
      </span>
      {datasetCount !== null && <span>датасетов на диске: {datasetCount}</span>}
      {trainedCount !== null && <span>обучено моделей: {trainedCount}/2</span>}
    </div>
  )
}
