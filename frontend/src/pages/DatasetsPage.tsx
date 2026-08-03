import { useEffect, useRef, useState } from 'react'
import { type DatasetEntry, listDatasets, uploadDataset } from '../api'

function formatBytes(bytes?: number): string {
  if (bytes === undefined) return '—'
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} КБ`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} МБ`
  return `${(bytes / 1024 ** 3).toFixed(2)} ГБ`
}

function DetailRow({ label, text }: { label: string; text: string | null }) {
  if (!text) return null
  return (
    <div style={{ marginBottom: '0.5rem' }}>
      <strong style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>{label}</strong>
      <p className="hint" style={{ margin: '0.15rem 0 0', whiteSpace: 'pre-wrap' }}>
        {text}
      </p>
    </div>
  )
}

function DatasetRow({ d, expanded, onToggle }: { d: DatasetEntry; expanded: boolean; onToggle: () => void }) {
  return (
    <>
      <tr onClick={onToggle} style={{ cursor: 'pointer' }}>
        <td>
          {expanded ? '▾' : '▸'} {d.title}
        </td>
        <td className="hint">{d.license_summary.slice(0, 90)}{d.license_summary.length > 90 ? '…' : ''}</td>
        <td>
          <span className={`badge ${d.on_disk ? 'ok' : 'missing'}`}>{d.on_disk ? 'есть' : 'нет'}</span>
        </td>
        <td>{d.file_count ?? '—'}</td>
        <td>{formatBytes(d.total_size_bytes)}</td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={5} style={{ background: '#fafbf9' }}>
            <DetailRow label="Источник и мотивация" text={d.source} />
            <DetailRow label="Состав" text={d.composition} />
            <DetailRow label="Полный текст лицензии" text={d.license_summary} />
            <DetailRow label="Использование в проекте" text={d.usage_in_project} />
            <DetailRow label="Известные ограничения" text={d.limitations} />
            {!d.source && (
              <p className="hint">
                Карточка не найдена в <code>docs/dataset_cards/</code> -- заведите её по шаблону{' '}
                <code>docs/dataset_cards/TEMPLATE.md</code> перед использованием в продовом обучении (см.{' '}
                <code>docs/governance/best_practices.md</code>).
              </p>
            )}
          </td>
        </tr>
      )}
    </>
  )
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [name, setName] = useState('')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const fileRef = useRef<HTMLInputElement>(null)

  function refresh() {
    listDatasets().then(setDatasets).catch((e) => setError(String(e.message ?? e)))
  }

  useEffect(refresh, [])

  function toggle(slug: string) {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(slug)) next.delete(slug)
      else next.add(slug)
      return next
    })
  }

  async function onUpload(ev: React.FormEvent) {
    ev.preventDefault()
    const file = fileRef.current?.files?.[0]
    if (!file || !name) {
      setError('Укажите имя датасета и выберите .zip файл')
      return
    }
    setUploading(true)
    setError(null)
    setSuccess(null)
    try {
      const entry = await uploadDataset(name, file)
      setSuccess(`Загружено: data/${entry.slug} (${entry.file_count ?? '?'} файлов)`)
      setName('')
      if (fileRef.current) fileRef.current.value = ''
      refresh()
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setUploading(false)
    }
  }

  const onDiskCount = datasets?.filter((d) => d.on_disk).length ?? 0
  const totalBytes = datasets?.reduce((sum, d) => sum + (d.total_size_bytes ?? 0), 0) ?? 0

  return (
    <>
      <div className="card">
        <h2>Каталог датасетов</h2>
        <p className="hint">
          Сопоставление карточек датасетов (<code>docs/dataset_cards/</code>) с тем, что реально лежит в{' '}
          <code>data/</code>. Строка кликабельна -- разворачивает полную карточку (источник, состав, ограничения).
          Статус лицензии — см. <code>docs/governance/licenses.md</code>.
        </p>
        {error && <div className="error-banner">{error}</div>}
        {!datasets && <p className="muted">Загрузка...</p>}
        {datasets && (
          <>
            <div className="stat-row">
              <div className="stat">
                <div className="value">{datasets.length}</div>
                <div className="label">датасетов в каталоге</div>
              </div>
              <div className="stat">
                <div className="value">{onDiskCount}</div>
                <div className="label">загружено на диск</div>
              </div>
              <div className="stat">
                <div className="value">{formatBytes(totalBytes)}</div>
                <div className="label">суммарный размер</div>
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Датасет</th>
                  <th>Лицензия</th>
                  <th>На диске</th>
                  <th>Файлов</th>
                  <th>Размер</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <DatasetRow key={d.slug} d={d} expanded={expanded.has(d.slug)} onToggle={() => toggle(d.slug)} />
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <div className="card">
        <h2>Загрузить новый датасет</h2>
        <p className="hint">
          Архив (.zip) распаковывается в <code>data/&lt;имя&gt;/</code>. Карточку датасета (лицензия, источник) после этого
          нужно завести вручную по шаблону <code>docs/dataset_cards/TEMPLATE.md</code> -- загрузка не подтверждает лицензию
          автоматически.
        </p>
        {success && <div className="success-banner">{success}</div>}
        <form onSubmit={onUpload} style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <label style={{ flex: '0 0 220px' }}>
            Имя (латиница/цифры/-/_)
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="my_dataset" />
          </label>
          <label>
            Архив
            <input type="file" accept=".zip" ref={fileRef} />
          </label>
          <button className="primary" type="submit" disabled={uploading}>
            {uploading ? 'Загружаю...' : 'Загрузить'}
          </button>
        </form>
      </div>
    </>
  )
}
