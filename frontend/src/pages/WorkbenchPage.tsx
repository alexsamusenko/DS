import { useEffect, useRef, useState } from 'react'
import { downloadWorkbenchFile, listWorkbenchFiles, runPipeline, uploadWorkbenchFile, type WorkbenchUpload } from '../api'

const DEMO = {
  farm_id: 'demo-farm',
  season_id: '2026',
  crop: 'wheat',
  fields: [{ field_id: 'field-1', plot_id: 1, area_ha: 10 }],
  observations: [],
  economics: { price_yield_per_centner: 1300, fertilizer_price_per_kg: 30, budget_amount: 3000 },
  response_model_evidence_status: 'unknown',
  options: { synthetic_only: true, export_requested: false },
}

type Stage = { name: string; status: string; reason?: string | null; data?: unknown; warnings?: string[] }
type PipelineResult = {
  run_id: string
  status: string
  recommendation: { status: string; reason?: string; actions?: { field_id: string; dose_kg_ha: number }[] }
  stages: Stage[]
  warnings?: string[]
  errors?: { code: string; message: string }[]
}

const explain: Record<string, string> = {
  ENGINEERED_FEATURES_MISSING: 'Не хватает рассчитанных признаков для прогноза.',
  RESPONSE_MODEL_NOT_VALIDATED: 'Модель отклика удобрений не подтверждена.',
  PREDICTION_STAGE_NOT_READY: 'Прогноз не прошёл проверку; оптимизация остановлена.',
  MISSING_VALUES_REQUIRE_GAPFILL_MATRIX: 'Для восстановления пропусков нужна пространственно-временная матрица.',
  MODEL_DID_NOT_BEAT_BASELINE: 'Прогнозная модель не превзошла базовую.',
  UPSTREAM_GATE_FAILED: 'Предыдущий этап не прошёл проверку.',
}

export default function WorkbenchPage() {
  const picker = useRef<HTMLInputElement>(null)
  const [uploads, setUploads] = useState<WorkbenchUpload[]>([])
  const [uploading, setUploading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<PipelineResult | null>(null)
  const [payload, setPayload] = useState(JSON.stringify(DEMO, null, 2))

  useEffect(() => {
    void listWorkbenchFiles().then((result) => setUploads(result.files)).catch((err: unknown) => setError(String(err)))
  }, [])

  async function addFiles(files: FileList | null) {
    if (!files?.length) return
    setUploading(true)
    setError('')
    for (const file of Array.from(files)) {
      try {
        const record = await uploadWorkbenchFile(file)
        setUploads((prev) => [...prev, record])
      } catch (err) {
        setError((prev) => prev + `${file.name}: ${String(err)}\n`)
      }
    }
    if (picker.current) picker.current.value = ''
    setUploading(false)
  }

  async function calculate() {
    setError('')
    setRunning(true)
    setResult(null)
    try {
      const request = JSON.parse(payload) as Record<string, unknown>
      setResult(await runPipeline(request) as PipelineResult)
    } catch (err) {
      setError(String(err))
    } finally {
      setRunning(false)
    }
  }

  return (
    <main className="workbench">
      <div className="card">
        <h2>Центр принятия решений АПК</h2>
        <p className="hint">Загрузите источники, проверьте их структуру и выполните демонстрационный расчёт. Никакие загруженные файлы автоматически не превращаются в проверенные агрономические факты.</p>
        <div className="stat-row">
          <div className="stat"><div className="value">{uploads.length}</div><div className="label">принято файлов</div></div>
          <div className="stat"><div className="value">{uploads.filter((f) => f.category === 'table').length}</div><div className="label">таблиц</div></div>
          <div className="stat"><div className="value">{uploads.filter((f) => f.category === 'image').length}</div><div className="label">изображений</div></div>
          <div className="stat"><div className="value">{uploads.filter((f) => f.category === 'document').length}</div><div className="label">документов</div></div>
        </div>
      </div>

      <div className="card">
        <h2>1. Исходные материалы хозяйства</h2>
        <div className="workbench-drop" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); void addFiles(event.dataTransfer.files) }}>
          <p><strong>Перетащите файлы сюда</strong> или выберите их на компьютере</p>
          <p className="hint">Excel, Word, PDF, фото рукописей, изображения, CSV, JSON, XML, GIS, телеметрия и ZIP · до 20 МиБ на файл</p>
          <input aria-label="Исходные файлы" ref={picker} type="file" multiple accept=".csv,.tsv,.xlsx,.xls,.xlsm,.ods,.json,.geojson,.xml,.pdf,.doc,.docx,.odt,.txt,.jpg,.jpeg,.png,.tif,.tiff,.heic,.webp,.zip,.kml,.gpx,.gml,.shp,.shx,.dbf,.prj,.gpkg,.las,.log" onChange={(event) => { void addFiles(event.target.files) }} />
          {uploading && <p role="status">Приём файлов…</p>}
        </div>
        {uploads.length > 0 && <div style={{ overflowX: 'auto' }}><table><thead><tr><th>Файл</th><th>Тип</th><th>Размер</th><th>Статус</th></tr></thead><tbody>{uploads.map((f) => <tr key={f.file_id}><td><strong>{f.filename}</strong> <button type="button" onClick={() => { void downloadWorkbenchFile(f).catch((err: unknown) => setError(String(err))) }}>Скачать</button><details><summary className="hint">Предпросмотр и контрольная сумма</summary><pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(f.preview, null, 2)}</pre><small>SHA-256: {f.sha256}</small></details></td><td>{f.category}</td><td>{(f.size_bytes / 1024).toFixed(1)} КБ</td><td>Зарегистрирован, не проверен</td></tr>)}</tbody></table></div>}
        <p className="hint">CSV/JSON, XLSX, DOCX, текстовые PDF и изображения: предпросмотр и экспериментальное извлечение, включая OCR фотографий. Рукописи могут распознаваться неточно. Все материалы хранятся в постоянном томе и доступны после перезапуска. Неподдерживаемые форматы сохраняются без извлечения. Загрузка не запускает обучение.</p>
      </div>

      <div className="card">
        <h2>2. Структурированный сценарий</h2>
        <p className="hint">Отредактируйте JSON с полями, наблюдениями, экономикой и параметрами моделей. Он отправляется в существующий POST /pipeline/runs. Загруженные файлы пока не подмешиваются в JSON автоматически.</p>
        <textarea aria-label="JSON сценария" spellCheck={false} value={payload} onChange={(event) => setPayload(event.target.value)} style={{ width: '100%', minHeight: 270, fontFamily: 'monospace', fontSize: 13, padding: 12, border: '1px solid var(--border)', borderRadius: 8 }} />
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
          <button className="primary" type="button" disabled={running || uploading} onClick={() => { void calculate() }}>{running ? 'Выполняется…' : 'Запустить анализ'}</button>
          <button type="button" onClick={() => { setPayload(JSON.stringify(DEMO, null, 2)); setResult(null) }}>Восстановить пример</button>
        </div>
      </div>

      {error && <div className="error-banner" role="alert" style={{ whiteSpace: 'pre-wrap' }}>{error}</div>}

      {result && <div className="card" aria-live="polite">
        <h2>3. Результат анализа</h2>
        <div className="stat-row">
          <div className="stat"><div className="value">{result.status}</div><div className="label">статус конвейера</div></div>
          <div className="stat"><div className="value">{result.recommendation.status}</div><div className="label">статус рекомендации</div></div>
          <div className="stat"><div className="value">{result.stages.filter((s) => s.status === 'ok').length}/{result.stages.length}</div><div className="label">пройдено этапов</div></div>
        </div>
        {result.recommendation.status === 'withheld' && <p className="error-banner">Решение не выдано: {explain[result.recommendation.reason ?? ''] ?? result.recommendation.reason ?? 'не выполнены необходимые условия'}. Нельзя использовать такой запуск для назначения доз.</p>}
        {!!result.recommendation.actions?.length && <><h3>Рекомендуемые действия</h3><table><thead><tr><th>Поле</th><th>Доза, кг/га</th></tr></thead><tbody>{result.recommendation.actions.map((a) => <tr key={a.field_id}><td>{a.field_id}</td><td>{a.dose_kg_ha.toFixed(2)}</td></tr>)}</tbody></table><p className="hint">В синтетическом режиме рекомендации имеют только исследовательский характер.</p></>}
        <h3>Трассировка этапов</h3>
        <div style={{ overflowX: 'auto' }}><table><thead><tr><th>Этап</th><th>Статус</th><th>Причина</th></tr></thead><tbody>{result.stages.map((stage, index) => <tr key={index}><td>{stage.name}</td><td>{stage.status}</td><td>{explain[stage.reason ?? ''] ?? stage.reason ?? '—'}{stage.warnings?.length ? <details><summary>Предупреждения</summary>{stage.warnings.join('; ')}</details> : null}</td></tr>)}</tbody></table></div>
        {!!result.errors?.length && <pre>{JSON.stringify(result.errors, null, 2)}</pre>}
        <details><summary>Полный JSON-ответ</summary><pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(result, null, 2)}</pre></details>
      </div>}
    </main>
  )
}
