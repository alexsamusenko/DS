import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from 'recharts'
import { predict, trainingSummary, type PredictionFeatures, type PredictionResponse, type TrainingSummary } from '../api'

const FIELD_META: { key: keyof PredictionFeatures; label: string; modality: string; def: number }[] = [
  { key: 'soil_moisture_mean', label: 'Влажность почвы, %', modality: 'num', def: 24 },
  { key: 'soil_nitrogen_mean', label: 'Содержание азота', modality: 'num', def: 18 },
  { key: 'precip_sum', label: 'Сумма осадков, мм', modality: 'num', def: 320 },
  { key: 'temp_sum', label: 'Сумма эфф. температур', modality: 'num', def: 1800 },
  { key: 'ndvi_peak', label: 'Пиковый NDVI', modality: 'geo', def: 0.72 },
  { key: 'ndvi_integral', label: 'Интеграл NDVI за сезон', modality: 'geo', def: 45 },
  { key: 'disease_events_count', label: 'Случаев поражения (по фото)', modality: 'img', def: 1 },
  { key: 'max_risk_stage', label: 'Макс. стадия риска', modality: 'img', def: 2 },
  { key: 'fertilizer_applications_count', label: 'Внесений удобрений', modality: 'text', def: 3 },
  { key: 'total_dose', label: 'Суммарная доза, кг', modality: 'text', def: 140 },
]

const MODALITY_LABEL: Record<string, string> = { num: 'числовые', geo: 'гео (NDVI)', img: 'фото', text: 'текст (агроприёмы)' }

const PRESETS: { name: string; hint: string; values: Partial<PredictionFeatures> }[] = [
  { name: 'Типичное поле', hint: 'значения по умолчанию', values: {} },
  {
    name: 'Здоровое, ухоженное',
    hint: 'высокий NDVI, нет поражений, вовремя внесены удобрения',
    values: { ndvi_peak: 0.85, ndvi_integral: 58, disease_events_count: 0, max_risk_stage: 0, fertilizer_applications_count: 4, total_dose: 180 },
  },
  {
    name: 'Поражённое болезнью',
    hint: 'много случаев поражения по фото, высокая стадия риска',
    values: { ndvi_peak: 0.55, ndvi_integral: 28, disease_events_count: 4, max_risk_stage: 4 },
  },
  {
    name: 'Засуха, мало удобрений',
    hint: 'низкая влажность/осадки, минимум внесений',
    values: { soil_moisture_mean: 10, precip_sum: 120, fertilizer_applications_count: 1, total_dose: 40, ndvi_peak: 0.45, ndvi_integral: 20 },
  },
]

function defaultFeatures(): PredictionFeatures {
  return Object.fromEntries(FIELD_META.map((f) => [f.key, f.def])) as unknown as PredictionFeatures
}

function importanceChartData(importance: Record<string, number>) {
  return Object.entries(importance).map(([modality, value]) => ({ modality: MODALITY_LABEL[modality] ?? modality, value }))
}

function topModality(importance: Record<string, number>): [string, number] {
  const entries = Object.entries(importance)
  const total = entries.reduce((s, [, v]) => s + v, 0)
  const [modality, value] = entries.reduce((best, e) => (e[1] > best[1] ? e : best))
  return [MODALITY_LABEL[modality] ?? modality, total > 0 ? (value / total) * 100 : 0]
}

interface HistoryItem {
  id: number
  label: string
  predicted_yield: number
}

export default function PredictPage() {
  const [features, setFeatures] = useState<PredictionFeatures>(defaultFeatures())
  const [activePreset, setActivePreset] = useState('Типичное поле')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [summary, setSummary] = useState<TrainingSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])

  useEffect(() => {
    trainingSummary().then(setSummary).catch((e) => setError(String(e.message ?? e)))
  }, [])

  function applyPreset(preset: (typeof PRESETS)[number]) {
    setActivePreset(preset.name)
    setFeatures({ ...defaultFeatures(), ...preset.values })
    setResult(null)
  }

  async function onSubmit(ev: React.FormEvent) {
    ev.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const r = await predict(features)
      setResult(r)
      setHistory((prev) => [{ id: Date.now(), label: activePreset, predicted_yield: r.predicted_yield }, ...prev].slice(0, 8))
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="card">
        <h2>Прогноз урожайности по точке (§2.3, L4)</h2>
        <p className="hint">
          Модель обучается один раз лениво на демо-данных (§2.3.5) и кешируется в процессе сервиса -- прогноз ниже проверяет
          готовый пайплайн предсказания + SHAP-объяснение по модальностям, а не служит источником продовых данных хозяйства.
        </p>
        {error && <div className="error-banner">{error}</div>}
        <div className="stat-row" style={{ marginBottom: '0.5rem' }}>
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              className="primary"
              style={{
                background: activePreset === p.name ? undefined : 'transparent',
                color: activePreset === p.name ? undefined : 'var(--accent-dark)',
                border: activePreset === p.name ? undefined : '1px solid var(--border)',
              }}
              onClick={() => applyPreset(p)}
              title={p.hint}
            >
              {p.name}
            </button>
          ))}
        </div>
        <form className="field-grid" onSubmit={onSubmit}>
          {FIELD_META.map((f) => (
            <label key={f.key}>
              {f.label} <span className="muted">({MODALITY_LABEL[f.modality]})</span>
              <input
                type="number"
                step="any"
                value={features[f.key]}
                onChange={(e) => setFeatures({ ...features, [f.key]: Number(e.target.value) })}
              />
            </label>
          ))}
        </form>
        <div style={{ marginTop: '1rem' }}>
          <button className="primary" onClick={onSubmit} disabled={loading}>
            {loading ? 'Считаю...' : 'Спрогнозировать'}
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>Результат</h2>
          <div className="stat-row">
            <div className="stat">
              <div className="value">{result.predicted_yield.toFixed(1)} ц/га</div>
              <div className="label">прогноз урожайности</div>
            </div>
          </div>
          <p className="hint">
            Сильнее всего на прогноз повлияла модальность «{topModality(result.modality_importance)[0]}» —{' '}
            {topModality(result.modality_importance)[1].toFixed(0)}% суммарного |SHAP| по этой точке.
          </p>
          <h3>Вклад модальностей (Σ|SHAP|, §2.3.4)</h3>
          <BarChart width={520} height={220} data={importanceChartData(result.modality_importance)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
            <XAxis dataKey="modality" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#3f7d3f" />
          </BarChart>
        </div>
      )}

      {history.length > 1 && (
        <div className="card">
          <h2>Сравнение сценариев (эта сессия)</h2>
          <table>
            <thead>
              <tr>
                <th>Сценарий</th>
                <th>Прогноз, ц/га</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.id}>
                  <td>{h.label}</td>
                  <td>{h.predicted_yield.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {summary && (
        <div className="card">
          <h2>Справка: важность модальностей на демо-датасете (§2.3.5-2.3.6)</h2>
          <p className="hint">
            RMSE по GroupKFold (группировка по полю, чтобы избежать пространственной утечки) при исключении каждой модальности
            по очереди -- чем сильнее рост RMSE без модальности, тем она важнее для качества прогноза.
          </p>
          <BarChart
            width={640}
            height={220}
            data={Object.entries(summary.rmse_without_modality).map(([modality, rmse]) => ({
              modality: MODALITY_LABEL[modality] ?? modality,
              rmse,
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
            <XAxis dataKey="modality" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="rmse" fill="#b3432b" name="RMSE без модальности" />
          </BarChart>
          <p className="hint">RMSE на всех модальностях: {summary.rmse_all_modalities.toFixed(2)} ц/га</p>
        </div>
      )}
    </>
  )
}
