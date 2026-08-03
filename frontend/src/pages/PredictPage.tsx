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

function defaultFeatures(): PredictionFeatures {
  return Object.fromEntries(FIELD_META.map((f) => [f.key, f.def])) as unknown as PredictionFeatures
}

function importanceChartData(importance: Record<string, number>) {
  return Object.entries(importance).map(([modality, value]) => ({ modality: MODALITY_LABEL[modality] ?? modality, value }))
}

export default function PredictPage() {
  const [features, setFeatures] = useState<PredictionFeatures>(defaultFeatures())
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [summary, setSummary] = useState<TrainingSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    trainingSummary().then(setSummary).catch((e) => setError(String(e.message ?? e)))
  }, [])

  async function onSubmit(ev: React.FormEvent) {
    ev.preventDefault()
    setLoading(true)
    setError(null)
    try {
      setResult(await predict(features))
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
