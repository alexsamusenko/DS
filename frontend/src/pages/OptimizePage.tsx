import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from 'recharts'
import { exportIsoxml, optimize, type OptimizeResponse, type Plot } from '../api'

function defaultPlots(): Plot[] {
  return [
    { plot_id: 0, baseline: 38, R: 9, s: 55, area: 1.2 },
    { plot_id: 1, baseline: 41, R: 4, s: 65, area: 1.0 },
    { plot_id: 2, baseline: 35, R: 11, s: 50, area: 1.5 },
    { plot_id: 3, baseline: 42, R: 3, s: 70, area: 0.9 },
  ]
}

export default function OptimizePage() {
  const [plots, setPlots] = useState<Plot[]>(defaultPlots())
  const [priceYield, setPriceYield] = useState(1300)
  const [priceFert, setPriceFert] = useState(50)
  const [budget, setBudget] = useState<number | ''>('')
  const [result, setResult] = useState<OptimizeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [customerName, setCustomerName] = useState('Демо-хозяйство')
  const [farmName, setFarmName] = useState('Демо-поле')

  function updatePlot(index: number, field: keyof Plot, value: number) {
    setPlots((prev) => prev.map((p, i) => (i === index ? { ...p, [field]: value } : p)))
  }

  function addPlot() {
    const nextId = Math.max(-1, ...plots.map((p) => p.plot_id)) + 1
    setPlots((prev) => [...prev, { plot_id: nextId, baseline: 40, R: 6, s: 60, area: 1.0 }])
  }

  function removePlot(index: number) {
    setPlots((prev) => prev.filter((_, i) => i !== index))
  }

  function buildRequest() {
    return {
      plots,
      price_yield: priceYield,
      price_fert: priceFert,
      dose_min: 0,
      dose_max: 150,
      budget: budget === '' ? null : budget,
    }
  }

  async function onOptimize() {
    setLoading(true)
    setError(null)
    try {
      setResult(await optimize(buildRequest()))
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setLoading(false)
    }
  }

  async function onExport() {
    setExporting(true)
    setError(null)
    try {
      const blob = await exportIsoxml({ ...buildRequest(), customer_name: customerName, farm_name: farmName, task_designator: 'Карта-задание' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'TASKDATA.xml'
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setExporting(false)
    }
  }

  return (
    <>
      <div className="card">
        <h2>Дифференцированное внесение удобрений (§2.4, L5)</h2>
        <p className="hint">
          Оптимальная доза по каждому участку максимизирует прибыль (цена продукции минус стоимость удобрения) с учётом
          кривой отклика Митчерлиха. Без бюджета — независимый оптимум по каждому участку; с бюджетом — метод Лагранжа,
          результат сравнивается с равномерным внесением того же суммарного количества.
        </p>
        {error && <div className="error-banner">{error}</div>}

        <table>
          <thead>
            <tr>
              <th>Участок</th>
              <th>Baseline, ц/га</th>
              <th>R (отклик), ц/га</th>
              <th>s (насыщение), кг/га</th>
              <th>Площадь, га</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {plots.map((p, i) => (
              <tr key={p.plot_id}>
                <td>{p.plot_id}</td>
                <td>
                  <input type="number" step="any" value={p.baseline} onChange={(e) => updatePlot(i, 'baseline', Number(e.target.value))} />
                </td>
                <td>
                  <input type="number" step="any" value={p.R} onChange={(e) => updatePlot(i, 'R', Number(e.target.value))} />
                </td>
                <td>
                  <input type="number" step="any" value={p.s} onChange={(e) => updatePlot(i, 's', Number(e.target.value))} />
                </td>
                <td>
                  <input type="number" step="any" value={p.area} onChange={(e) => updatePlot(i, 'area', Number(e.target.value))} />
                </td>
                <td>
                  <button type="button" className="primary" style={{ background: 'transparent', color: 'var(--danger)', border: 'none', padding: '0.2rem 0.5rem' }} onClick={() => removePlot(i)}>
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button type="button" className="primary" style={{ background: 'transparent', color: 'var(--accent-dark)', border: '1px solid var(--border)', marginTop: '0.5rem' }} onClick={addPlot}>
          + участок
        </button>

        <form className="field-grid" style={{ marginTop: '1rem' }}>
          <label>
            Цена продукции, руб./ц
            <input type="number" step="any" value={priceYield} onChange={(e) => setPriceYield(Number(e.target.value))} />
          </label>
          <label>
            Цена удобрения, руб./кг
            <input type="number" step="any" value={priceFert} onChange={(e) => setPriceFert(Number(e.target.value))} />
          </label>
          <label>
            Бюджет удобрений, кг (пусто = без ограничения)
            <input type="number" step="any" value={budget} onChange={(e) => setBudget(e.target.value === '' ? '' : Number(e.target.value))} />
          </label>
        </form>

        <div style={{ marginTop: '1rem' }}>
          <button className="primary" onClick={onOptimize} disabled={loading || plots.length === 0}>
            {loading ? 'Считаю...' : 'Оптимизировать'}
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>Результат</h2>
          <div className="stat-row">
            <div className="stat">
              <div className="value">{result.total_profit.toFixed(0)}</div>
              <div className="label">суммарная прибыль, руб.</div>
            </div>
            <div className="stat">
              <div className="value">{result.total_profit_uniform.toFixed(0)}</div>
              <div className="label">прибыль при равномерном внесении</div>
            </div>
            <div className="stat">
              <div className="value">+{result.profit_gain_percent.toFixed(1)}%</div>
              <div className="label">прирост дифференцированного сценария</div>
            </div>
          </div>
          <h3>Дозы по участкам, кг/га</h3>
          <BarChart width={520} height={220} data={plots.map((p, i) => ({ plot: `№${p.plot_id}`, dose: result.doses[i] }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
            <XAxis dataKey="plot" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="dose" fill="#3f7d3f" />
          </BarChart>

          <h3 style={{ marginTop: '1.25rem' }}>Экспорт для внешних систем (§2.6, L7)</h3>
          <p className="hint">
            Карта-задание в формате ISO 11783-10 (Task Data) -- понимают бортовые терминалы техники и FMIS независимо от
            производителя. Геометрия участков в файле синтетическая (в системе нет реальных границ полей хозяйства), код
            DDI для дозы внесения требует сверки с официальным реестром AEF ISOBUS перед использованием с реальной техникой
            -- см. <code>docs/governance/licenses.md</code>.
          </p>
          <div className="field-grid" style={{ maxWidth: 480 }}>
            <label>
              Хозяйство
              <input type="text" value={customerName} onChange={(e) => setCustomerName(e.target.value)} />
            </label>
            <label>
              Ферма/поле
              <input type="text" value={farmName} onChange={(e) => setFarmName(e.target.value)} />
            </label>
          </div>
          <div style={{ marginTop: '0.75rem' }}>
            <button className="primary" onClick={onExport} disabled={exporting}>
              {exporting ? 'Формирую...' : 'Скачать TASKDATA.xml'}
            </button>
          </div>
        </div>
      )}
    </>
  )
}
