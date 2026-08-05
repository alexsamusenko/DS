import { lazy, Suspense, useState } from 'react'
import StatusBar from './components/StatusBar'

// Разбито на отдельные чанки (lazy + code-splitting) -- recharts тянется
// только той вкладкой, что его использует, а не в общий бандл при первой
// загрузке страницы (см. предупреждение vite о чанке >500 КБ).
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const PredictPage = lazy(() => import('./pages/PredictPage'))
const OptimizePage = lazy(() => import('./pages/OptimizePage'))
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'))

const TABS = [
  { id: 'training', label: 'Обучение', Component: TrainingPage },
  { id: 'predict', label: 'Тест на точке', Component: PredictPage },
  { id: 'optimize', label: 'Внесение удобрений', Component: OptimizePage },
  { id: 'datasets', label: 'Датасеты', Component: DatasetsPage },
] as const

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('training')
  const Active = TABS.find((t) => t.id === tab)!.Component

  return (
    <>
      <header className="app-header">
        <h1>DS -- практическая часть диссертации</h1>
        <p>Статистика обучения, ручная проверка прогноза на точке, внесение удобрений с экспортом для внешних систем, каталог датасетов (L1-L7)</p>
        <div style={{ marginTop: '0.6rem' }}>
          <StatusBar />
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={t.id === tab ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      <Suspense fallback={<p className="muted">Загрузка...</p>}>
        <Active />
      </Suspense>
    </>
  )
}
