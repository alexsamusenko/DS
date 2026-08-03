import { useState } from 'react'
import TrainingPage from './pages/TrainingPage'
import PredictPage from './pages/PredictPage'
import DatasetsPage from './pages/DatasetsPage'

const TABS = [
  { id: 'training', label: 'Обучение', Component: TrainingPage },
  { id: 'predict', label: 'Тест на точке', Component: PredictPage },
  { id: 'datasets', label: 'Датасеты', Component: DatasetsPage },
] as const

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('training')
  const Active = TABS.find((t) => t.id === tab)!.Component

  return (
    <>
      <header className="app-header">
        <h1>DS -- практическая часть диссертации</h1>
        <p>Статистика обучения, ручная проверка прогноза на точке, каталог датасетов (L1-L4)</p>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={t.id === tab ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      <Active />
    </>
  )
}
