import { Component, type ErrorInfo, type ReactNode } from 'react'

// Раньше в приложении не было ни одного Error Boundary: ошибка рендера в
// любой из вкладок (например, .toFixed() на undefined из неожиданного ответа
// API) схлопывала всё приложение в белый экран без обратной связи
// пользователю (см. аудит перед развёртыванием). Ловит только ошибки
// рендера/жизненного цикла React -- сетевые ошибки API уже обрабатываются
// на уровне страниц через loading/error-состояния (api.ts, PredictPage.tsx
// и т.п.), это не дублирование, а другой уровень защиты.
interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Необработанная ошибка рендера:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="error-banner" style={{ margin: '2rem' }}>
          <p>Что-то пошло не так при отображении страницы.</p>
          <p className="muted">{this.state.error.message}</p>
          <button onClick={() => this.setState({ error: null })}>Попробовать снова</button>
        </div>
      )
    }
    return this.props.children
  }
}
