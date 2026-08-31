import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ErrorBoundary from './ErrorBoundary'

function Boom(): never {
  throw new Error('тестовый сбой рендера')
}

describe('ErrorBoundary', () => {
  it('рендерит детей, если ошибок нет', () => {
    render(
      <ErrorBoundary>
        <p>всё хорошо</p>
      </ErrorBoundary>,
    )
    expect(screen.getByText('всё хорошо')).toBeInTheDocument()
  })

  it('ловит ошибку рендера и показывает запасной интерфейс вместо белого экрана', () => {
    // React логирует ошибку в консоль даже при пойманном Error Boundary -- глушим это в тесте.
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    )
    expect(screen.getByText(/что-то пошло не так/i)).toBeInTheDocument()
    expect(screen.getByText('тестовый сбой рендера')).toBeInTheDocument()
    consoleError.mockRestore()
  })
})
