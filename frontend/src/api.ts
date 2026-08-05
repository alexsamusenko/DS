// В dev (`npm run dev`, порт 5173) FastAPI слушает отдельный порт 8000 --
// CORS для этого разрешён в service/app.py. В production фронт собран в
// статику и отдаётся тем же FastAPI-приложением на одном origin, поэтому
// относительный путь работает без изменений (см. service/app.py).
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

// По умолчанию (VITE_API_KEY не задан) сервер не требует ключа (service/auth.py)
// -- локальный запуск работает без какой-либо настройки, как и раньше. Ключ
// нужен только если бэкенд явно защищён (DS_API_KEY на сервере) для внешнего
// доступа -- тогда его нужно задать и здесь на этапе сборки фронта.
function authHeaders(): Record<string, string> {
  const key = import.meta.env.VITE_API_KEY as string | undefined
  return key ? { 'X-API-Key': key } : {}
}

// EventSource (используется для живого потока обучения) не умеет слать кастомные
// заголовки -- ключ, если он задан, передаётся через query-параметр (service/auth.py
// принимает оба варианта).
export function trainingHistoryStreamUrl(): string {
  const key = import.meta.env.VITE_API_KEY as string | undefined
  const query = key ? `?api_key=${encodeURIComponent(key)}` : ''
  return `${API_BASE}/training/history/stream${query}`
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, { ...init, headers: { ...authHeaders(), ...init.headers } })
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}))
    throw new Error(detail.detail ?? `${resp.status} ${resp.statusText}`)
  }
  return resp.json() as Promise<T>
}

export interface PredictionFeatures {
  soil_moisture_mean: number
  soil_nitrogen_mean: number
  precip_sum: number
  temp_sum: number
  ndvi_peak: number
  ndvi_integral: number
  disease_events_count: number
  max_risk_stage: number
  fertilizer_applications_count: number
  total_dose: number
}

export interface PredictionResponse {
  predicted_yield: number
  modality_importance: Record<string, number>
}

export interface TrainingSummary {
  rmse_all_modalities: number
  rmse_without_modality: Record<string, number>
  modality_importance: Record<string, number>
}

export function predict(features: PredictionFeatures) {
  return request<PredictionResponse>('/prediction/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  })
}

export function trainingSummary() {
  return request<TrainingSummary>('/prediction/training-summary')
}

export interface NerHistory {
  task: 'ner'
  status: 'running' | 'completed'
  smoke_test: boolean
  model_name: string
  updated_at: string
  hyperparameters: { epochs: number; batch_size: number; lr: number; max_length: number; train_examples: number; eval_examples: number }
  log_history: Record<string, number>[]
  // Появляется только когда status === 'completed' -- во время обучения (running)
  // финальная оценка ещё не посчитана, есть только log_history по эпохам.
  final_eval_metrics?: Record<string, number>
}

export interface LeafClassifierHistory {
  task: 'leaf_classifier'
  status: 'running' | 'completed'
  smoke_test: boolean
  pretrained: boolean
  updated_at: string
  hyperparameters: {
    epochs_head: number
    epochs_full: number
    lr_head: number
    lr_full: number
    batch_size: number
    image_size: number
    train_examples: number
    val_examples: number
  }
  class_names: string[]
  best_val_acc: number
  epoch_log: { phase: string; epoch: number; train_loss: number; train_acc: number; val_loss: number; val_acc: number }[]
}

export interface TrainingHistoryResponse {
  ner: NerHistory | null
  leaf_classifier: LeafClassifierHistory | null
}

export function trainingHistory() {
  return request<TrainingHistoryResponse>('/training/history')
}

export interface DatasetEntry {
  slug: string
  title: string
  license_summary: string
  source: string | null
  composition: string | null
  limitations: string | null
  usage_in_project: string | null
  on_disk: boolean
  file_count?: number
  total_size_bytes?: number
}

export function listDatasets() {
  return request<DatasetEntry[]>('/datasets')
}

export async function uploadDataset(name: string, archive: File): Promise<DatasetEntry> {
  const form = new FormData()
  form.append('name', name)
  form.append('archive', archive)
  return request<DatasetEntry>('/datasets/upload', { method: 'POST', body: form })
}

export interface Plot {
  plot_id: number
  baseline: number
  R: number
  s: number
  area: number
}

export interface OptimizeRequest {
  plots: Plot[]
  price_yield: number
  price_fert: number
  dose_min: number
  dose_max: number
  budget: number | null
}

export interface OptimizeResponse {
  doses: number[]
  total_profit: number
  total_profit_uniform: number
  profit_gain_percent: number
}

export function optimize(req: OptimizeRequest) {
  return request<OptimizeResponse>('/optimization/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
}

export interface IsoxmlExportRequest extends OptimizeRequest {
  customer_name: string
  farm_name: string
  task_designator: string
}

export async function exportIsoxml(req: IsoxmlExportRequest): Promise<Blob> {
  const resp = await fetch(`${API_BASE}/integration/export-isoxml`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}))
    throw new Error(detail.detail ?? `${resp.status} ${resp.statusText}`)
  }
  return resp.blob()
}
