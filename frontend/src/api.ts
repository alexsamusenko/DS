// В dev (`npm run dev`, порт 5173) FastAPI слушает отдельный порт 8000 --
// CORS для этого разрешён в service/app.py. В production фронт собран в
// статику и отдаётся тем же FastAPI-приложением на одном origin, поэтому
// относительный путь работает без изменений (см. service/app.py).
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, init)
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
  smoke_test: boolean
  model_name: string
  epochs: number
  log_history: Record<string, number>[]
  final_eval_metrics: Record<string, number>
}

export interface LeafClassifierHistory {
  task: 'leaf_classifier'
  smoke_test: boolean
  pretrained: boolean
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
