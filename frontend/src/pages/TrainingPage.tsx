import { useEffect, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { trainingHistory, type LeafClassifierHistory, type NerHistory, type TrainingHistoryResponse } from '../api'

function mergeNerLog(logHistory: NerHistory['log_history']) {
  const byEpoch = new Map<number, Record<string, number>>()
  for (const entry of logHistory) {
    const epoch = Math.round((entry.epoch ?? 0) * 100) / 100
    byEpoch.set(epoch, { ...byEpoch.get(epoch), ...entry, epoch })
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch)
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString('ru-RU', { dateStyle: 'medium', timeStyle: 'short' })
}

function RawJson({ data }: { data: unknown }) {
  return (
    <details style={{ marginTop: '0.75rem' }}>
      <summary className="hint" style={{ cursor: 'pointer' }}>
        Полный лог обучения (JSON)
      </summary>
      <pre style={{ fontSize: '0.75rem', overflowX: 'auto', background: '#eef1ec', padding: '0.75rem', borderRadius: '6px' }}>
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  )
}

function HyperparamRow({ params }: { params: Record<string, number | string> }) {
  return (
    <div className="stat-row">
      {Object.entries(params).map(([key, value]) => (
        <div className="stat" key={key}>
          <div className="value" style={{ fontSize: '1rem' }}>
            {value}
          </div>
          <div className="label">{key}</div>
        </div>
      ))}
    </div>
  )
}

function NerCard({ run }: { run: NerHistory | null }) {
  if (!run) {
    return (
      <div className="card">
        <h2>NER (извлечение агрономических сущностей)</h2>
        <p className="hint">
          Ещё не обучалась. Запустите <code>python3 training/finetune_ner.py</code> (кладёт{' '}
          <code>data/ner_train.jsonl</code> / <code>data/ner_eval.jsonl</code>) или{' '}
          <code>docker compose run --rm train python3 training/finetune_ner.py</code>.
        </p>
      </div>
    )
  }

  const points = mergeNerLog(run.log_history)
  const bestF1Point = points.reduce((best, p) => ((p.eval_f1 ?? -1) > (best?.eval_f1 ?? -1) ? p : best), points[0])

  return (
    <div className="card">
      <h2>NER (извлечение агрономических сущностей)</h2>
      <p className="hint">
        {run.smoke_test ? 'Smoke-test: ' : 'Продовый прогон: '}
        {run.smoke_test
          ? 'проверка пайплайна на программно сгенерированном корпусе, модель со случайной инициализацией -- метрики не отражают реальное качество NER.'
          : `дообучение ${run.model_name} на реальном размеченном корпусе.`}{' '}
        Завершено {formatDate(run.finished_at)}.
      </p>
      <div className="stat-row">
        <div className="stat">
          <div className="value">{(run.final_eval_metrics.eval_f1 ?? 0).toFixed(3)}</div>
          <div className="label">итоговый entity-level F1</div>
        </div>
        <div className="stat">
          <div className="value">{(bestF1Point?.eval_f1 ?? 0).toFixed(3)}</div>
          <div className="label">лучший F1 (эпоха {bestF1Point?.epoch ?? '—'})</div>
        </div>
        <div className="stat">
          <div className="value">{run.hyperparameters.train_examples}</div>
          <div className="label">обучающих примеров</div>
        </div>
        <div className="stat">
          <div className="value">{run.hyperparameters.eval_examples}</div>
          <div className="label">валидационных примеров</div>
        </div>
      </div>
      <h3>Гиперпараметры</h3>
      <HyperparamRow
        params={{
          эпохи: run.hyperparameters.epochs,
          'batch size': run.hyperparameters.batch_size,
          lr: run.hyperparameters.lr,
          'max length': run.hyperparameters.max_length,
        }}
      />
      <div className="grid-2">
        <div>
          <h3>Loss</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
              <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="loss" name="train" stroke="#3f7d3f" dot={false} connectNulls />
              <Line type="monotone" dataKey="eval_loss" name="eval" stroke="#b3432b" dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3>Entity-level F1 / precision / recall</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
              <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="eval_f1" name="F1" stroke="#3f7d3f" dot={false} connectNulls />
              <Line type="monotone" dataKey="eval_precision" name="precision" stroke="#7d8f3f" dot={false} connectNulls />
              <Line type="monotone" dataKey="eval_recall" name="recall" stroke="#3f6d7d" dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <RawJson data={run} />
    </div>
  )
}

function LeafClassifierCard({ run }: { run: LeafClassifierHistory | null }) {
  if (!run) {
    return (
      <div className="card">
        <h2>Классификатор поражений листа</h2>
        <p className="hint">
          Ещё не обучался. Запустите <code>python3 training/finetune_leaf_classifier.py</code> (кладёт датасет в{' '}
          <code>data/plantdoc/</code>) или <code>docker compose run --rm train python3 training/finetune_leaf_classifier.py</code>.
        </p>
      </div>
    )
  }

  const points = run.epoch_log.map((e, i) => ({ ...e, x: i, label: `${e.phase === 'head' ? 'голова' : 'сеть'} #${e.epoch}` }))
  const bestPoint = points.reduce((best, p) => (p.val_acc > best.val_acc ? p : best), points[0])

  return (
    <div className="card">
      <h2>Классификатор поражений листа</h2>
      <p className="hint">
        {run.smoke_test
          ? 'Smoke-test: процедурно сгенерированные изображения (не фото растений), проверка сходимости пайплайна.'
          : `Продовый прогон на реальных фото, ${run.pretrained ? 'веса ImageNet (transfer learning)' : 'обучение с нуля'}.`}{' '}
        Завершено {formatDate(run.finished_at)}.
      </p>
      <div className="stat-row">
        <div className="stat">
          <div className="value">{run.class_names.length}</div>
          <div className="label">классов</div>
        </div>
        <div className="stat">
          <div className="value">{(run.best_val_acc * 100).toFixed(1)}%</div>
          <div className="label">лучшая val accuracy</div>
        </div>
        <div className="stat">
          <div className="value">{bestPoint?.label ?? '—'}</div>
          <div className="label">на какой эпохе достигнут максимум</div>
        </div>
        <div className="stat">
          <div className="value">
            {run.hyperparameters.train_examples} / {run.hyperparameters.val_examples}
          </div>
          <div className="label">train / val примеров</div>
        </div>
      </div>
      <h3>Гиперпараметры</h3>
      <HyperparamRow
        params={{
          'эпох (голова/сеть)': `${run.hyperparameters.epochs_head}/${run.hyperparameters.epochs_full}`,
          'lr (голова/сеть)': `${run.hyperparameters.lr_head}/${run.hyperparameters.lr_full}`,
          'batch size': run.hyperparameters.batch_size,
          'размер изображения': run.hyperparameters.image_size,
        }}
      />
      <p className="hint">Классы: {run.class_names.join(', ')}</p>
      <div className="grid-2">
        <div>
          <h3>Loss</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
              <XAxis dataKey="label" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="train_loss" name="train" stroke="#3f7d3f" dot={false} />
              <Line type="monotone" dataKey="val_loss" name="val" stroke="#b3432b" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3>Accuracy</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e7df" />
              <XAxis dataKey="label" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="train_acc" name="train" stroke="#3f7d3f" dot={false} />
              <Line type="monotone" dataKey="val_acc" name="val" stroke="#b3432b" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <RawJson data={run} />
    </div>
  )
}

export default function TrainingPage() {
  const [history, setHistory] = useState<TrainingHistoryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    trainingHistory().then(setHistory).catch((e) => setError(String(e.message ?? e)))
  }, [])

  if (error) return <div className="error-banner">{error}</div>
  if (!history) return <p className="muted">Загрузка...</p>

  return (
    <>
      <NerCard run={history.ner} />
      <LeafClassifierCard run={history.leaf_classifier} />
    </>
  )
}
