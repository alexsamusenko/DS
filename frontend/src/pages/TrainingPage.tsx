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

  return (
    <div className="card">
      <h2>NER (извлечение агрономических сущностей)</h2>
      <div className="stat-row">
        <div className="stat">
          <div className="value">{run.smoke_test ? 'smoke-test' : 'прод'}</div>
          <div className="label">режим</div>
        </div>
        <div className="stat">
          <div className="value">{run.model_name}</div>
          <div className="label">базовая модель</div>
        </div>
        <div className="stat">
          <div className="value">{(run.final_eval_metrics.eval_f1 ?? 0).toFixed(3)}</div>
          <div className="label">итоговый entity-level F1</div>
        </div>
      </div>
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

  return (
    <div className="card">
      <h2>Классификатор поражений листа</h2>
      <div className="stat-row">
        <div className="stat">
          <div className="value">{run.smoke_test ? 'smoke-test' : 'прод'}</div>
          <div className="label">режим</div>
        </div>
        <div className="stat">
          <div className="value">{run.pretrained ? 'ImageNet' : 'с нуля'}</div>
          <div className="label">инициализация весов</div>
        </div>
        <div className="stat">
          <div className="value">{run.class_names.length}</div>
          <div className="label">классов</div>
        </div>
        <div className="stat">
          <div className="value">{(run.best_val_acc * 100).toFixed(1)}%</div>
          <div className="label">лучшая val accuracy</div>
        </div>
      </div>
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
