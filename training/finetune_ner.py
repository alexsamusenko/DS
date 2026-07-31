#!/usr/bin/env python3
"""Дообучение RuBERT для извлечения агрономических сущностей (L1 NER, §2.5.4).

Метки соответствуют классам онтологии (docs/chapter2/ontology_model.md, §2.1.3),
схема разметки BIO:
    O, B-KULTURA, I-KULTURA, B-AGROPRIEM, I-AGROPRIEM,
    B-POKAZATEL, I-POKAZATEL, B-VREDITEL, I-VREDITEL

Формат обучающих данных -- JSONL, одна строка на предложение:
    {"tokens": ["Внесение", "азотных", "удобрений", "на", "поле", "7"],
     "tags":   ["O", "B-AGROPRIEM", "I-AGROPRIEM", "O", "O", "O"]}

Два режима запуска:

1. Продовый (по умолчанию) -- дообучение реальной модели на реальном корпусе:
       python3 training/finetune_ner.py --train data/ner_train.jsonl --eval data/ner_eval.jsonl
   Требует доступа к huggingface.co (скачивание весов DeepPavlov/rubert-base-cased)
   и размеченного корпуса -- см. docs/datasets.md про сбор корпуса вручную.

2. Smoke-test -- проверка, что пайплайн (данные -> токенизация -> обучение ->
   метрики) работает end-to-end, БЕЗ обращения к интернету:
       python3 training/finetune_ner.py --smoke-test --epochs 30 --batch-size 16 --lr 3e-4
   Использует программно сгенерированный синтетический корпус (500 предложений
   из шаблонов и словарей реальных агротерминов, generate_synthetic_corpus)
   и маленькую BERT-конфигурацию со случайной инициализацией (без
   предобученных весов) -- проверено: F1 сходится с 0 до 1.0 за ~15-20 эпох,
   ~1 минута на CPU. Первая версия (9-12 предложений вручную) не давала
   пайплайну достаточно данных для сходимости даже за 200 эпох -- это не
   баг, а нехватка объёма и разнообразия обучающей выборки.
   Из этой песочницы huggingface.co недоступен (проверено -- см.
   docs/governance/licenses.md про сетевые ограничения), поэтому продовый
   режим здесь не запускался; smoke-test -- единственное, что можно
   реально прогнать в этом окружении.

Гиперпараметры по умолчанию и их обоснование -- docs/chapter2/model_training.md, §2.5.4.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABELS = [
    "O",
    "B-KULTURA", "I-KULTURA",
    "B-AGROPRIEM", "I-AGROPRIEM",
    "B-POKAZATEL", "I-POKAZATEL",
    "B-VREDITEL", "I-VREDITEL",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

DEFAULT_MODEL_NAME = "DeepPavlov/rubert-base-cased"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

def load_jsonl(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def build_smoke_test_corpus():
    """Псевдоним для обратной совместимости -- см. generate_synthetic_corpus()."""
    return generate_synthetic_corpus()


# ---------------------------------------------------------------------------
# Программный генератор синтетического корпуса (шаблоны x словари).
#
# Ручная разметка 9-12 предложений (первая версия скрипта) недостаточна: при
# нескольких сотнях шагов обучения модель не видит нужного разнообразия и не
# сходится даже на собственной обучающей выборке (см. docs/chapter2/model_training.md,
# запись о первом прогоне). Вместо того чтобы наращивать число эпох на тех же
# 9 примерах, генерируем на порядки больше размеченных предложений программно --
# из словарей реальных агрономических терминов и набора шаблонов. Это не
# заменяет ручную разметку реального корпуса (docs/datasets.md) для продовой
# модели, но даёт пайплайну достаточно данных, чтобы обучение было
# содержательным, а не искусственным.
# ---------------------------------------------------------------------------

VOCAB = {
    "KULTURA": [
        "Пшеница озимая", "Пшеница яровая", "Ячмень яровой", "Ячмень озимый",
        "Кукуруза", "Подсолнечник", "Соя", "Рапс яровой", "Рожь озимая",
        "Овёс", "Гречиха", "Картофель", "Свёкла сахарная", "Горох",
    ],
    "AGROPRIEM": [
        "внесение азотных удобрений", "внесение калийных удобрений",
        "внесение фосфорных удобрений", "опрыскивание фунгицидом",
        "опрыскивание инсектицидом", "обработку гербицидом", "боронование",
        "вспашку", "весенний полив", "довсходовое боронование",
        "подкормку азотом", "протравливание семян",
    ],
    "POKAZATEL": [
        "Влажность почвы", "Содержание азота", "Содержание фосфора",
        "Содержание калия", "Кислотность почвы", "Температура почвы",
        "Электропроводность почвы", "Сумма эффективных температур",
    ],
    "VREDITEL": [
        "тля", "колорадский жук", "мучнистая роса", "ржавчина", "фузариоз",
        "проволочник", "саранча", "слизни", "септориоз", "корневая гниль",
    ],
}

SENTENCE_TEMPLATES = [
    "Провели {AGROPRIEM} на поле {NUM}.",
    "{KULTURA} показала признаки поражения: {VREDITEL}.",
    "{POKAZATEL} снизилась до {NUM} процентов.",
    "{POKAZATEL} повысилась до {NUM} единиц.",
    "На участке {NUM} обнаружена {VREDITEL}.",
    "Провели {AGROPRIEM} против {VREDITEL}.",
    "{KULTURA} взошла неравномерно на поле {NUM}.",
    "{POKAZATEL} в норме на всех участках.",
    "Запланировано {AGROPRIEM} на следующей неделе.",
    "Отмечена стадия кущения у культуры {KULTURA}.",
    "{KULTURA} пострадала от {VREDITEL} на поле {NUM}.",
    "После дождя {POKAZATEL} резко изменилась.",
    "Агроном рекомендовал {AGROPRIEM} для участка {NUM}.",
    "{POKAZATEL} остаётся низкой третью неделю подряд.",
    "На поле {NUM} провели {AGROPRIEM} и {AGROPRIEM}.",
    "{KULTURA} и {KULTURA} посеяны в этом сезоне.",
    "Специалист обнаружил {VREDITEL} рядом с посевами {KULTURA}.",
    "{POKAZATEL} измерена сенсором на глубине тридцати сантиметров.",
]

FILLER_WORDS_BEFORE = {"AGROPRIEM": [], "KULTURA": [], "POKAZATEL": [], "VREDITEL": []}


def _tag_phrase(phrase, entity_type):
    words = phrase.split()
    tags = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(words) - 1)
    return words, tags


def _fill_template(template, rng):
    tokens, tags = [], []
    for part in template.split():
        # части шаблона -- это либо литеральные слова (возможно, со знаком
        # препинания на конце), либо плейсхолдеры вида {KULTURA}.
        placeholder = part.strip(".,:")
        trailing_punct = part[len(placeholder):]

        if placeholder.startswith("{") and placeholder.endswith("}"):
            entity_type = placeholder[1:-1]
            if entity_type == "NUM":
                tokens.append(str(rng.randint(1, 24)))
                tags.append("O")
            else:
                phrase = rng.choice(VOCAB[entity_type])
                words, entity_tags = _tag_phrase(phrase, entity_type)
                tokens.extend(words)
                tags.extend(entity_tags)
        else:
            tokens.append(part)
            tags.append("O")

        if trailing_punct:
            tokens.append(trailing_punct)
            tags.append("O")

    return tokens, tags


def generate_synthetic_corpus(n_examples=400, seed=42):
    """Сгенерировать n_examples размеченных предложений из шаблонов и словарей.

    Не является заменой ручной разметке реального корпуса (docs/datasets.md) --
    это контролируемый суррогат, достаточный для содержательной проверки
    пайплайна дообучения (в т.ч. в этой песочнице, без доступа к huggingface.co).
    """
    rng = random.Random(seed)
    examples = []
    seen = set()
    attempts = 0
    while len(examples) < n_examples and attempts < n_examples * 20:
        attempts += 1
        template = rng.choice(SENTENCE_TEMPLATES)
        tokens, tags = _fill_template(template, rng)
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        examples.append({"tokens": tokens, "tags": tags})
    return examples


def _train_eval_split(examples, eval_fraction=0.25, seed=42):
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


# ---------------------------------------------------------------------------
# Офлайн-токенизатор для smoke-теста (без обращения к huggingface.co)
# ---------------------------------------------------------------------------

class ToyWordTokenizer:
    """Минимальный пословный токенизатор на словаре, построенном по корпусу.

    Не предназначен для продового использования -- только для smoke-теста,
    когда AutoTokenizer.from_pretrained недоступен (нет сети). Один токен = одно слово,
    поэтому выравнивание меток с суб-токенами (нужное для реального BPE/WordPiece
    токенизатора) здесь не требуется.
    """

    def __init__(self, examples):
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for ex in examples:
            for token in ex["tokens"]:
                key = token.lower()
                if key not in vocab:
                    vocab[key] = len(vocab)
        self.vocab = vocab
        self.pad_token_id = vocab["[PAD]"]
        self.cls_token_id = vocab["[CLS]"]
        self.sep_token_id = vocab["[SEP]"]
        self.unk_token_id = vocab["[UNK]"]

    def __len__(self):
        return len(self.vocab)

    def encode(self, tokens, max_length):
        ids = [self.cls_token_id] + [self.vocab.get(t.lower(), self.unk_token_id) for t in tokens] + [self.sep_token_id]
        ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attention_mask.append(0)
        return ids, attention_mask


class ToyNERDataset(Dataset):
    """Датасет для smoke-теста: пословное кодирование через ToyWordTokenizer."""

    def __init__(self, examples, tokenizer, max_length=64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids, attention_mask = self.tokenizer.encode(ex["tokens"], self.max_length)

        # [CLS] + токены + [SEP]/паддинг: метка -100 для [CLS], [SEP] и паддинга
        # (стандартное значение ignore_index в CrossEntropyLoss transformers)
        labels = [-100]
        for tag in ex["tags"][: self.max_length - 2]:
            labels.append(LABEL2ID[tag])
        while len(labels) < self.max_length:
            labels.append(-100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class HFNERDataset(Dataset):
    """Датасет для продового режима: субтокенизация реальным AutoTokenizer.

    Метка ставится только на первый суб-токен слова (стандартная практика
    для NER на BPE/WordPiece токенизаторах), остальные суб-токены и
    служебные токены получают -100.
    """

    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = encoding.word_ids()
        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(LABEL2ID[ex["tags"][word_idx]])
            else:
                labels.append(-100)  # непервый суб-токен того же слова
            previous_word_idx = word_idx

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Метрика: entity-level F1 по BIO-разметке (собственная реализация -- seqeval
# не устанавливается в текущем окружении из-за конфликта сборочных зависимостей)
# ---------------------------------------------------------------------------

def _bio_to_spans(tags):
    """['O','B-KULTURA','I-KULTURA','O'] -> {(1,3,'KULTURA')} (полуинтервал индексов)."""
    spans = set()
    start, entity_type = None, None
    for i, tag in enumerate(tags + ["O"]):  # хвостовой 'O' закрывает последний спан
        if tag.startswith("B-"):
            if start is not None:
                spans.add((start, i, entity_type))
            start, entity_type = i, tag[2:]
        elif tag.startswith("I-") and entity_type == tag[2:]:
            continue
        else:
            if start is not None:
                spans.add((start, i, entity_type))
            start, entity_type = None, None
    return spans


def compute_entity_f1(eval_pred):
    predictions, label_ids = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    tp = fp = fn = 0
    for pred_row, label_row in zip(predictions, label_ids):
        pred_tags, true_tags = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_tags.append(ID2LABEL[int(p)])
            true_tags.append(ID2LABEL[int(l)])

        pred_spans = _bio_to_spans(pred_tags)
        true_spans = _bio_to_spans(true_tags)

        tp += len(pred_spans & true_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, help="Путь к обучающему JSONL (§docstring). Игнорируется при --smoke-test")
    parser.add_argument("--eval", type=Path, help="Путь к валидационному JSONL. Игнорируется при --smoke-test")
    parser.add_argument("--smoke-test", action="store_true", help="Офлайн-проверка на программно сгенерированном синтетическом корпусе (без сети и внешних данных)")
    parser.add_argument("--smoke-corpus-size", type=int, default=500, help="Размер синтетического корпуса для --smoke-test (§docstring, generate_synthetic_corpus)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Базовая модель HF Hub (по умолчанию {DEFAULT_MODEL_NAME})")
    parser.add_argument("--output-dir", type=Path, default=Path("build/ner_model"))
    parser.add_argument("--epochs", type=int, default=15, help="15 по умолчанию -- малый корпус требует больше эпох (§2.5.4)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5, help="Стандартный LR для fine-tuning BERT (Devlin et al., 2018)")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        print("=== Smoke-test: офлайн-проверка пайплайна, без обращения к huggingface.co ===")
        corpus = generate_synthetic_corpus(n_examples=args.smoke_corpus_size, seed=args.seed)
        train_examples, eval_examples = _train_eval_split(corpus, seed=args.seed)
        print(f"Синтетический корпус: {len(train_examples)} обучающих / {len(eval_examples)} валидационных примеров")

        tokenizer = ToyWordTokenizer(corpus)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=args.max_length,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = BertForTokenClassification(config)  # случайные веса, не предобученные

        train_dataset = ToyNERDataset(train_examples, tokenizer, args.max_length)
        eval_dataset = ToyNERDataset(eval_examples, tokenizer, args.max_length)
        data_collator = None  # ToyNERDataset уже паддит вручную до max_length
    else:
        if not args.train or not args.eval:
            raise SystemExit(
                "Для продового режима нужны --train и --eval (JSONL с ручной разметкой, "
                "см. docs/datasets.md). Для проверки пайплайна без данных используйте --smoke-test."
            )
        print(f"=== Продовое дообучение: {args.model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
        )
        train_dataset = HFNERDataset(load_jsonl(args.train), tokenizer, args.max_length)
        eval_dataset = HFNERDataset(load_jsonl(args.eval), tokenizer, args.max_length)
        data_collator = None

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        logging_strategy="epoch",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_entity_f1,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Итоговые метрики на валидации (entity-level, §2.5.4):", metrics)

    if args.smoke_test:
        # Диагностика "может ли пайплайн вообще чему-то обучиться" (классическая
        # проверка overfit-on-train-set): на 3 примерах held-out и случайно
        # инициализированной модели генерализация не показательна, а вот
        # способность подогнаться под собственную обучающую выборку -- да.
        train_metrics = trainer.evaluate(eval_dataset=train_dataset)
        print("Метрики на обучающей выборке (диагностика сходимости пайплайна):", train_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    print(f"Модель сохранена в {args.output_dir}")


if __name__ == "__main__":
    main()
