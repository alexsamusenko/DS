#!/usr/bin/env python3
"""Дообучение классификатора поражений листа (L1, D_img, §2.5.4).

ВАЖНО: PlantVillage -- целевой датасет для продовой версии -- пока НЕ
используется: лицензия не подтверждена (см. docs/governance/licenses.md).
Скрипт рассчитан на формат torchvision.datasets.ImageFolder (одна папка на
класс: data/train/<class_name>/*.jpg), но реальных данных в проекте нет.

Два режима запуска:

1. Продовый -- дообучение EfficientNet-B0 с весами ImageNet на реальных данных
   (предобученные веса используются ПО УМОЛЧАНИЮ). Положите датасет в формате
   ImageFolder в data/plantdoc/ (train/<class>/*.jpg + test-или-val/<class>/*.jpg,
   например распакованный PlantDoc -- docs/dataset_cards/plantdoc.md) и запустите
   без единого флага:
       python3 training/finetune_leaf_classifier.py
   Другой путь к данным -- через --data-dir; val/test-подпапка определяется
   автоматически (--val-subdir -- только если нужно переопределить).
   При первом запуске torchvision скачивает веса `EfficientNet_B0_Weights.IMAGENET1K_V1`
   (~21 МБ, с download.pytorch.org, кешируются в `~/.cache/torch/hub/checkpoints/`) --
   нужен доступ в сеть один раз. Источник и лицензия весов -- см.
   docs/governance/licenses.md, раздел «Предобученные веса». Если сеть недоступна
   (как в песочнице разработки) -- флаг `--no-pretrained` обучает с нуля
   (`weights=None`); это заведомо хуже на малом датасете (PlantDoc -- 2578 фото)
   и предназначен только как офлайн-резерв, а не рекомендуемый режим.

2. Smoke-test -- проверка пайплайна на процедурно сгенерированных
   изображениях (НЕ фото растений, просто цветовые паттерны с фиксированной
   зависимостью класс -> паттерн, чтобы было что учить), без сети:
       python3 training/finetune_leaf_classifier.py --smoke-test --epochs 12
   Модель обучается с нуля (weights=None) -- предобученные веса тут намеренно не
   нужны: цель smoke-теста -- проверить сходимость пайплайна на контролируемых
   данных, а не качество transfer learning.

Гиперпараметры по умолчанию и их обоснование -- docs/chapter2/model_training.md, §2.5.4.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Статистика нормализации ImageNet -- обязательна при использовании
# предобученных весов (EfficientNet_B0_Weights.IMAGENET1K_V1 обучены на входах,
# нормализованных именно этими mean/std; без этого шага предобученные фильтры
# получают распределение пикселей, для которого не калибровались, и transfer
# learning работает существенно хуже). Совпадает с
# EfficientNet_B0_Weights.IMAGENET1K_V1.transforms().
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Классы соответствуют стадиям стадия_riska класса ВредительБолезнь (§2.1.3):
# отсутствие поражения и три условных типа визуального паттерна поражения.
SYNTHETIC_CLASSES = ["healthy", "leaf_spot", "rust", "powdery_mildew"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Синтетический генератор изображений для smoke-теста.
#
# Это НЕ имитация PlantVillage и не претендует на визуальное сходство с
# реальными поражениями растений -- только контролируемая, программно
# генерируемая зависимость "класс -> визуальный паттерн" с шумом, достаточная
# для проверки, что CNN-пайплайн (данные -> обучение -> сходимость) работает
# корректно, без использования датасетов с неподтверждённой лицензией.
# ---------------------------------------------------------------------------

def _generate_synthetic_image(class_idx, size, rng):
    img = Image.new("RGB", (size, size), color=(34, 120, 34))  # зелёный фон "листа"
    draw = ImageDraw.Draw(img)

    class_name = SYNTHETIC_CLASSES[class_idx]
    n_blobs = rng.integers(3, 7)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, size), rng.integers(0, size)
        r = rng.integers(size // 12, size // 6)
        if class_name == "healthy":
            color = (34 + rng.integers(-10, 10), 120 + rng.integers(-15, 15), 34 + rng.integers(-10, 10))
        elif class_name == "leaf_spot":
            color = (60 + rng.integers(-10, 10), 40 + rng.integers(-10, 10), 20 + rng.integers(-10, 10))  # тёмные пятна
        elif class_name == "rust":
            color = (190 + rng.integers(-15, 15), 110 + rng.integers(-15, 15), 30 + rng.integers(-10, 10))  # рыжие пятна
        else:  # powdery_mildew
            color = (230 + rng.integers(-15, 15), 230 + rng.integers(-15, 15), 220 + rng.integers(-15, 15))  # белёсый налёт
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # общий шум по изображению
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-8, 8, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class SyntheticLeafDataset(Dataset):
    """Процедурно генерируемый датасет для smoke-теста (см. пояснение выше)."""

    def __init__(self, n_per_class, size, seed, transform):
        rng = np.random.default_rng(seed)
        self.images = []
        self.labels = []
        for class_idx in range(len(SYNTHETIC_CLASSES)):
            for _ in range(n_per_class):
                self.images.append(_generate_synthetic_image(class_idx, size, rng))
                self.labels.append(class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def build_transforms(image_size, train):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def restrict_to_common_classes(train_dataset, val_dataset):
    """Выровнять классы train/val по пересечению (в PlantDoc train содержит на один
    класс больше, чем test -- без этого индексы классов молча разъезжаются между
    train и val, т.к. torchvision.ImageFolder присваивает индексы по алфавиту
    внутри каждой папки отдельно)."""
    common = sorted(set(train_dataset.classes) & set(val_dataset.classes))
    class_to_idx = {name: i for i, name in enumerate(common)}

    for dataset in (train_dataset, val_dataset):
        old_idx_to_name = {v: k for k, v in dataset.class_to_idx.items()}
        new_samples = [
            (path, class_to_idx[old_idx_to_name[old_label]])
            for path, old_label in dataset.samples
            if old_idx_to_name[old_label] in class_to_idx
        ]
        dataset.samples = new_samples
        dataset.targets = [label for _, label in new_samples]
        dataset.classes = common
        dataset.class_to_idx = class_to_idx

    return common


def build_model(num_classes, pretrained):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/plantdoc"), help="Папка в формате ImageFolder (train/<class>/*.jpg + val-или-test/<class>/*.jpg). По умолчанию data/plantdoc -- положите туда датасет и запускайте вообще без флагов. Игнорируется при --smoke-test")
    parser.add_argument("--val-subdir", default=None, help="Имя подпапки для валидации внутри --data-dir. Если не указано -- автоопределение: сперва 'val', иначе 'test' (PlantDoc использует 'test')")
    parser.add_argument("--no-pretrained", action="store_true", help="Офлайн-резерв: обучение с нуля вместо весов ImageNet (только если download.pytorch.org недоступен). По умолчанию (без флага) веса ImageNet ИСПОЛЬЗУЮТСЯ -- см. docs/governance/licenses.md")
    parser.add_argument("--smoke-test", action="store_true", help="Офлайн-проверка на процедурно сгенерированных изображениях")
    parser.add_argument("--smoke-images-per-class", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=64, help="64 для smoke-теста (скорость на CPU); 224 для продового EfficientNet-B0")
    parser.add_argument("--output-dir", type=Path, default=Path("build/leaf_classifier"))
    parser.add_argument("--epochs-head", type=int, default=10, help="Фаза 1: обучается только классификационная голова (§2.5.4)")
    parser.add_argument("--epochs-full", type=int, default=20, help="Фаза 2: дообучение всей сети на малом LR")
    parser.add_argument("--epochs", type=int, help="Переопределяет и --epochs-head, и --epochs-full единым числом (удобно для smoke-теста)")
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-full", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs_head = args.epochs if args.epochs is not None else args.epochs_head
    epochs_full = args.epochs if args.epochs is not None else args.epochs_full

    if args.smoke_test:
        print("=== Smoke-test: офлайн-проверка на синтетических изображениях (не PlantVillage) ===")
        class_names = SYNTHETIC_CLASSES
        train_tf = build_transforms(args.image_size, train=True)
        eval_tf = build_transforms(args.image_size, train=False)

        full_dataset = SyntheticLeafDataset(args.smoke_images_per_class, args.image_size, args.seed, transform=eval_tf)
        n_val = int(0.2 * len(full_dataset))
        train_subset, val_subset = random_split(
            full_dataset, [len(full_dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_subset.dataset.transform = train_tf  # аугментации только для обучающей части
        pretrained = False
    else:
        if not (args.data_dir / "train").exists():
            raise SystemExit(
                f"Не найдена {args.data_dir / 'train'}. Положите датасет в формате ImageFolder "
                f"в {args.data_dir} (train/<class>/*.jpg + val-или-test/<class>/*.jpg) -- например, "
                "распакуйте туда PlantDoc, см. docs/dataset_cards/plantdoc.md -- и запустите скрипт "
                "снова. Для проверки пайплайна без данных используйте --smoke-test."
            )
        val_subdir = args.val_subdir
        if val_subdir is None:
            if (args.data_dir / "val").exists():
                val_subdir = "val"
            elif (args.data_dir / "test").exists():
                val_subdir = "test"
            else:
                raise SystemExit(
                    f"Не найдена ни {args.data_dir / 'val'}, ни {args.data_dir / 'test'} -- "
                    "укажите подпапку валидации явно через --val-subdir."
                )
        print(f"=== Продовое дообучение EfficientNet-B0 на {args.data_dir} (val: {val_subdir}) ===")
        train_dataset = ImageFolder(args.data_dir / "train", transform=build_transforms(args.image_size, train=True))
        val_dataset = ImageFolder(args.data_dir / val_subdir, transform=build_transforms(args.image_size, train=False))
        class_names = restrict_to_common_classes(train_dataset, val_dataset)
        print(f"Классов после выравнивания train/val: {len(class_names)} (было {len(train_dataset.classes)} train)")
        train_subset, val_subset = train_dataset, val_dataset
        pretrained = not args.no_pretrained

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=len(class_names), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Фаза 1: замороженный backbone, обучается только голова
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
    print(f"--- Фаза 1: голова, {epochs_head} эпох, lr={args.lr_head} ---")
    for epoch in range(1, epochs_head + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[голова] эпоха {epoch}/{epochs_head}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    # Фаза 2: разморозка всей сети, малый LR
    set_backbone_trainable(model, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_full)
    print(f"--- Фаза 2: вся сеть, {epochs_full} эпох, lr={args.lr_full} ---")
    for epoch in range(1, epochs_full + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[полная сеть] эпоха {epoch}/{epochs_full}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    print(f"Лучшая валидационная точность: {best_val_acc:.4f}")
    print(f"Классы: {class_names}")
    print(f"Модель сохранена в {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
