"""Каталог датасетов + загрузка нового датасета архивом (.zip) в data/<name>/.

Загрузка -- удобство для локального использования (пользователь работает через
браузер, а не git/CLI): архив распаковывается прямо в data/, дальше с ним можно
работать как с любым датасетом, положенным туда вручную (training/finetune_*.py,
docs/datasets.md). Карточка датасета (docs/dataset_cards/) при этом не создаётся
автоматически -- её описание и проверка лицензии остаются ручным шагом, см.
docs/governance/best_practices.md, обязательная проверка лицензии.
"""

import re
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..datasets_catalog import DATA_DIR, list_datasets

router = APIRouter(prefix="/datasets", tags=["Каталог датасетов"])

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 ГБ -- достаточно для PlantDoc-подобных датасетов, отсекает случайный мусор
# Проверялся только сжатый размер архива -- маленький, сильно сжимаемый файл
# (например, несколько сотен КБ нулей) мог распаковаться в десятки ГБ и
# забить диск (decompression bomb, см. аудит перед развёртыванием). Лимиты
# ниже -- на РАСПАКОВАННЫЙ суммарный размер и число файлов в архиве,
# проверяются до extractall.
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3  # 4 ГБ -- вдвое больше MAX_UPLOAD_BYTES, с запасом на типичное для фото/текста сжатие
MAX_ARCHIVE_ENTRIES = 200_000
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@router.get("", summary="Список датасетов: карточка (docs/dataset_cards/) + наличие на диске (data/)")
def get_datasets() -> list[dict]:
    return list_datasets()


@router.post("/upload", summary="Загрузить датасет архивом (.zip) в data/<name>/")
async def upload_dataset(name: str = Form(...), archive: UploadFile = File(...)) -> dict:
    if not NAME_PATTERN.match(name):
        raise HTTPException(422, "Имя датасета -- только латиница, цифры, дефис, подчёркивание")
    if not archive.filename or not archive.filename.lower().endswith(".zip"):
        raise HTTPException(422, "Ожидается .zip архив")

    target_dir = DATA_DIR / name
    if target_dir.exists():
        raise HTTPException(409, f"data/{name} уже существует -- удалите вручную или выберите другое имя")

    contents = await archive.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Архив больше лимита ({MAX_UPLOAD_BYTES // 1024**3} ГБ)")

    tmp_zip = DATA_DIR / f".upload-{name}.zip"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip.write_bytes(contents)

    try:
        with zipfile.ZipFile(tmp_zip) as zf:
            infolist = zf.infolist()
            if len(infolist) > MAX_ARCHIVE_ENTRIES:
                raise HTTPException(413, f"Архив содержит слишком много файлов (>{MAX_ARCHIVE_ENTRIES})")

            total_uncompressed = sum(member.file_size for member in infolist)
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    413,
                    f"Суммарный распакованный размер архива ({total_uncompressed / 1024**3:.1f} ГБ) "
                    f"превышает лимит ({MAX_UNCOMPRESSED_BYTES // 1024**3} ГБ)",
                )

            resolved_target = target_dir.resolve()
            for member in infolist:
                member_path = (target_dir / member.filename).resolve()
                if resolved_target not in member_path.parents and member_path != resolved_target:
                    raise HTTPException(400, f"Архив содержит путь за пределами целевой папки: {member.filename}")
            zf.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(422, "Файл повреждён или не является zip-архивом") from exc
    finally:
        tmp_zip.unlink(missing_ok=True)

    stats = list_datasets()
    return next((d for d in stats if d["slug"] == name), {"slug": name, "on_disk": True})
