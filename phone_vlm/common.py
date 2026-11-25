"""Пути, контрольные суммы и запись служебных файлов.

Здесь нет импортов PyTorch: подготовка таблиц и просмотр справки должны работать
без загрузки нейросети. Относительные пути всегда считаются от каталога config,
поэтому перенос проекта не требует исправлять исходный код.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("android", "iphone")  # Порядок фиксирован во всех массивах и отчётах.
DISPLAY = {"android": "свой (Android)", "iphone": "чужой (iPhone / iOS)"}


class ProjectError(ValueError):
    """Ожидаемая ошибка пользователя или несовместимого артефакта."""


def now() -> str:
    """В файлах храним однозначное время UTC с часовым поясом."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    """Стабильный хеш структуры, не зависящий от отступов в JSON."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    """Сначала пишем временный файл, затем заменяем целевой целиком."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                   + "\n", encoding="utf-8")
    tmp.replace(path)


def load_config(path: str | Path | None = None) -> tuple[dict, Path]:
    location = Path(path).resolve() if path else ROOT / "config.json"
    config = read_json(location)
    if config.get("schema_version") != 1:
        raise ProjectError("Неподдерживаемая версия config.json.")
    return config, location.parent


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def new_directory(path: Path) -> None:
    """Эксперимент не перезаписывается случайно повторным запуском команды."""
    if path.exists():
        raise ProjectError(f"Папка уже существует: {path}. Выберите новое имя.")
    path.mkdir(parents=True)


def environment() -> dict:
    names = ["torch", "torchvision", "timm", "safetensors", "numpy", "Pillow",
             "scikit-learn", "scipy", "psutil"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"code_version": __version__, "packages": versions}
