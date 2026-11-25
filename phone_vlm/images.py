"""Единая обработка изображений для обучения и применения.

Важна не только форма тензора: алгоритм масштабирования, цвет и нормализация
тоже должны совпадать. Letterbox вписывает весь кадр в квадрат с полями, чтобы
центральное кадрирование не срезало камеры и другие части длинного телефона.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import ProjectError


def read_rgb(path: Path, policy: dict) -> Image.Image:
    """Читает копию изображения, проверив его заголовок до полной распаковки.

    Исходный файл никогда не изменяется. Поворот по EXIF учитывается один раз.
    RGB, RGBA и 8-битный grayscale явно приводятся к RGB; прозрачность кладётся
    на белый фон. CMYK и высокобитные режимы отклоняются понятным сообщением.
    """
    if not path.is_file():
        raise ProjectError(f"Файл не найден: {path}")
    if path.stat().st_size > policy["max_file_mb"] * 1024 * 1024:
        raise ProjectError("Файл превышает допустимый размер в мегабайтах.")
    try:
        with Image.open(path) as source:
            if source.format not in policy["allowed_formats"]:
                raise ProjectError("Поддерживаются форматы: " + ", ".join(policy["allowed_formats"]))
            w, h = source.size
            if min(w, h) < policy["min_short_side"] or max(w, h) < policy["min_long_side"]:
                raise ProjectError(f"Недостаточное разрешение: короткая сторона не менее {policy['min_short_side']}, "
                                   f"длинная — {policy['min_long_side']} пикселей.")
            if max(w, h) > policy["max_side"]:
                raise ProjectError("Слишком большое разрешение изображения.")
            if source.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                raise ProjectError(f"Неподдерживаемый цветовой режим: {source.mode}")
            if getattr(source, "n_frames", 1) != 1:
                raise ProjectError("Ожидается один неподвижный кадр.")
            source.load()  # Здесь обнаруживаются многие повреждённые файлы.
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA", "P"):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ProjectError(f"Не удалось прочитать изображение: {path.name}") from exc


def quality_warnings(image: Image.Image) -> list[str]:
    """Эвристики предупреждают, но не доказывают отсутствие телефона.

    Тёмная фотография может быть корректной. Поэтому по яркости нельзя
    автоматически отвергать весь кадр или обещать проверку его содержания.
    """
    pixels = np.asarray(image.resize((128, 128)).convert("L"), dtype=np.float32)
    warnings = []
    if float(pixels.std()) < 1.0:
        warnings.append("Почти однотонный кадр: объект может быть неразличим.")
    if float(pixels.mean()) < 12.0:
        warnings.append("Очень тёмный кадр.")
    elif float(pixels.mean()) > 245.0:
        warnings.append("Очень светлый кадр.")
    return warnings


def preprocess(image: Image.Image, spec: dict) -> np.ndarray:
    """Возвращает float32 в формате C×H×W; никаких загрузок через интернет."""
    if spec["resize"] != "letterbox" or spec["interpolation"] != "bilinear":
        raise ProjectError("Эта версия поддерживает только letterbox и bilinear.")
    size = int(spec["size"])
    if size != 256 or not spec.get("exif_transpose"):
        raise ProjectError("Для данной модели зафиксированы вход 256 и обработка EXIF.")
    # ImageOps.contain сохраняет соотношение сторон. Нечётное поле может
    # отличаться слева и справа на один пиксель; это одинаково в обоих режимах.
    fitted = ImageOps.contain(image, (size, size), method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), tuple(spec["fill"]))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    values = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.asarray(spec["mean"], dtype=np.float32)
    std = np.asarray(spec["std"], dtype=np.float32)
    if np.any(std <= 0):
        raise ProjectError("Значения std должны быть положительными.")
    return np.ascontiguousarray(((values - mean) / std).transpose(2, 0, 1))
