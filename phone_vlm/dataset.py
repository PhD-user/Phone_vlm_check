"""Разметка, контроль данных и воспроизводимое групповое разбиение.

Строка CSV описывает фотографию. device_id обозначает ФИЗИЧЕСКИЙ аппарат,
session_id — серию съёмки, group_id — дополнительную связь (например, исходное
изображение и его варианты). Объединение всех этих связей защищает разбиение
даже тогда, когда одна сессия содержит несколько устройств.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import CLASSES, ProjectError, fingerprint, resolve, sha256, write_json
from .images import read_rgb

FIELDS = ["path", "label", "device_id", "session_id", "group_id", "phone_model",
          "source", "split", "sha256", "dhash"]
SPLITS = ("train", "val", "test")


def read_manifest(path: Path, root: Path, *, require_split: bool = False,
                  verify_hashes: bool = False) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"path", "label", "device_id", "session_id"}.issubset(reader.fieldnames):
            raise ProjectError("В CSV нужны path,label,device_id,session_id.")
        rows = [{key: (row.get(key) or "").strip() for key in FIELDS} for row in reader]
    if not rows:
        raise ProjectError("В таблице нет фотографий.")
    seen = set()
    for i, row in enumerate(rows, 2):
        if row["label"] not in CLASSES:
            raise ProjectError(f"Строка {i}: метка должна быть android или iphone.")
        if not row["device_id"] or not row["session_id"]:
            raise ProjectError(f"Строка {i}: заполните device_id и session_id вручную.")
        if require_split and row["split"] not in SPLITS:
            raise ProjectError(f"Строка {i}: split должен быть train, val или test.")
        if not row["path"]:
            raise ProjectError(f"Строка {i}: пустой путь.")
        # CSV остаётся переносимым: абсолютные пути и выход из проекта запрещены.
        candidate = Path(row["path"])
        image_path = resolve(root, candidate)
        if candidate.is_absolute() or not image_path.is_relative_to(root.resolve()):
            raise ProjectError(f"Строка {i}: путь должен находиться внутри проекта.")
        if image_path in seen:
            raise ProjectError(f"Повторный путь в таблице: {row['path']}")
        seen.add(image_path)
        if verify_hashes:
            if not row["sha256"] or sha256(image_path) != row["sha256"]:
                raise ProjectError(f"Файл изменён или отсутствует его хеш: {row['path']}. Повторите audit и split.")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)


def index_folders(root: Path, folder: Path, output: Path) -> int:
    """Черновик разметки: идентификаторы намеренно остаются пустыми.

    Автоматически назначить каждой фотографии отдельный device_id было бы
    удобно, но это создало бы ложную защиту от утечки одной серии в тест.
    """
    if output.exists():
        raise ProjectError("Таблица уже существует; разметка не будет перезаписана.")
    rows = []
    for label in CLASSES:
        for image in sorted((folder / label).rglob("*")):
            if image.is_file() and image.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append({"path": image.resolve().relative_to(root.resolve()).as_posix(), "label": label})
    if not rows:
        raise ProjectError("Фотографии не найдены в подпапках android и iphone.")
    write_manifest(output, rows)
    return len(rows)


def difference_hash(image: Image.Image) -> str:
    """64-битный dHash — только подсказка для поиска похожих снимков.

    Он не обнаружит все обрезки, повороты и водяные знаки. Поэтому защита
    опирается прежде всего на ручные идентификаторы, а кандидаты проверяются.
    """
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return f"{result:016x}"


def groups(rows: list[dict]) -> list[list[int]]:
    """Union-find объединяет транзитивные связи устройства, сессии и дублей."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        for key in ("device_id", "session_id", "group_id", "sha256"):
            value = row.get(key)
            if not value:
                continue
            token = (key, value)
            if token in owners:
                parent[find(i)] = find(owners[token])
            else:
                owners[token] = i
    result = defaultdict(list)
    for i in range(len(rows)):
        result[find(i)].append(i)
    return list(result.values())


def assert_no_leakage(rows: list[dict]) -> None:
    for group in groups(rows):
        assigned = {rows[i]["split"] for i in group}
        if len(assigned) > 1:
            raise ProjectError("Утечка между частями: связанные фотографии " +
                               ", ".join(rows[i]["path"] for i in group[:4]))
    for part in SPLITS:
        labels = {r["label"] for r in rows if r["split"] == part}
        if labels != set(CLASSES):
            raise ProjectError(f"В части {part} должны присутствовать оба класса.")


def audit(rows: list[dict], root: Path, policy: dict, output: Path) -> dict:
    """Проверка файлов выполняется ДО обучения, хеши фиксируются в новой CSV."""
    if output.exists():
        raise ProjectError("Выходная таблица уже существует.")
    rows = [dict(row) for row in rows]
    labels_by_hash = defaultdict(set)
    labels_by_device = defaultdict(set)
    for row in rows:
        path = resolve(root, row["path"])
        image = read_rgb(path, policy)
        row["sha256"] = sha256(path)
        row["dhash"] = difference_hash(image)
        labels_by_hash[row["sha256"]].add(row["label"])
        labels_by_device[row["device_id"]].add(row["label"])
    if any(len(labels) > 1 for labels in labels_by_hash.values()):
        raise ProjectError("Один и тот же файл размечен разными классами.")
    if any(len(labels) > 1 for labels in labels_by_device.values()):
        raise ProjectError("Один device_id присвоен устройствам разных классов. Исправьте идентификатор или разметку.")
    # При 600 кадрах полный перебор пар невелик. Выходной отчёт предназначен
    # для человека: автоматически сливать визуально похожие разные телефоны нельзя.
    candidates = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= 4:
                candidates.append({"a": a["path"], "b": b["path"], "distance": distance,
                                   "cross_split": bool(a["split"] and b["split"] and a["split"] != b["split"])})
    write_manifest(output, rows)
    report = {"images": len(rows), "classes": dict(Counter(r["label"] for r in rows)),
              "connected_groups": len(groups(rows)),
              "exact_duplicate_extra_files": len(rows) - len(labels_by_hash),
              "near_duplicate_candidates": candidates,
              "note": "Проверьте похожие пары вручную; общий источник обозначьте одинаковым group_id."}
    write_json(output.with_suffix(".audit.json"), report)
    return report


def split_dataset(rows: list[dict], output: Path, seed: int = 42,
                  ratios: tuple[float, float, float] = (8/15, 2/15, 1/3),
                  trials: int = 2000) -> dict:
    """Ищем близкое к целевым размерам разбиение целых групп.

    Точные 160/40/100 на класс не всегда достижимы без разделения одного
    телефона. Из нескольких случайных назначений выбираем наиболее близкое
    к целям по каждому классу. Метрики нейросети при этом НЕ используются.
    """
    if output.exists():
        raise ProjectError("Разбиение уже существует. Сохраните новую версию под другим именем.")
    if any(r["split"] for r in rows):
        raise ProjectError("Для нового разбиения используйте таблицу с пустым split.")
    if any(not r["sha256"] for r in rows):
        raise ProjectError("Сначала выполните audit, чтобы зафиксировать хеши.")
    ratios_array = np.asarray(ratios, dtype=float)
    if np.any(ratios_array <= 0) or not np.isclose(ratios_array.sum(), 1):
        raise ProjectError("Три положительные доли разбиения должны давать сумму 1.")
    connected = groups(rows)
    if len(connected) < 3:
        raise ProjectError("Нужно минимум три независимые группы; сейчас их меньше.")
    counts = np.array([[sum(rows[i]["label"] == c for i in g) for c in CLASSES] for g in connected])
    targets = ratios_array[:, None] * counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    best_score = float("inf")
    # Пробуем и вероятностное назначение, и жадное заполнение дефицита.
    # Полностью воспроизводимо при том же seed, CSV и версии NumPy.
    for trial in range(trials):
        actual = np.zeros((3, 2), dtype=int)
        assigned = np.zeros(len(connected), dtype=int)
        for j in rng.permutation(len(connected)):
            if trial % 2:
                choices = []
                for part in range(3):
                    proposal = actual.copy()
                    proposal[part] += counts[j]
                    choices.append(float(np.sum(((proposal - targets) / np.maximum(targets, 1))**2)))
                part = int(np.argmin(np.asarray(choices) + rng.random(3) * 1e-8))
            else:
                part = int(rng.choice(3, p=ratios_array))
            assigned[j] = part
            actual[part] += counts[j]
        if np.any(actual == 0):
            continue
        score = float(np.sum(((actual - targets) / np.maximum(targets, 1))**2))
        if score < best_score:
            best, best_score = assigned.copy(), score
    if best is None:
        raise ProjectError("Не удалось разместить оба класса в трёх частях. Нужно больше независимых устройств/сессий.")
    result = [dict(row) for row in rows]
    for group, part in zip(connected, best):
        for i in group:
            result[i]["split"] = SPLITS[int(part)]
    assert_no_leakage(result)
    write_manifest(output, result)
    summary = {part: dict(Counter(r["label"] for r in result if r["split"] == part)) for part in SPLITS}
    report = {"seed": seed, "ratios": list(ratios), "counts": summary,
              "dataset_id": fingerprint(result), "groups": len(connected),
              "meets_original_test_size": all(summary["test"].get(c, 0) >= 100 for c in CLASSES),
              "note": "Размеры приблизительные: целостность устройств важнее точного числа кадров."}
    write_json(output.with_suffix(".summary.json"), report)
    return report
