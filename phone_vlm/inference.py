"""Применение сохранённого эксперимента и журналирование результатов.

Файл run.json задаёт контракт модели. Значения preprocessing и порога берём
из эксперимента, поэтому последующее редактирование config.json не меняет
поведение уже обученной версии. config нужен только для расположения базы.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import CLASSES, DISPLAY, ProjectError, now, read_json, resolve, sha256
from .images import quality_warnings, read_rgb
from .metrics import probabilities


def load_head(run_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = read_json(run_dir / "run.json")
    if meta.get("schema_version") != 1 or meta.get("classes") != list(CLASSES):
        raise ProjectError("Несовместимый формат эксперимента или порядок классов.")
    if sha256(run_dir / "head.npz") != meta["head_sha256"]:
        raise ProjectError("Файл классификатора изменён или повреждён.")
    with np.load(run_dir / "head.npz", allow_pickle=False) as head:
        weights, bias = head["weights"].copy(), head["bias"].copy()
    if weights.shape != (512, 2) or bias.shape != (2,) or not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ProjectError("Неверные размеры или значения весов классификатора.")
    return meta, weights, bias


class Recognizer:
    def __init__(self, run_dir: Path, config: dict, root: Path, device: str = "cpu"):
        from .backbone import Extractor
        self.metadata, self.weights, self.bias = load_head(run_dir)
        self.extractor = Extractor(resolve(root, config["base_dir"]), self.metadata["preprocess"],
                                  self.metadata["image_policy"], device, config["cpu_threads"])
        if self.extractor.weights_hash != self.metadata["base_sha256"]:
            raise ProjectError("Эксперимент обучался с другим визуальным энкодером.")

    def predict(self, path: Path) -> dict:
        started = perf_counter()
        try:
            image = read_rgb(path, self.metadata["image_policy"])
            warnings = quality_warnings(image)
            features = self.extractor.encode_images([image])
            probs = probabilities(features, self.weights, self.bias)[0]
            selected = int(probs.argmax())
            confidence = float(probs[selected])
            low = confidence < self.metadata["confidence_threshold"]
            if low:
                warnings.append("Низкая уверенность результата.")
            result = {"status": "warning" if warnings else "ok", "filename": path.name,
                      "class": CLASSES[selected], "answer": DISPLAY[CLASSES[selected]],
                      "confidence": confidence, "confidence_percent": round(100 * confidence, 2),
                      "probabilities": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
                      "low_confidence": low, "warnings": warnings,
                      "confidence_calibrated": False}
        except (ProjectError, OSError) as exc:
            result = {"status": "error", "filename": path.name, "class": None,
                      "confidence": None, "message": str(exc)}
        result.update(timestamp=now(), model_version=self.metadata["model_version"],
                      elapsed_seconds=round(perf_counter() - started, 6))
        return result


def append_log(path: Path, result: dict) -> None:
    """JSON Lines: одна запись на запуск, без пикселей и полных путей к фото."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {key: result.get(key) for key in ("timestamp", "filename", "status", "class",
              "confidence", "low_confidence", "model_version", "elapsed_seconds", "message", "warnings")}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def predict_folder(recognizer: Recognizer, folder: Path, output: Path, log: Path) -> dict:
    """Одна ошибка не останавливает соседние файлы; модель общая на весь цикл."""
    if not folder.is_dir():
        raise ProjectError("Входная папка не найдена.")
    if output.exists():
        raise ProjectError("CSV уже существует, укажите новое имя.")
    # Выход и журнал должны быть снаружи: иначе они попадут в список входов
    # при следующем запуске и создадут неоднозначную процедуру испытаний.
    if output.resolve().parent == folder.resolve() or log.resolve().parent == folder.resolve():
        raise ProjectError("Результат и журнал сохраняйте вне входной папки.")
    files = sorted(path for path in folder.iterdir() if path.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["filename", "status", "class", "confidence", "model_version", "elapsed_seconds", "message", "warnings"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for path in files:
            result = recognizer.predict(path)
            append_log(log, result)
            errors += result["status"] == "error"
            row = {key: result.get(key, "") for key in fields}
            row["warnings"] = "; ".join(result.get("warnings", []))
            writer.writerow(row)
    return {"files": len(files), "errors": errors, "output": str(output)}
