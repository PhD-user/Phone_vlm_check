"""Независимая оценка классификатора и измерение ресурсов полного конвейера."""

from __future__ import annotations

import csv
import platform
import threading
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import CLASSES, ProjectError, environment, new_directory, now, resolve, write_json
from .dataset import read_manifest
from .features import load_features
from .inference import Recognizer, load_head
from .metrics import classification_metrics, probabilities


def evaluate(config: dict, root: Path, run_dir: Path, output: Path, split: str) -> dict:
    x, y, splits, features_meta = load_features(resolve(root, config["features"]), config, root)
    run, weights, bias = load_head(run_dir)
    if features_meta["fingerprint"] != run["features_fingerprint"]:
        raise ProjectError("Набор данных или кэш отличаются от версии обучения.")
    mask = splits == split
    probs = probabilities(x[mask], weights, bias)
    report = classification_metrics(y[mask], probs, run["confidence_threshold"])
    report.update(split=split, created_at=now(), model_version=run["model_version"],
                  dataset_id=run["dataset_id"], environment=environment())
    # Небольшой пробный тест может иметь идеальную accuracy, но не выполнить
    # приёмку из ТЗ. Размер теста включён в проверяемые условия явно.
    report["acceptance"] = {
        "is_final_test": split == "test",
        "at_least_100_per_class": all(v["support"] >= 100 for v in report["per_class"].values()),
        "balanced_accuracy_at_least_0_90": report["balanced_accuracy"] >= 0.90,
        "recall_each_at_least_0_85": all(v["recall"] >= 0.85 for v in report["per_class"].values())}
    report["acceptance"]["classification_passed"] = all(report["acceptance"].values())
    new_directory(output)
    write_json(output / "metrics.json", report)
    rows = [r for r in features_meta["contract"]["rows"] if r["split"] == split]
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true", "predicted", "confidence", "low_confidence", "p_android", "p_iphone"])
        for row, probability in zip(rows, probs):
            label = CLASSES[int(probability.argmax())]
            writer.writerow([row["path"], row["label"], label, float(probability.max()),
                             bool(probability.max() < run["confidence_threshold"]), *map(float, probability)])
    return report


def benchmark(config: dict, root: Path, run_dir: Path, output: Path,
              device: str, limit: int = 100) -> dict:
    """Меряем чтение+предобработку+модель+голову, без записи журнала и GUI.

    Память RSS отслеживается отдельным потоком каждые 10 мс. Это наблюдаемый
    пик процесса, не строгий максимум всех выделений и не измерение VRAM.
    Запускать на целевом ПК; результат контейнера его не заменяет.
    """
    import psutil
    rows = read_manifest(resolve(root, config["manifest"]), root,
                         require_split=True, verify_hashes=True)
    if limit < 1:
        raise ProjectError("limit должен быть положительным.")
    test = [r for r in rows if r["split"] == "test"][:limit]
    new_directory(output)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        start = perf_counter()
        recognizer = Recognizer(run_dir, config, root, device)
        load_seconds = perf_counter() - start
        if not test:
            raise ProjectError("Нет тестовых фотографий.")
        warmup = recognizer.predict(resolve(root, test[0]["path"]))
        if warmup["status"] == "error":
            raise ProjectError(warmup["message"])
        results = [recognizer.predict(resolve(root, row["path"])) for row in test]
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stopped.set()
        monitor.join()
    times = np.array([r["elapsed_seconds"] for r in results])
    ok = np.array([r["status"] != "error" for r in results])
    report = {"created_at": now(), "n": len(results), "device": recognizer.extractor.device,
              "model_version": recognizer.metadata["model_version"],
              "load_seconds": load_seconds, "median_seconds": float(np.median(times)),
              "p95_seconds": float(np.percentile(times, 95)), "max_seconds": float(times.max()),
              "technical_success_rate": float(ok.mean()),
              "successful_within_2s": int(np.sum(ok & (times <= 2))),
              "rss_peak_bytes_sampled": peak[0], "rss_growth_bytes_sampled": peak[0] - baseline,
              "environment": environment(), "platform": platform.platform(),
              "processor": platform.processor(), "cpu_logical_count": psutil.cpu_count(),
              "cpu_threads": config["cpu_threads"], "system_ram_bytes": psutil.virtual_memory().total,
              "measurement_note": "RSS процесса с шагом 10 мс; VRAM и процессы ОС отдельно не измеряются.",
              "acceptance_100_images": len(results) >= 100 and int(np.sum(ok & (times <= 2))) >= 95,
              "observed_process_rss_within_2gb": peak[0] <= 2 * 1024**3}
    write_json(output / "benchmark.json", report)
    write_json(output / "per_image.json", results)
    return report
