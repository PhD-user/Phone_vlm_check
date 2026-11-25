"""Извлечение и кэширование признаков с проверкой происхождения.

Кэш экономит время перебора параметров классификатора. Его нельзя применять
после замены фотографий, разметки, разбиения, предобработки или базовых весов.
Для этого сохраняем отпечатки всех перечисленных компонентов.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import CLASSES, ProjectError, fingerprint, now, resolve, sha256
from .dataset import assert_no_leakage, read_manifest


def feature_contract(rows: list[dict], config: dict, base_hash: str) -> dict:
    return {"rows": rows, "classes": list(CLASSES), "preprocess": config["preprocess"],
            "image_policy": config["image_policy"], "base_sha256": base_hash,
            "normalization": "l2", "schema_version": 1}


def extract_features(config: dict, root: Path, output: Path, device: str) -> dict:
    from .backbone import Extractor
    if output.exists():
        raise ProjectError("Кэш уже существует. Для другого эксперимента укажите новый --output.")
    rows = read_manifest(resolve(root, config["manifest"]), root,
                         require_split=True, verify_hashes=True)
    assert_no_leakage(rows)
    extractor = Extractor(resolve(root, config["base_dir"]), config["preprocess"],
                          config["image_policy"], device, config["cpu_threads"])
    batch_size = int(config["batch_size"])
    if batch_size < 1:
        raise ProjectError("batch_size должен быть положительным.")
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        chunks.append(extractor.encode_paths([resolve(root, row["path"]) for row in batch]))
        print(f"Признаки: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    contract = feature_contract(rows, config, extractor.weights_hash)
    meta = {"contract": contract, "fingerprint": fingerprint(contract),
            "created_at": now(), "device": extractor.device}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Unicode-строка JSON сохраняется без pickle. allow_pickle=False при чтении
    # исключает скрытые исполняемые Python-объекты из формата наших артефактов.
    with output.open("wb") as stream:
        np.savez_compressed(stream, x=np.concatenate(chunks),
                            y=np.array([CLASSES.index(r["label"]) for r in rows], dtype=np.int64),
                            splits=np.array([r["split"] for r in rows]),
                            metadata=np.array(json.dumps(meta, ensure_ascii=False)))
    return {"images": len(rows), "shape": [len(rows), 512], "fingerprint": meta["fingerprint"]}


def load_features(path: Path, config: dict, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rows = read_manifest(resolve(root, config["manifest"]), root,
                         require_split=True, verify_hashes=True)
    assert_no_leakage(rows)
    base_hash = sha256(resolve(root, config["base_dir"]) / "visual.safetensors")
    expected = feature_contract(rows, config, base_hash)
    with np.load(path, allow_pickle=False) as data:
        x, y, splits = data["x"].copy(), data["y"].copy(), data["splits"].copy()
        meta = json.loads(str(data["metadata"].item()))
    if meta["fingerprint"] != fingerprint(expected) or meta["contract"] != expected:
        raise ProjectError("Кэш признаков не соответствует данным/конфигурации. Повторите extract в новый файл.")
    expected_y = np.array([CLASSES.index(r["label"]) for r in rows])
    expected_splits = np.array([r["split"] for r in rows])
    if x.shape != (len(rows), 512) or not np.isfinite(x).all():
        raise ProjectError("Повреждён массив признаков.")
    if not np.array_equal(y, expected_y) or not np.array_equal(splits, expected_splits):
        raise ProjectError("Разметка кэша не соответствует таблице.")
    return x, y, splits, meta
