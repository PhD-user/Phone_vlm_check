"""Обучение линейной головы; визуальная нейросеть здесь не переобучается.

Обучающие примеры используются в fit, валидационные — только для выбора C.
Тестовые признаки могут лежать в общем кэше, но тестовые метки не участвуют
ни в fit, ни в выборе C, ни в расчёте отчёта этой команды.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .common import (CLASSES, ProjectError, environment, fingerprint, new_directory,
                     now, resolve, sha256, write_json)
from .features import load_features
from .metrics import classification_metrics, probabilities


def train_head(config: dict, root: Path, output: Path) -> dict:
    x, y, splits, features_meta = load_features(resolve(root, config["features"]), config, root)
    train_mask, val_mask = splits == "train", splits == "val"
    settings = config["training"]
    threshold = float(settings["confidence_threshold"])
    if not 0.5 <= threshold <= 1:
        raise ProjectError("confidence_threshold должен находиться в диапазоне 0.5–1.")
    values = [float(value) for value in settings["C_grid"]]
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ProjectError("C_grid должен содержать положительные конечные числа.")
    if output.exists():
        raise ProjectError("Папка эксперимента уже существует. Укажите новую --run.")
    candidates, best = [], None
    for c in sorted(set(values)):
        classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=int(settings["max_iter"]),
                                        random_state=config["seed"], class_weight="balanced")
        # Недосошедшее обучение нельзя молча объявлять успешным.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                classifier.fit(x[train_mask], y[train_mask])
            except ConvergenceWarning as exc:
                raise ProjectError("Классификатор не сошёлся. Увеличьте max_iter в новой конфигурации.") from exc
        if classifier.classes_.tolist() != [0, 1]:
            raise ProjectError("В обучении должны присутствовать оба класса.")
        # sklearn хранит один logit z для класса 1. Два logit [-z/2, z/2]
        # дают точно ту же вероятность sigmoid(z), что позволяет применять
        # голову через NumPy без загрузки pickle/joblib и без scikit-learn.
        vector = classifier.coef_[0].astype(np.float64)
        intercept = float(classifier.intercept_[0])
        weights = np.stack([-vector / 2, vector / 2], axis=1)
        bias = np.array([-intercept / 2, intercept / 2])
        val_probs = probabilities(x[val_mask], weights, bias)
        if not np.allclose(val_probs, classifier.predict_proba(x[val_mask]), atol=1e-7):
            raise ProjectError("Ошибка преобразования классификатора в переносимый формат.")
        metrics = classification_metrics(y[val_mask], val_probs, threshold)
        candidates.append({"C": c, "validation": metrics})
        # При равной balanced accuracy предпочитаем меньший log_loss, затем
        # более сильную регуляризацию. Тестовая часть здесь недоступна отбору.
        rank = (metrics["balanced_accuracy"], -metrics["log_loss"], -c)
        if best is None or rank > best[0]:
            best = (rank, c, weights, bias, metrics)
        print(f"C={c:g}: val balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
    _, chosen_c, weights, bias, best_metrics = best
    new_directory(output)
    with (output / "head.npz").open("wb") as stream:
        np.savez_compressed(stream, weights=weights, bias=bias)
    code_hash = fingerprint({path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))})
    run = {"schema_version": 1, "model_version": output.name,
           "created_at": now(), "classes": list(CLASSES), "embedding_dim": 512,
           "architecture": "MobileCLIP2-S0 visual + L2 features + binary logistic regression",
           "training_mode": "frozen_visual_linear_probe", "selected_C": chosen_c,
           "confidence_threshold": threshold, "confidence_calibrated": False,
           "preprocess": config["preprocess"], "image_policy": config["image_policy"],
           "base_sha256": features_meta["contract"]["base_sha256"],
           "features_fingerprint": features_meta["fingerprint"],
           "dataset_id": fingerprint(features_meta["contract"]["rows"]),
           "manifest_sha256": sha256(resolve(root, config["manifest"])),
           "head_sha256": sha256(output / "head.npz"), "code_sha256": code_hash,
           "seed": config["seed"], "train_n": int(train_mask.sum()), "val_n": int(val_mask.sum()),
           "test_used_for_training_or_selection": False, "environment": environment()}
    write_json(output / "run.json", run)
    write_json(output / "config_snapshot.json", config)
    write_json(output / "training_metrics.json", {"candidates": candidates, "selected_C": chosen_c,
               "selected_validation": best_metrics, "test_metrics": None})
    # Точная таблица и метки эксперимента сохраняются для последующего аудита.
    from .dataset import write_manifest
    write_manifest(output / "dataset_snapshot.csv", features_meta["contract"]["rows"])
    return {"run": str(output), "selected_C": chosen_c, "validation": best_metrics,
            "note": "Это валидация. Итоговый тест запускается отдельной командой evaluate."}
