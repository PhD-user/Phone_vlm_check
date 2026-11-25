"""Метрики считают ошибки на ВСЕХ корректных кадрах выбранной части.

Низкая уверенность — отдельный флаг, а не способ убрать ошибки из accuracy.
Отдельно показываем долю уверенных ответов и точность только на них.
"""

from __future__ import annotations

import numpy as np
from .common import CLASSES


def probabilities(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = x @ weights + bias
    # Вычитание максимума не меняет softmax, но предотвращает переполнение exp.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    # Для простого predict эта зависимость не загружается.
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, log_loss, precision_recall_fscore_support)
    predicted = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    confident = probs.max(axis=1) >= threshold
    calibration_error = 0.0
    confidence = probs.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    for bin_index in range(10):
        mask = bins == bin_index
        if mask.any():
            calibration_error += float(mask.mean()) * abs(float((predicted[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return {"n": len(y), "labels": list(CLASSES),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "log_loss": float(log_loss(y, probs, labels=[0, 1])),
            "brier_iphone": float(np.mean((probs[:, 1] - y)**2)),
            "ece_10_bins": calibration_error,
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])} for i, c in enumerate(CLASSES)},
            "confidence_threshold": threshold, "confident_coverage": float(confident.mean()),
            "accuracy_on_confident": float((predicted[confident] == y[confident]).mean()) if confident.any() else None,
            "confidence_note": "Некалиброванная оценка softmax, не гарантированная вероятность правильности."}
