"""Локальный визуальный энкодер MobileCLIP2-S0 и его безопасная загрузка.

В конфигурации OpenCLIP у S0 нет отдельной проекции после visual.trunk.
Поэтому timm fastvit_mci0 с выходом 512 численно воспроизводит encode_image.
Это позволяет не создавать текстовый энкодер и не включать OpenCLIP в рабочие
зависимости. Веса берутся из адаптации авторов timm с фиксированной ревизией.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np

from .common import ProjectError, read_json, sha256
from .images import preprocess, read_rgb


def make_visual():
    """pretrained=False принципиален: библиотека не скачивает весов сама."""
    import timm
    return timm.create_model("fastvit_mci0", pretrained=False, num_classes=512,
                             global_pool="avg", drop_path_rate=0.0)


def select_device(requested: str) -> str:
    import torch
    if requested not in ("auto", "cpu", "cuda"):
        raise ProjectError("Устройство должно быть auto, cpu или cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ProjectError("CUDA недоступна. Проверьте драйвер/сборку PyTorch или выберите --device cpu.")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


class Extractor:
    """Один экземпляр обслуживает все фотографии; модель загружается один раз."""

    def __init__(self, base: Path, preprocess_spec: dict, policy: dict,
                 device: str = "auto", threads: int = 4):
        # Дополнительная защита от неявной сети. Все пути загрузки ниже локальные.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        from safetensors.torch import load_file

        self.metadata = read_json(base / "base.json")
        if self.metadata.get("architecture") != "fastvit_mci0" or self.metadata.get("embedding_dim") != 512:
            raise ProjectError("Несовместимая архитектура базовой модели.")
        self.weights_hash = sha256(base / "visual.safetensors")
        if self.weights_hash != self.metadata["visual_sha256"]:
            raise ProjectError("Контрольная сумма визуального энкодера не совпадает.")
        torch.set_num_threads(max(1, int(threads)))
        self.device = select_device(device)
        self.model = make_visual()
        self.model.load_state_dict(load_file(str(base / "visual.safetensors")), strict=True)
        self.model.eval()  # BatchNorm не должен обновляться при извлечении признаков.
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.spec, self.policy = preprocess_spec, policy
        gc.collect()

    def encode_images(self, images: list) -> np.ndarray:
        """Нормировка L2 зафиксирована и при обучении, и при предсказании."""
        import torch
        if not images:
            raise ProjectError("Передан пустой пакет изображений.")
        values = np.stack([preprocess(image, self.spec) for image in images])
        batch = torch.from_numpy(values).to(self.device)
        # FP32 выбран для простой воспроизводимости между CPU и GPU. Для
        # маленькой модели это экономнее и надёжнее, чем усложнять код AMP.
        with torch.inference_mode():
            features = self.model(batch)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        result = features.cpu().numpy()
        if result.shape != (len(images), 512) or not np.isfinite(result).all():
            raise ProjectError("Некорректный выход визуального энкодера.")
        return result

    def encode_paths(self, paths: list[Path]) -> np.ndarray:
        images = [read_rgb(path, self.policy) for path in paths]
        return self.encode_images(images)
