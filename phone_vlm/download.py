"""Явное получение базовой модели. Только эта команда обращается к сети.

Фиксируем ревизию и SHA-256 исходного safetensors. Из полного checkpoint
извлекается visual.trunk; текстовые веса в рабочую поставку не входят.
Исходный файл примерно 300 МБ временно сохраняется в models/download_cache.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from .common import ProjectError, environment, now, sha256, write_json

REPO = "timm/MobileCLIP2-S0-OpenCLIP"
REVISION = "095906d28bf54d7584dc411e8ffe448f34289e05"
SOURCE_SHA256 = "ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95"
APPLE_REVISION = "3136ea51c8ed56b9f9abfab04cb816735aaad6cb"


def fetch(url: str, destination: Path) -> None:
    """Незавершённая загрузка остаётся .part и не считается готовой моделью."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "phone-vlm-starter/0.1.0"})
    with urllib.request.urlopen(request, timeout=120) as remote, partial.open("wb") as local:
        shutil.copyfileobj(remote, local, length=1024 * 1024)
    partial.replace(destination)


def download_base(base: Path, *, keep_source: bool = False) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file
    from .backbone import make_visual

    if (base / "base.json").exists():
        from .common import read_json
        current = read_json(base / "base.json")
        if sha256(base / "visual.safetensors") != current["visual_sha256"]:
            raise ProjectError("Модель уже существует, но повреждена. Сохраните её отдельно перед повторным скачиванием.")
        return current
    base.mkdir(parents=True, exist_ok=True)
    cache = base.parent / "download_cache"
    source = cache / "open_clip_model.safetensors"
    if not source.exists() or sha256(source) != SOURCE_SHA256:
        print("Скачивание исходных весов MobileCLIP2-S0 (~300 МБ)...", flush=True)
        fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/open_clip_model.safetensors", source)
    if sha256(source) != SOURCE_SHA256:
        raise ProjectError("SHA-256 скачанного checkpoint не совпадает с зафиксированным.")

    # safetensors не исполняет произвольный Python-код, в отличие от pickle.
    # Достаём только нужные тензоры: полный текстовый энкодер не создаётся.
    visual = {}
    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            if key.startswith("visual.trunk."):
                visual[key.removeprefix("visual.trunk.")] = tensors.get_tensor(key).contiguous()
            elif key.startswith("visual."):
                raise ProjectError("У исходной модели появилась дополнительная визуальная проекция.")
    if not visual:
        raise ProjectError("В checkpoint не найдены visual.trunk веса.")
    probe = make_visual()
    probe.load_state_dict(visual, strict=True)  # Несовпадение слоёв — ошибка, не тихий пропуск.
    weights = base / "visual.safetensors"
    save_file(visual, str(weights), metadata={"source_repo": REPO, "source_revision": REVISION})
    fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/open_clip_config.json", base / "upstream_open_clip_config.json")
    fetch(f"https://huggingface.co/apple/MobileCLIP2-S0/resolve/{APPLE_REVISION}/LICENSE", base / "APPLE_MODEL_LICENSE.txt")
    metadata = {"schema_version": 1, "model": "MobileCLIP2-S0", "architecture": "fastvit_mci0",
                "embedding_dim": 512, "image_size": 256, "source_repo": REPO,
                "source_revision": REVISION, "source_sha256": SOURCE_SHA256,
                "visual_sha256": sha256(weights), "created_at": now(),
                "license": "apple-amlr; research and academic use only",
                "modification": "Only visual.trunk retained; text tower removed; tensor names stripped of visual.trunk prefix.",
                "environment": environment()}
    write_json(base / "base.json", metadata)
    (base / "ATTRIBUTION.txt").write_text(
        "Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.\n"
        "Source model: apple/MobileCLIP2-S0. Canonical conversion: timm/MobileCLIP2-S0-OpenCLIP.\n"
        "Modification: retained only visual.trunk, removed its prefix. No phone-task fine-tuning.\n"
        "Research and academic use only. See APPLE_MODEL_LICENSE.txt.\n", encoding="utf-8")
    if not keep_source:
        source.unlink()
    return metadata
