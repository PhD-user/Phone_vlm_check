"""Командная строка проекта; импорты тяжёлых модулей отложены до команды.

Для обучения, тестирования и применения разные команды. Нельзя незаметно
переобучить модель обычным вызовом predict. Все артефакты пишутся отдельно
от исходных изображений; существующие эксперименты не затираются.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import __version__
from .common import ProjectError, environment, load_config, resolve


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Локальное распознавание iPhone и Android на MobileCLIP2-S0")
    p.add_argument("--config", help="Путь к config.json; задаётся ПЕРЕД названием команды")
    p.add_argument("--debug", action="store_true", help="Показать traceback ошибки")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Проверить зависимости, CUDA и базовые веса")
    download = sub.add_parser("download", help="Скачать и подготовить базовые веса, если их нет")
    download.add_argument("--keep-source", action="store_true", help="Оставить полный исходный checkpoint ~300 МБ")
    index = sub.add_parser("index", help="Создать черновик таблицы разметки")
    index.add_argument("--input", default="data/raw")
    index.add_argument("--output", default="data/labels.csv")
    audit = sub.add_parser("audit", help="Проверить изображения и найти похожие пары")
    audit.add_argument("--manifest", default="data/labels.csv")
    audit.add_argument("--output", default="data/audited.csv")
    split = sub.add_parser("split", help="Создать групповое train/val/test разбиение")
    split.add_argument("--manifest", default="data/audited.csv")
    split.add_argument("--output", default="data/splits.csv")
    split.add_argument("--ratios", type=float, nargs=3, default=[8/15, 2/15, 1/3], metavar=("TRAIN", "VAL", "TEST"))
    extract = sub.add_parser("extract", help="Извлечь признаки замороженного энкодера")
    extract.add_argument("--output", help="По умолчанию путь features из config")
    extract.add_argument("--device", choices=["auto", "cpu", "cuda"])
    train = sub.add_parser("train", help="Обучить классификатор по кэшу признаков")
    train.add_argument("--run", default="runs/experiment_001")
    evaluate = sub.add_parser("evaluate", help="Посчитать метрики на val или test")
    evaluate.add_argument("--run", default="runs/experiment_001")
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--output", default="reports/test_001")
    for name, help_text in [("predict", "Обработать одну фотографию"), ("batch", "Обработать файлы папки"),
                            ("gui", "Открыть локальное окно"), ("benchmark", "Измерить скорость и RSS процесса")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--run", default="runs/experiment_001")
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
        if name in ("predict", "batch", "gui"):
            command.add_argument("--log", default="reports/predictions.jsonl")
        if name == "predict":
            command.add_argument("image", help="Путь к изображению, можно абсолютный")
        elif name == "batch":
            command.add_argument("folder", help="Папка с входами, без рекурсивного обхода")
            command.add_argument("--output", default="reports/batch_001.csv")
        elif name == "benchmark":
            command.add_argument("--output", default="reports/benchmark_001")
            command.add_argument("--limit", type=int, default=100)
    return p


def doctor(config: dict, root: Path) -> dict:
    import platform
    info = environment()
    info.update(python=sys.version, platform=platform.platform(), base_available=False)
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    base = resolve(root, config["base_dir"])
    if (base / "base.json").is_file():
        from .backbone import Extractor
        extractor = Extractor(base, config["preprocess"], config["image_policy"], "cpu", config["cpu_threads"])
        from PIL import Image
        info["embedding_shape"] = list(extractor.encode_images([Image.new("RGB", (640, 480), "gray")]).shape)
        info["base_available"] = True
    else:
        info["next_step"] = "python -m phone_vlm download"
    info["phone_classifier_exists"] = (root / "runs/experiment_001/run.json").is_file()
    return info


def execute(args: argparse.Namespace) -> object:
    config, root = load_config(args.config)
    path = lambda value: resolve(root, value)
    if args.command == "doctor":
        return doctor(config, root)
    if args.command == "download":
        from .download import download_base
        return download_base(path(config["base_dir"]), keep_source=args.keep_source)
    if args.command in ("index", "audit", "split"):
        from .dataset import audit, index_folders, read_manifest, split_dataset
        if args.command == "index":
            return {"images": index_folders(root, path(args.input), path(args.output)),
                    "next_step": "Вручную заполните device_id и session_id в созданной CSV."}
        rows = read_manifest(path(args.manifest), root, verify_hashes=args.command == "split")
        if args.command == "audit":
            return audit(rows, root, config["image_policy"], path(args.output))
        return split_dataset(rows, path(args.output), config["seed"], tuple(args.ratios))
    if args.command == "extract":
        from .features import extract_features
        return extract_features(config, root, path(args.output or config["features"]), args.device or config["device"])
    if args.command == "train":
        from .training import train_head
        return train_head(config, root, path(args.run))
    if args.command in ("evaluate", "benchmark"):
        from .evaluation import benchmark, evaluate
        if args.command == "evaluate":
            return evaluate(config, root, path(args.run), path(args.output), args.split)
        return benchmark(config, root, path(args.run), path(args.output), args.device, args.limit)
    if args.command == "gui":
        from .gui import launch
        launch(path(args.run), config, root, args.device, path(args.log))
        return {"status": "closed"}
    from .inference import Recognizer, append_log, predict_folder
    recognizer = Recognizer(path(args.run), config, root, args.device)
    if args.command == "predict":
        result = recognizer.predict(path(args.image))
        append_log(path(args.log), result)
        return result
    return predict_folder(recognizer, path(args.folder), path(args.output), path(args.log))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("status") == "error" else 0
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        if isinstance(exc, FileNotFoundError):
            message = f"Файл не найден: {exc.filename}. Проверьте пути и выполненные шаги инструкции."
        elif isinstance(exc, ModuleNotFoundError):
            message = f"Не установлена зависимость {exc.name}. Используйте Python из .venv и requirements.txt."
        else:
            message = str(exc)
        print(json.dumps({"status": "error", "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
